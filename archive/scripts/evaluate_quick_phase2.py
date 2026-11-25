"""Evaluate a Phase 2 policy across fixed total background traffic levels."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from env import road_builder
from env.ego_routes import read_ego_route_groups
from env.parallel_traffic import TrafficCurriculum
from env.traffic_snapshots import TrafficSnapshotBank
from scripts.evaluate import OUTCOMES, _curve, _read_log, plt
from scripts.evaluate_quick_phase1 import STEP, TICKS, _evaluate_worker, _plot_final_panels, summarize
from test import load_saved_configs


def _heatmap(fig, ax, values, levels, labels, title, cmap, vmax, fmt):
    view = np.ma.masked_invalid(values)
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#e5e7eb")
    image = ax.imshow(view, cmap=color_map, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set(title=title, xlabel="Total requested vehicles/hour")
    ax.set_xticks(range(len(levels)), levels, rotation=45 if len(levels) > 8 else 0, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    fig.colorbar(image, ax=ax, shrink=.72, pad=.02)
    if len(levels) <= 20:
        for y, x in np.ndindex(values.shape):
            value = values[y, x]
            if np.isfinite(value):
                r, g, b, _ = image.cmap(image.norm(value))
                ink = "black" if .299 * r + .587 * g + .114 * b > .57 else "white"
                ax.text(x, y, format(value, fmt), ha="center", va="center", color=ink, fontsize=6)


def _plot_overview(run, out, routes, levels, summaries, *, training_plotter=None, policy_label="Phase 2"):
    fig = plt.figure(figsize=(29, 14), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, 4.2))
    if training_plotter is not None:
        training_plotter(fig, grid, run)
    else:
        training = _read_log(run / "training_log.csv")
        for i, (key, title, window) in enumerate((
            ("ep_rew_mean", "Training return (10-iteration mean)", 10),
            ("traffic_vph_mean", "Mean training demand (veh/h)", 1),
            ("value_loss", "Value loss (10-iteration mean)", 10),
            ("traffic_vph_p90", "90th percentile of training demand (veh/h)", 1),
            ("approx_kl", "Approximate KL", 1),
            ("p_lane_change", "Training lane-change probability", 1),
        )):
            _curve(fig.add_subplot(grid[i // 2, i % 2]), training, key, title, window)
    bottom = grid[3, :].subgridspec(1, 4, width_ratios=(1.15, 1.15, 1.15, 1.25))
    labels = [route.removeprefix("ego_d_").replace("ego_r_RING_", "Ring ") for route in routes]
    rates = [[summaries[level]["by_route"][route]["success_rate"] * 100 for level in levels]
             for route in routes]
    _heatmap(fig, fig.add_subplot(bottom[0]), np.array(rates), levels, labels,
             "Arrivals by route (%)", "YlGn", 100, ".0f")
    for column, reference, title in [(1, "stop_line", "Median stop-line distance (m)"),
                                     (2, "queued_vehicle", "Median stopped-leader gap (m)")]:
        medians = np.array([[summaries[level]["by_route"][route]["red_wait_clearance_by_reference"][reference]["median_m"]
                             for level in levels] for route in routes], dtype=float)
        finite = medians[np.isfinite(medians)]
        _heatmap(fig, fig.add_subplot(bottom[column]), medians, levels, labels,
                 title, "Blues", max(4., float(finite.max()) if finite.size else 0.), ".1f")

    side = bottom[3].subgridspec(3, 1)
    outcomes = fig.add_subplot(side[0])
    distance_axes = [fig.add_subplot(side[1]), fig.add_subplot(side[2])]
    x = np.arange(len(levels))
    totals = np.array([summaries[level]["all"]["episodes"] for level in levels])
    base = np.zeros(len(levels))
    for status, (label, _, color) in OUTCOMES.items():
        percent = 100 * np.array([summaries[level]["all"]["statuses"].get(status, 0)
                                  for level in levels]) / totals
        if np.any(percent):
            outcomes.bar(x, percent, bottom=base, color=color, label=label)
            base += percent
    outcomes.set(title="Outcomes across all routes", ylabel="Episodes (%)", ylim=(0, 105))
    outcomes.legend(fontsize=8, loc="lower left", ncol=2)
    outcomes.tick_params(axis="x", labelbottom=False)

    for distance, reference, title, color in zip(
            distance_axes, ("stop_line", "queued_vehicle"),
            ("Stop-line distance", "Stopped-leader gap"), ("#2563eb", "#d97706")):
        def clearances(key):
            return np.array([summaries[level]["all"]["red_wait_clearance_by_reference"][reference][key]
                             for level in levels], dtype=float)
        distance.fill_between(x, clearances("q10_m"), clearances("q90_m"), color=color, alpha=.12, label="P10–P90")
        distance.fill_between(x, clearances("q1_m"), clearances("q3_m"), color=color, alpha=.3, label="Q1–Q3")
        if len(levels) == 1:
            distance.vlines(x, clearances("q10_m"), clearances("q90_m"), color=color, alpha=.2, linewidth=8)
            distance.vlines(x, clearances("q1_m"), clearances("q3_m"), color=color, alpha=.5, linewidth=8)
        distance.plot(x, clearances("median_m"), color=color, marker=".", label="Median")
        counts = [summaries[level]["all"]["red_wait_clearance_by_reference"][reference]["n"] for level in levels]
        distance.set(title=title + " (last red stop/episode)", ylabel="Distance (m)",
                     xlabel="Total requested vehicles/hour")
        distance.set_xticks(x, [f"{level}\nn={n}" for level, n in zip(levels, counts)], fontsize=8)
        distance.legend(fontsize=7)
    fig.suptitle(f"{run.name} | {policy_label} quick evaluation · {len(levels)} traffic levels · "
                 f"{len(routes) * len(TICKS):,} episodes per level")
    fig.savefig(out / "learning_curve.png", dpi=150)
    plt.close(fig)


def _plot_level(out, vph, routes, rows, summary, *, policy_label="Phase 2"):
    fig = plt.figure(figsize=(34, 9), layout="constrained")
    grid = fig.add_gridspec(1, 1)
    _plot_final_panels(fig, grid[0], routes, rows, summary)
    all_stats = summary["all"]
    arrivals = all_stats["statuses"].get("arrival", 0)
    fig.suptitle(f"{policy_label} · {vph:,} requested vehicles/hour · {arrivals:,}/{len(rows):,} arrivals")
    fig.savefig(out / f"vph_{vph:04d}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Phase 2 결과 폴더 또는 model.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    traffic = parser.add_mutually_exclusive_group()
    traffic.add_argument("--traffic-step", type=int, default=100, help="교통량 평가 간격 (기본 100대/시간)")
    traffic.add_argument("--traffic-vph", type=int, help="지정한 교통량 하나만 평가")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.is_file() and run.name == "model.pt":
        run = run.parent
    model = run / "model.pt"
    config_path = run / "parallel_config.json"
    if not model.is_file() or not config_path.is_file():
        parser.error("model.pt와 parallel_config.json이 있는 결과 폴더가 필요합니다.")
    if not 1 <= args.num_workers <= len(TICKS):
        parser.error(f"--num-workers는 1~{len(TICKS)} 사이여야 합니다.")
    if args.traffic_step <= 0:
        parser.error("--traffic-step는 양수여야 합니다.")

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    settings = saved.get("traffic_curriculum")
    if saved.get("phase") != 2 or not settings:
        parser.error("Phase 2 모델을 지정하세요.")
    curriculum = TrafficCurriculum(**settings)
    if not (curriculum.total_vph and curriculum.snapshot_warmup_seconds):
        parser.error("전체 투입량과 초기 교통 상태를 사용하는 Phase 2 모델이 필요합니다.")
    if args.traffic_vph is not None:
        if not 0 <= args.traffic_vph <= curriculum.max_vph:
            parser.error(f"--traffic-vph는 0~{curriculum.max_vph} 사이여야 합니다.")
        levels = [args.traffic_vph]
    else:
        if (curriculum.max_vph + args.traffic_step - 1) // args.traffic_step + 1 > 31:
            parser.error("한 번에 최대 31개 교통량을 평가할 수 있도록 --traffic-step을 늘려주세요.")
        levels = sorted({*range(0, curriculum.max_vph + 1, args.traffic_step), curriculum.max_vph})

    start = perf_counter()
    out_name = "quick_phase2"
    if args.traffic_vph is not None:
        out_name += f"_traffic_{args.traffic_vph}"
    elif args.traffic_step != 100:
        out_name += f"_step_{args.traffic_step}"
    out = run / "analysis" / out_name
    details = out / "details"
    details.mkdir(parents=True, exist_ok=True)
    road, _, _ = load_saved_configs(str(model))
    rows = {level: [] for level in levels}
    summaries = {}

    with TemporaryDirectory(prefix="sumo-quick-phase2-") as directory:
        root = Path(directory)
        config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(root / "base"))
        direct, ring = read_ego_route_groups(root / "base/highway.ego.rou.xml")
        if (len(direct), len(ring)) != (18, 5):
            parser.error("평가에는 Ring 미경유 18개와 Ring 경유 5개의 고유 경로가 필요합니다.")
        routes = direct + ring
        # Reuse one frozen bank so different evaluation workers see identical traffic at a given tick.
        positive = tuple(level for level in levels if level)
        bank = (TrafficSnapshotBank(config, curriculum, road["TRAFFIC"]["vehs_per_hour"],
                                    positive, STEP, 0, refresh=False) if positive else None)
        try:
            chunks = np.array_split(TICKS, max(2, args.num_workers))
            with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as pool:
                jobs = {
                    pool.submit(_evaluate_worker, model, config, root / f"vph_{level}_{i}", saved, level,
                                [(j, route, int(tick)) for j, route in enumerate(routes) for tick in chunk],
                                snapshot_path=bank.path if bank is not None else None): level
                    for level in levels for i, chunk in enumerate(chunks)
                }
                for job in as_completed(jobs):
                    level = jobs[job]
                    rows[level].extend(job.result())
                    if len(rows[level]) == len(routes) * len(TICKS):
                        rows[level].sort(key=lambda row: (row["route"], row["signal_offset_s"]))
                        summaries[level] = {
                            "all": summarize(rows[level]),
                            "by_route": {route: summarize([r for r in rows[level] if r["route"] == route])
                                         for route in routes},
                        }
                        arrived = summaries[level]["all"]["statuses"].get("arrival", 0)
                        print(f"{level:4}대/시간 | 완주 {arrived}/{len(rows[level])}", flush=True)
        finally:
            if bank is not None:
                bank.close()

    _plot_overview(run, out, routes, levels, summaries)
    for level in levels:
        _plot_level(details, level, routes, rows[level], summaries[level])
    with (out / "route_evaluation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[levels[0]][0]))
        writer.writeheader()
        writer.writerows(row for level in levels for row in rows[level])
    summary = {
        "phase": 2,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "traffic_levels_vph": levels,
        "traffic_vph_definition": "Total requested background vehicles per hour, including ring traffic.",
        "randomize_departures": saved.get("randomize_departures", False),
        "signal_offsets_s": [tick * STEP for tick in TICKS],
        "deterministic": True,
        "elapsed_seconds": round(perf_counter() - start, 2),
        "by_traffic_vph": {level: summaries[level] for level in levels},
    }
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{summary['elapsed_seconds']}초 | {out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
