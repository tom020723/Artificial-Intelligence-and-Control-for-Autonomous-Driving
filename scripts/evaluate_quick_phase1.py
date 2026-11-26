"""Evaluate 23 unique routes at fixed offsets across the actual signal cycle."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scripts.evaluate import OUTCOMES, _plot_training_panels, _summarize, plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from algorithms.ppo import PPO
from env import road_builder
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo, _signal_schedule
from env.parallel_traffic import TrafficCurriculum
from test import load_saved_configs
from utils.evaluator import evaluate_parallel_routes


STEP = 0.5


def signal_offsets(config):
    """Sample every 2.5 s within the generated network's actual signal cycle."""
    config = Path(config)
    network = ET.parse(config).find("input/net-file").get("value")
    _, ends = _signal_schedule(ET.parse(config.parent / network).getroot(), STEP, True)
    return int(ends[-1]), tuple(range(2, int(ends[-1]), 5))


def _evaluate_worker(model, config, work, saved, vph, cases, depart_lane=None, snapshot_path=None):
    import torch

    torch.set_num_threads(1)
    road, mdp, train = load_saved_configs(str(model))
    settings = saved.get("traffic_curriculum")
    curriculum = TrafficCurriculum(**settings) if settings else None
    kwargs = dict(cfg_path=config, road=road["ROAD"], ego=road["EGO"],
                  mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
                  action=mdp["ACTION"], reward=train["REWARD"], traffic=road["TRAFFIC"])
    env = ParallelSumo(len(cases), kwargs, work, seed=0, randomize_signals=True,
                       randomize_departures=depart_lane is not None or saved.get("randomize_departures", False),
                       independent_episode_rng=saved.get("independent_episode_rng", False),
                       traffic_curriculum=curriculum,
                       fixed_traffic_vph=vph if curriculum else None,
                       traffic_snapshot_path=snapshot_path)
    if env.prototype.step_length != STEP:
        env.close()
        raise ValueError("Quick evaluation requires 0.5 s simulation steps")
    agent = PPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim,
                **(train["HPARAMS"] | {"device": "cpu"}))
    agent.policy.load_state_dict(torch.load(model, map_location="cpu", weights_only=True))
    if depart_lane is not None:
        env._find_departures()
        for route in {route for _, route, _ in cases}:
            lanes = tuple(item for item in env.departures[route] if item[2] == depart_lane)
            if not lanes:
                raise ValueError(f"{route}: unavailable departure lane {depart_lane}")
            env.departures[route] = lanes
    rows = evaluate_parallel_routes(
        agent.policy, env, agent.device, [route for _, route, _ in cases],
        signal_ticks=[tick for _, _, tick in cases],
        replica_seeds=[np.random.SeedSequence([0, route_index, tick])
                       for route_index, _, tick in cases])
    return [{"route": route, **row} for (_, route, _), row in zip(cases, rows, strict=True)]


CLEARANCE_COLUMNS = (
    ("min_m", "Min"), ("q10_m", "P10"), ("q1_m", "Q1"),
    ("median_m", "Median"), ("q3_m", "Q3"), ("q90_m", "P90"),
    ("max_m", "Max"), ("mean_m", "Mean"), ("n", "n"),
)


def _red_wait_stats(rows, field="red_wait_clearance_m", reference=None):
    distances = [row[field] for row in rows if row[field] is not None
                 and (reference is None or row.get("red_wait_reference") == reference)]
    stats = {key: None for key, _ in CLEARANCE_COLUMNS}
    stats["n"] = len(distances)
    if distances:
        quantile_keys = [key for key, _ in CLEARANCE_COLUMNS[:7]]
        stats.update(zip(quantile_keys, np.percentile(distances, [0, 10, 25, 50, 75, 90, 100]).tolist()))
        stats["mean_m"] = float(np.mean(distances))
    return stats


def summarize(group):
    green = [row["green_restart_delay_s"] for row in group if row["green_restart_observed"]]
    restarted = [delay for delay in green if delay is not None]
    return {**_summarize(group), "red_wait_clearance": _red_wait_stats(group),
            "red_wait_clearance_by_reference": {reference: _red_wait_stats(group, reference=reference)
                                                 for reference in ("stop_line", "queued_vehicle")},
            "first_signal_stop_clearance": _red_wait_stats(group, "first_signal_stop_clearance_m"),
            "green_restart": {
                "observed": len(green), "no_restart": len(green) - len(restarted),
                "median_delay_s": float(np.median(restarted)) if restarted else None,
                "p90_delay_s": float(np.percentile(restarted, 90)) if restarted else None,
            }}


def _plot_clearances(table, distribution, routes, summary, reference):
    title = "Stop-line distance" if reference == "stop_line" else "Stopped-leader gap"
    table.set_title(title + " (m)", pad=28)
    table.set_xlim(-.5, len(CLEARANCE_COLUMNS) - .5)
    table.set_xticks([])
    for axis in (table, distribution):
        axis.tick_params(axis="y", left=False, labelleft=False)
    for x, (_, label) in enumerate(CLEARANCE_COLUMNS):
        table.text(x, 1.01, label, transform=table.get_xaxis_transform(),
                   ha="center", va="bottom", weight="bold", fontsize=8)

    by_route = [summary["by_route"][route]["red_wait_clearance_by_reference"][reference] for route in routes]
    all_stats = summary["all"]["red_wait_clearance_by_reference"][reference]
    distribution.set(title=title + " distribution", xlabel="Distance (m)",
                     xlim=(0, max(1.0, (all_stats["max_m"] or 0.0) * 1.05)))
    distribution.grid(axis="x", alpha=.2)
    distribution.set_axisbelow(True)
    for y, stats in enumerate([*by_route, all_stats]):
        if y % 2 == 0 or y == len(routes):
            for axis in (table, distribution):
                axis.axhspan(y - .5, y + .5, color="#e2e8f0" if y == len(routes) else "#f2f4f7",
                            zorder=0)
        for x, (key, _) in enumerate(CLEARANCE_COLUMNS):
            value = stats[key]
            label = str(value) if key == "n" else ("—" if value is None else f"{value:.1f}")
            table.text(x, y, label, ha="center", va="center", fontsize=8)
        if stats["n"]:
            distribution.hlines(y, stats["min_m"], stats["max_m"], color="#64748b", linewidth=1)
            distribution.hlines(y, stats["q10_m"], stats["q90_m"], color="#60a5fa", linewidth=3)
            distribution.hlines(y, stats["q1_m"], stats["q3_m"], color="#2563eb", linewidth=6)
            distribution.scatter(stats["median_m"], y, marker="D", s=19,
                                 color="#172554", edgecolors="white", linewidths=.6, zorder=4)
    distribution.legend(handles=[
        Line2D([], [], color="#64748b", label="Min–Max"),
        Line2D([], [], color="#60a5fa", linewidth=3, label="P10–P90"),
        Line2D([], [], color="#2563eb", linewidth=6, label="Q1–Q3"),
        Line2D([], [], color="#172554", marker="D", linestyle="none", label="Median"),
    ], ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.10), frameon=False)


def _plot_final_panels(fig, spec, routes, rows, summary):
    offsets_s = sorted({row["signal_offset_s"] for row in rows})
    count = len(offsets_s)
    final = spec.subgridspec(1, 6, width_ratios=(1.0, 1.75, 1.75, 1.35, 1.75, 1.35))
    rates = fig.add_subplot(final[0, 0])
    offsets = fig.add_subplot(final[0, 1], sharey=rates)
    clearance = fig.add_subplot(final[0, 2], sharey=rates)
    distribution = fig.add_subplot(final[0, 3], sharey=rates)
    _plot_clearances(clearance, distribution, routes, summary, "stop_line")
    queue_table = fig.add_subplot(final[0, 4], sharey=rates)
    queue_distribution = fig.add_subplot(final[0, 5], sharey=rates)
    _plot_clearances(queue_table, queue_distribution, routes, summary, "queued_vehicle")
    labels = [route.removeprefix("ego_d_").replace("ego_r_RING_", "Ring ") for route in routes]
    counts = {route: Counter() for route in routes}
    for row in rows:
        counts[row["route"]][row["status"]] += 1
    y = np.arange(len(routes))
    left = np.zeros(len(routes))
    for key, (label, _, color) in OUTCOMES.items():
        values = np.array([counts[route][key] for route in routes])
        rates.barh(y, values, left=left, height=.72, color=color)
        left += values
    for i, route in enumerate(routes):
        rates.text(count + 1, i, f"{counts[route]['arrival']}/{count}",
                   ha="left", va="center", fontsize=8)
    rates.set(xlim=(0, count + 10), title="Final outcomes by route",
              xlabel="Episodes (one per signal offset)")
    rates.set_xticks(range(0, count + 1, 10))
    rates.set_yticks(np.arange(len(routes) + 1), [*labels, "All routes"], fontsize=8)
    rates.grid(axis="x", alpha=.2)
    rates.set_axisbelow(True)

    codes = {key: i for i, key in enumerate(OUTCOMES)}
    index = {(row["route"], row["signal_offset_s"]): codes[row["status"]] for row in rows}
    matrix = [[index[route, offset] for offset in offsets_s] for route in routes]
    colors = [item[2] for item in OUTCOMES.values()]
    spacing = offsets_s[1] - offsets_s[0] if count > 1 else STEP
    offsets.imshow(matrix, cmap=ListedColormap(colors),
                   norm=BoundaryNorm(np.arange(len(colors) + 1) - .5, len(colors)),
                   interpolation="nearest", aspect="auto",
                   extent=(offsets_s[0] - spacing / 2, offsets_s[-1] + spacing / 2,
                           len(routes) - .5, -.5))
    offsets.set(xlim=(0, offsets_s[-1] + spacing / 2), ylim=(len(routes) + .5, -.5),
                title="Final outcome by route and signal start",
                xlabel="Signal offset at episode start (s)")
    offsets.set_xticks(np.linspace(0, offsets_s[-1] + spacing / 2, 11))
    offsets.tick_params(axis="y", left=False, labelleft=False)
    for boundary in np.arange(.5, len(routes) - .5, 1):
        offsets.axhline(boundary, color="white", linewidth=.35, alpha=.7)
    for axis in (rates, offsets, clearance, distribution, queue_table, queue_distribution):
        axis.axhline(len(routes) - .5, color="#374151", linewidth=.8)
    handles = [Patch(color=color, label=label) for label, _, color in OUTCOMES.values()]
    offsets.legend(handles=handles, ncol=3, fontsize=8, loc="upper center",
                   bbox_to_anchor=(.5, -.10), frameon=False)


def _plot(run, out, routes, rows, summary):
    fig = plt.figure(figsize=(34, 16), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, 3.3))
    _plot_training_panels(fig, grid, run)
    _plot_final_panels(fig, grid[3, :], routes, rows, summary)
    total = summary["all"]["episodes"]
    arrived = summary["all"]["statuses"].get("arrival", 0)
    lane_mode = "sampled start lane" if summary["randomize_departures"] else "fixed start lane"
    fig.suptitle(f"{run.name} | Quick evaluation: {arrived}/{total} arrivals ({arrived / total:.1%})\n"
                 f"Deterministic policy · {lane_mode} · No background traffic")
    fig.savefig(out / "learning_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="학습 결과 폴더 또는 model.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--a2-ring-lane-sweep", action="store_true",
                        help="A2→A4 Ring 출발 차선별로 한 주기의 신호 시작 시점을 추가 평가")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.is_file() and run.name == "model.pt":
        run = run.parent
    model = run / "model.pt"
    if not model.is_file():
        parser.error(f"모델을 찾을 수 없습니다: {model}")
    if args.num_workers < 1:
        parser.error("--num-workers는 양수여야 합니다.")
    road, _, _ = load_saved_configs(str(model))
    if road["NOHYEONG"]["all_signals_green"]:
        parser.error("빠른 신호 평가에는 신호가 작동하는 모델 설정이 필요합니다.")
    config_file = run / "parallel_config.json"
    saved = json.loads(config_file.read_text()) if config_file.is_file() else {}
    if road["TRAFFIC"].get("background_enabled", False) or saved.get("phase") == 2:
        parser.error("Phase 2 모델은 scripts/evaluate_quick_phase2.py로 평가하세요.")
    vph = 0

    start = perf_counter()
    with TemporaryDirectory(prefix="sumo-quick-eval-") as directory:
        root = Path(directory)
        config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(root / "base"))
        cycle_ticks, ticks = signal_offsets(config)
        non_ring_routes, ring_routes = read_ego_route_groups(root / "base/highway.ego.rou.xml")
        if (len(non_ring_routes), len(ring_routes)) != (18, 5):
            parser.error("평가에는 Ring 미경유 18개와 Ring 경유 5개의 고유 경로가 필요합니다.")
        routes = non_ring_routes + ring_routes
        chunks = np.array_split(ticks, min(len(ticks), args.num_workers))
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(_evaluate_worker, model, config, root / f"worker_{i}", saved, vph,
                                [(j, route, int(tick)) for j, route in enumerate(routes) for tick in chunk])
                    for i, chunk in enumerate(chunks)]
            rows = [row for job in jobs for row in job.result()]
            if args.a2_ring_lane_sweep:
                route = "ego_r_RING_A2_A4_A2_A4"
                cases = [(routes.index(route), route, tick) for tick in ticks]
                lane_jobs = [pool.submit(_evaluate_worker, model, config, root / f"lane_{lane}",
                                         saved, vph, cases, lane) for lane in range(4)]
                lane_rows = [row for job in lane_jobs for row in job.result()]
    rows.sort(key=lambda row: (row["route"], row["signal_offset_s"]))
    out = run / "analysis" / "quick"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "route_evaluation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.a2_ring_lane_sweep:
        with (out / "a2_ring_lane_evaluation.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(lane_rows[0]))
            writer.writeheader()
            writer.writerows(lane_rows)
    summary = {
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "phase": saved.get("phase"),
        "num_workers": args.num_workers,
        "traffic_vph": vph,
        "deterministic": True,
        "randomize_departures": saved.get("randomize_departures", False),
        "signal_cycle_s": cycle_ticks * STEP,
        "signal_offsets_s": [tick * STEP for tick in ticks],
        "red_wait_definition": "Last measurable stop per episode with speed <= 0.5 m/s "
                               "and nearest observed signal red; nearest stop line or stopped leader.",
        "first_signal_stop_definition": "First measurable stop per episode with speed <= 0.5 m/s "
                                        "and nearest observed signal yellow or red.",
        "green_restart_definition": "From the first observed green for the same signal link after a red stop "
                                    "(speed <= 0.5 m/s), count time until speed exceeds 0.5 m/s or green is crossed. "
                                    "No restart if the green window, signal observation, or episode ends first.",
        "all": summarize(rows),
        "by_route": {route: summarize([row for row in rows if row["route"] == route])
                     for route in routes},
    }
    if args.a2_ring_lane_sweep:
        summary["a2_ring_by_start_lane"] = {
            str(lane): summarize([row for row in lane_rows if row["start_lane"] == lane])
            for lane in range(4)
        }
    _plot(run, out, routes, rows, summary)
    summary["elapsed_seconds"] = round(perf_counter() - start, 2)
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    arrived = summary["all"]["statuses"].get("arrival", 0)
    print(f"완주 {arrived}/{len(rows)} | {summary['elapsed_seconds']}초 | {out / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
