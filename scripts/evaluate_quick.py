"""Evaluate 23 unique routes at 50 fixed signal offsets across independent SUMO workers."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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

from scripts.evaluate import OUTCOMES, _plot_training_panels, _summarize, plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from algorithms.ppo import PPO
from env import road_builder
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo
from env.parallel_traffic import TrafficCurriculum
from test import load_saved_configs
from utils.evaluator import evaluate_parallel_routes


# 1.0, 3.5, ..., 123.5 s: samples green, yellow and all-red for every approach.
STEP = 0.5
# Each offset set has 50 ticks (0.5 s steps). development and holdout share the
# same (route, offset-order) departure-lane seed so the same start lane is used
# across the two sets and across models.
OFFSET_SETS = {
    "development": tuple(range(2, 250, 5)),   # 1.0, 3.5, ..., 123.5 s
    "holdout": tuple(range(4, 250, 5)),       # 2.0, 4.5, ..., 124.5 s
}
# Fixed evaluation seed, independent of the training seed, so all models are
# evaluated against the same departure lanes.
EVAL_SEED = 0


def _departure_seed(route_index, tick, offset_set):
    """Episode seed keyed by (eval seed, route index, offset order), not the tick
    value, so development and holdout share the same start lane per route."""
    return np.random.SeedSequence([EVAL_SEED, route_index,
                                   OFFSET_SETS[offset_set].index(int(tick))])


def _evaluate_worker(model, config, work, saved, vph, cases, depart_lane=None,
                     offset_set="development"):
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
                       fixed_traffic_vph=vph if curriculum else None)
    if env.prototype.step_length != STEP or env.signal_end_ticks[-1] != 250:
        raise ValueError("Quick evaluation requires a 125 s signal cycle and 0.5 s simulation steps")
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
        replica_seeds=[_departure_seed(route_index, tick, offset_set)
                       for route_index, _, tick in cases])
    return [{"route": route, **row} for (_, route, _), row in zip(cases, rows, strict=True)]


CLEARANCE_COLUMNS = (
    ("min_m", "Min"), ("q10_m", "P10"), ("q1_m", "Q1"),
    ("median_m", "Median"), ("q3_m", "Q3"), ("q90_m", "P90"),
    ("max_m", "Max"), ("mean_m", "Mean"), ("n", "n"),
)


def _red_wait_stats(rows, field="red_wait_clearance_m"):
    distances = [row[field] for row in rows if row[field] is not None]
    stats = {key: None for key, _ in CLEARANCE_COLUMNS}
    stats["n"] = len(distances)
    if distances:
        quantile_keys = [key for key, _ in CLEARANCE_COLUMNS[:7]]
        stats.update(zip(quantile_keys, np.percentile(distances, [0, 10, 25, 50, 75, 90, 100]).tolist()))
        stats["mean_m"] = float(np.mean(distances))
    return stats


def _plot_clearances(table, distribution, routes, summary):
    table.set_title("Red-light waiting clearance (m)", pad=28)
    table.set_xlim(-.5, len(CLEARANCE_COLUMNS) - .5)
    table.set_xticks([])
    for axis in (table, distribution):
        axis.tick_params(axis="y", left=False, labelleft=False)
    for x, (_, label) in enumerate(CLEARANCE_COLUMNS):
        table.text(x, 1.01, label, transform=table.get_xaxis_transform(),
                   ha="center", va="bottom", weight="bold", fontsize=8)

    by_route = [summary["by_route"][route]["red_wait_clearance"] for route in routes]
    all_stats = summary["all"]["red_wait_clearance"]
    distribution.set(title="Clearance distribution", xlabel="Distance (m)",
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


def _plot(run, out, routes, rows, summary, ticks):
    fig = plt.figure(figsize=(23, 16), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=(1, 1, 1, 3.3))
    _plot_training_panels(fig, grid, run)
    final = grid[3, :].subgridspec(1, 4, width_ratios=(1.0, 1.75, 1.75, 1.35))
    rates = fig.add_subplot(final[0, 0])
    offsets = fig.add_subplot(final[0, 1], sharey=rates)
    clearance = fig.add_subplot(final[0, 2], sharey=rates)
    distribution = fig.add_subplot(final[0, 3], sharey=rates)
    _plot_clearances(clearance, distribution, routes, summary)
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
        rates.text(len(ticks) + 1, i, f"{counts[route]['arrival']}/{len(ticks)}",
                   ha="left", va="center", fontsize=8)
    rates.set(xlim=(0, len(ticks) + 10), title="Final outcomes by route",
              xlabel="Episodes (one per signal offset)")
    rates.set_xticks(range(0, len(ticks) + 1, 10))
    rates.set_yticks(np.arange(len(routes) + 1), [*labels, "All routes"], fontsize=8)
    rates.grid(axis="x", alpha=.2)
    rates.set_axisbelow(True)

    codes = {key: i for i, key in enumerate(OUTCOMES)}
    index = {(row["route"], row["signal_offset_s"]): codes[row["status"]] for row in rows}
    matrix = [[index[route, tick * STEP] for tick in ticks] for route in routes]
    colors = [item[2] for item in OUTCOMES.values()]
    spacing = (ticks[1] - ticks[0]) * STEP
    offsets.imshow(matrix, cmap=ListedColormap(colors),
                   norm=BoundaryNorm(np.arange(len(colors) + 1) - .5, len(colors)),
                   interpolation="nearest", aspect="auto",
                   extent=(ticks[0] * STEP - spacing / 2, ticks[-1] * STEP + spacing / 2,
                           len(routes) - .5, -.5))
    offsets.set(xlim=(0, 125), ylim=(len(routes) + .5, -.5),
                title="Final outcome by route and signal start",
                xlabel="Signal offset at episode start (s)")
    offsets.set_xticks(np.arange(0, 126, 12.5))
    offsets.tick_params(axis="y", left=False, labelleft=False)
    for boundary in np.arange(.5, len(routes) - .5, 1):
        offsets.axhline(boundary, color="white", linewidth=.35, alpha=.7)
    for axis in (rates, offsets, clearance, distribution):
        axis.axhline(len(routes) - .5, color="#374151", linewidth=.8)
    handles = [Patch(color=color, label=label) for label, _, color in OUTCOMES.values()]
    offsets.legend(handles=handles, ncol=3, fontsize=8, loc="upper center",
                   bbox_to_anchor=(.5, -.10), frameon=False)
    total = summary["all"]["episodes"]
    arrived = summary["all"]["statuses"].get("arrival", 0)
    lane_mode = "sampled start lane" if summary["randomize_departures"] else "fixed start lane"
    fig.suptitle(f"{run.name} | Quick evaluation: {arrived}/{total} arrivals ({arrived / total:.1%})\n"
                 f"Deterministic policy · {lane_mode} · Central demand {summary['traffic_vph']} veh/h")
    fig.savefig(out / "learning_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="학습 결과 폴더 또는 model.pt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--traffic-vph", type=int, help="배경 교통이 있는 모델의 고정 중앙 투입량")
    parser.add_argument("--a2-ring-lane-sweep", action="store_true",
                        help="A2→A4 Ring 출발 차선별로 신호 시작 시점 50개를 추가 평가")
    parser.add_argument("--offset-set", choices=tuple(OFFSET_SETS), default="development",
                        help="평가할 신호 시작 시점 세트 (기본 development)")
    args = parser.parse_args()
    run = args.run.resolve()
    if run.is_file() and run.name == "model.pt":
        run = run.parent
    model = run / "model.pt"
    if not model.is_file():
        parser.error(f"모델을 찾을 수 없습니다: {model}")
    ticks = OFFSET_SETS[args.offset_set]
    if not 1 <= args.num_workers <= len(ticks):
        parser.error(f"--num-workers는 1~{len(ticks)} 사이여야 합니다.")
    road, _, _ = load_saved_configs(str(model))
    if road["NOHYEONG"]["all_signals_green"]:
        parser.error("빠른 신호 평가에는 신호가 작동하는 모델 설정이 필요합니다.")
    config_file = run / "parallel_config.json"
    saved = json.loads(config_file.read_text()) if config_file.is_file() else {}
    settings = saved.get("traffic_curriculum")
    if road["TRAFFIC"].get("background_enabled", False) and not settings:
        parser.error("배경 교통에는 저장된 커리큘럼 설정이 필요합니다.")
    maximum = settings["max_vph"] if settings else 0
    vph = min(800, maximum) if args.traffic_vph is None else args.traffic_vph
    if not 0 <= vph <= maximum:
        parser.error(f"--traffic-vph는 0~{maximum} 사이여야 합니다.")

    start = perf_counter()
    with TemporaryDirectory(prefix="sumo-quick-eval-") as directory:
        root = Path(directory)
        config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(root / "base"))
        non_ring_routes, ring_routes = read_ego_route_groups(root / "base/highway.ego.rou.xml")
        if (len(non_ring_routes), len(ring_routes)) != (18, 5):
            parser.error("평가에는 Ring 미경유 18개와 Ring 경유 5개의 고유 경로가 필요합니다.")
        routes = non_ring_routes + ring_routes
        chunks = np.array_split(ticks, args.num_workers)
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(_evaluate_worker, model, config, root / f"worker_{i}", saved, vph,
                                [(j, route, int(tick)) for j, route in enumerate(routes) for tick in chunk],
                                None, args.offset_set)
                    for i, chunk in enumerate(chunks)]
            rows = [row for job in jobs for row in job.result()]
            if args.a2_ring_lane_sweep:
                route = "ego_r_RING_A2_A4_A2_A4"
                cases = [(routes.index(route), route, tick) for tick in ticks]
                lane_jobs = [pool.submit(_evaluate_worker, model, config, root / f"lane_{lane}",
                                         saved, vph, cases, lane, args.offset_set)
                             for lane in range(4)]
                lane_rows = [row for job in lane_jobs for row in job.result()]
    rows.sort(key=lambda row: (row["route"], row["signal_offset_s"]))
    base = "quick" if args.offset_set == "development" else f"quick_{args.offset_set}"
    if args.traffic_vph is not None:
        base += f"_traffic_{vph}"
    out = run / "analysis" / base
    out.mkdir(parents=True, exist_ok=True)
    with (out / "route_evaluation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def summarize(group):
        green = [row["green_restart_delay_s"] for row in group if row["green_restart_observed"]]
        restarted = [delay for delay in green if delay is not None]
        return {**_summarize(group), "red_wait_clearance": _red_wait_stats(group),
                "first_signal_stop_clearance": _red_wait_stats(group, "first_signal_stop_clearance_m"),
                "green_restart": {
                    "observed": len(green), "no_restart": len(green) - len(restarted),
                    "median_delay_s": float(np.median(restarted)) if restarted else None,
                    "p90_delay_s": float(np.percentile(restarted, 90)) if restarted else None,
                }}

    if args.a2_ring_lane_sweep:
        with (out / "a2_ring_lane_evaluation.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(lane_rows[0]))
            writer.writeheader()
            writer.writerows(lane_rows)
    summary = {
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "phase": saved.get("phase"),
        "offset_set": args.offset_set,
        "num_workers": args.num_workers,
        "traffic_vph": vph,
        "deterministic": True,
        "randomize_departures": saved.get("randomize_departures", False),
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
    _plot(run, out, routes, rows, summary, ticks)
    summary["elapsed_seconds"] = round(perf_counter() - start, 2)
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    arrived = summary["all"]["statuses"].get("arrival", 0)
    _check_integrity(rows, routes, ticks)
    print(f"완주 {arrived}/{len(rows)} | {summary['elapsed_seconds']}초 | {out / 'learning_curve.png'}")


def _check_integrity(rows, routes, ticks):
    """Verify exactly one (route, offset) combination per route and offset set."""
    expected = {(route, tick * STEP) for route in routes for tick in ticks}
    observed = {(row["route"], row["signal_offset_s"]) for row in rows}
    if len(rows) != len(expected):
        raise RuntimeError(f"행 수 불일치: {len(rows)} != 기대 {len(expected)}")
    missing = expected - observed
    duplicated = len(rows) != len(observed)
    if missing or duplicated:
        raise RuntimeError(f"무결성 실패: 누락 {len(missing)}개, 중복 {duplicated}")


if __name__ == "__main__":
    main()
