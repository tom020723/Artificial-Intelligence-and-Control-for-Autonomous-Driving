"""Collect native IDM demonstrations with fixed center starts, then fit and evaluate BC."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_background_data import action_from_speeds
from env import road_builder
from env.background_demand import set_weighted_od_flow_rates
from env.collision_detection import configure_collision_gaps
from env.control_timing import lane_change_timing_allowed
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo, libsumo
from env.parallel_traffic import TrafficCurriculum
from env.traffic_snapshots import TrafficSnapshotBank
from scripts.evaluate_quick_bc import evaluate, prepare_context
from scripts.evaluate_quick_phase1 import STEP, TICKS
from test import load_saved_configs

TRAFFIC_VPH = 6000
PROTECTED_DISTANCE = 30.0
EXPERT_TYPE = "car_idm"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixed_start_context(run, context, *args, **kwargs):
    saved, curriculum, dataset = prepare_context(run, context, *args, **kwargs)
    with (context / "road_config.py").open("a") as file:
        file.write("\nEGO = dict(EGO, depart_lane='center')\n")
    with (context / "mdp_config.py").open("a") as file:
        file.write("\nACTION['lane_change_safety'].update("
                   "enabled=False, spawn_protect_only=False, spawn_protect=0.0)\n")
    saved.update(randomize_departures=False, fixed_departure="center",
                 spawn_protect_m=0.0, target_lane_safety_enabled=False)
    save_json(context / "evaluation_config.json", saved)
    return saved, curriculum, dataset


class ExpertSumo(ParallelSumo):
    def _respawn(self, ego, route=None, **kwargs):
        super()._respawn(ego, route, **kwargs)
        vehicle = libsumo.vehicle
        vehicle.setType(ego.ego_id, EXPERT_TYPE)
        vehicle.setSpeedMode(ego.ego_id, 31)
        vehicle.setSpeed(ego.ego_id, -1)
        vehicle.setLaneChangeMode(ego.ego_id, 0)
        return ego._get_obs()


def collect(worker, cases, context_model, cfg, settings, snapshot, output, replicas):
    road, mdp, train = load_saved_configs(str(context_model))
    env = ExpertSumo(min(replicas, len(cases)), dict(
        cfg_path=cfg, road=road["ROAD"], ego=road["EGO"], traffic=road["TRAFFIC"],
        mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"], action=mdp["ACTION"],
        reward=train["REWARD"]), output / "network", seed=worker,
        randomize_signals=True, randomize_departures=False, independent_episode_rng=True,
        traffic_curriculum=TrafficCurriculum(**settings), fixed_traffic_vph=TRAFFIC_VPH,
        traffic_snapshot_path=snapshot)
    vehicle = libsumo.vehicle
    queue = iter(cases)
    active = {i: next(queue) for i in range(env.num_envs)}
    buffers = {i: [] for i in active}
    completed = []
    observations, actions, episodes = [], [], []
    try:
        obs = env.reset([c["route"] for c in active.values()],
                        signal_ticks=[c["tick"] for c in active.values()],
                        replica_seeds=[np.random.SeedSequence(c["seed"]) for c in active.values()])
        assert obs.shape == (env.num_envs, 51)
        while active:
            now = libsumo.simulation.getTime()
            before = {}
            for i in active:
                ego = env.egos[i]
                vid = ego.ego_id
                edge = vehicle.getRoadID(vid)
                allowed = lane_change_timing_allowed(
                    edge, vehicle.getLanePosition(vid), vehicle.getRouteIndex(vid), PROTECTED_DISTANCE)
                vehicle.setLaneChangeMode(vid, 1621 if allowed else 0)
                before[i] = (edge, vehicle.getLaneIndex(vid), vehicle.getSpeed(vid), allowed)
                ego.background.spawn_due(libsumo, now, ego.prefix, ego.traffic_random)
            libsumo.simulationStep()
            alive = set(vehicle.getIDList())
            env._live_background = alive | set(libsumo.simulation.getPendingVehicles())
            arrived = set(libsumo.simulation.getArrivedIDList())
            collided = set(libsumo.simulation.getCollidingVehiclesIDList())
            teleported = set(libsumo.simulation.getStartingTeleportIDList())
            for i, case in list(active.items()):
                ego = env.egos[i]
                vid = ego.ego_id
                ego.step_count += 1
                ego.present = vid in alive
                edge, lane, speed, allowed = before[i]
                if ego.present and vid not in collided | teleported and vehicle.getRoadID(vid) == edge:
                    speed_after = vehicle.getSpeed(vid)
                    direction = vehicle.getLaneIndex(vid) - lane
                    accel, _, saturated = action_from_speeds(
                        speed, speed_after, ego.step_length, ego.max_accel, ego.max_decel)
                    if direction and not allowed:
                        raise RuntimeError("Expert changed lane within the protected region")
                    if abs(direction) <= 1 and not saturated and speed_after <= ego.vmax + 1e-5:
                        buffers[i].append((obs[i].copy(), (accel, direction)))
                status = next((name for ids, name in ((collided, "collision"), (arrived, "arrival"),
                                                       (teleported, "teleport")) if vid in ids), None)
                if status is None and not ego.present:
                    status = "disappeared"
                if status is None and ego.step_count >= ego.max_steps:
                    status = "timeout"
                if status is None:
                    obs[i] = ego._get_obs()
                    continue
                rows = buffers[i]
                if not rows:
                    raise RuntimeError(f"Episode {case['id']} produced no usable transitions")
                x, a = zip(*rows, strict=True)
                observations.extend(x)
                actions.extend(a)
                episodes.extend([case["id"]] * len(rows))
                completed.append(case | dict(status=status, start_lane=ego.start_lane, samples=len(rows)))
                if len(completed) % 50 == 0:
                    print(f"Worker {worker}: {len(completed)}/{len(cases)} episodes, "
                          f"{len(observations):,} labels", flush=True)
                following = next(queue, None)
                if following is None:
                    if ego.present:
                        vehicle.remove(vid)
                        ego.present = False
                    ego.background.reset(libsumo, libsumo.simulation.getTime(), 0,
                                         ego.traffic_random, env._live_background)
                    del active[i], buffers[i]
                else:
                    active[i], buffers[i] = following, []
                    ego.traffic_random = np.random.default_rng(np.random.SeedSequence(following["seed"]).spawn(4)[3])
                    obs[i] = env._respawn(ego, following["route"], traffic_vph=TRAFFIC_VPH,
                                          signal_tick=following["tick"])
    finally:
        env.close()
    shard = output / "data.npz"
    np.savez_compressed(shard, observation=np.asarray(observations, np.float32),
                        action_raw=np.asarray(actions, np.float32), episode=np.asarray(episodes, np.int32))
    return str(shard), completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="BC51 checkpoint used for observation and evaluation")
    parser.add_argument("--episodes-per-route", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=64)
    parser.add_argument("--seed", type=int, default=76051)
    args = parser.parse_args()
    if (not 1 <= args.workers <= 50 or min(args.episodes_per_route, args.epochs, args.replicas) < 1
            or args.episodes_per_route % args.workers):
        parser.error("Episodes per route must be a positive multiple of workers; replicas and epochs must be positive")
    baseline = args.baseline.resolve(strict=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = Path("data").resolve() / f"background_bc51_idm_center_{stamp}"
    run = Path("results").resolve() / f"BC_{stamp}"
    print(f"Dataset: {data}\nRun: {run}", flush=True)
    context = data / "context"
    saved, curriculum, original_data = fixed_start_context(baseline.parent, context, max_vph=TRAFFIC_VPH)
    context_model = context / "model.pt"
    road, mdp, train = load_saved_configs(str(context_model))
    if mdp["OBSERVATION"].get("guidance_lateral_scale") is None:
        raise ValueError("Expected a 51-dimensional BC observation configuration")
    cfg = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(data / "sumo"))
    set_weighted_od_flow_rates(data / "sumo/highway.rou.xml", TRAFFIC_VPH)
    collision_settings = configure_collision_gaps(cfg, ego_gap_m=road["EGO"]["collision_gap_m"],
                                                  background_gap_m=road["TRAFFIC"]["collision_gap_m"])
    routes = sum(read_ego_route_groups(data / "sumo/highway.ego.rou.xml"), ())
    if len(routes) != 23:
        raise ValueError(f"Expected 23 routes, got {len(routes)}")
    original_collection = json.loads((original_data / "config.json").read_text())
    save_json(data / "config.json", dict(observation=mdp["OBSERVATION"], simulation=mdp["SIMULATION"],
        traffic=road["TRAFFIC"], ego=road["EGO"], action=mdp["ACTION"], collision_detection=collision_settings,
        expert=EXPERT_TYPE, expert_control_timing=dict(spawn_protect=PROTECTED_DISTANCE),
        episodes_per_route=args.episodes_per_route, seed=args.seed, workers=args.workers,
        replicas_per_worker=args.replicas, traffic_vph=TRAFFIC_VPH, route_sampling="shuffled blocks",
        source_observation_dataset=str(original_data), filter=original_collection.get("filter", {})))
    bag = np.setdiff1d(np.arange(250), TICKS)
    cases = []
    for worker in range(args.workers):
        rng = np.random.default_rng([args.seed, worker])
        assignment = []
        for _ in range(args.episodes_per_route // args.workers):
            for route in rng.permutation(routes):
                ident = worker * (args.episodes_per_route // args.workers) * len(routes) + len(assignment)
                assignment.append(dict(id=ident, route=str(route), tick=int(rng.choice(bag)),
                                       seed=[args.seed, ident]))
        cases.append(assignment)
    bank = TrafficSnapshotBank(cfg, curriculum, road["TRAFFIC"]["vehs_per_hour"],
                               (TRAFFIC_VPH,), STEP, args.seed, refresh=False)
    try:
        with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(collect, i, batch, context_model, cfg, saved["traffic_curriculum"],
                                bank.path, data / f"worker_{i}", args.replicas)
                    for i, batch in enumerate(cases)]
            collected = [job.result() for job in jobs]
    finally:
        bank.close()
    rows = [case for _, records in collected for case in records]
    coverage = Counter(c["route"] for c in rows)
    if coverage != {route: args.episodes_per_route for route in routes}:
        raise RuntimeError(f"Unexpected route coverage: {coverage}")
    data_parts = {key: [] for key in ("observation", "action_raw", "episode")}
    for path, _ in collected:
        with np.load(path) as shard:
            for key, parts in data_parts.items():
                parts.append(shard[key])
    arrays = {key: np.concatenate(parts) for key, parts in data_parts.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Non-finite value in collected dataset")
    dataset = data / "dataset.npz"
    np.savez_compressed(dataset, **arrays)
    summary = dict(samples=len(arrays["episode"]), episodes=len(rows), per_route=coverage,
                   outcomes=Counter(c["status"] for c in rows),
                   lane_actions=Counter(str(int(x)) for x in arrays["action_raw"][:, 1]))
    save_json(data / "summary.json", summary)
    save_json(data / "episodes.json", rows)
    print(f"Collection complete: {summary}", flush=True)
    subprocess.run([sys.executable, "-u", "train_bc_actor.py", "--data", str(dataset),
                    "--out-dir", str(run), "--epochs", str(args.epochs),
                    "--patience", str(args.epochs)], check=True)
    shutil.copy2(run / "model.pt", run / "selected_model.pt")
    with (run / "mdp_config.py").open("a") as file:
        file.write(f"\nOBSERVATION = {mdp['OBSERVATION']!r}\nSIMULATION = {mdp['SIMULATION']!r}\n")
    evaluate(run, run / "analysis/traffic6000_center", args.workers,
             checkpoint=run / "selected_model.pt", levels=(TRAFFIC_VPH,), context_builder=fixed_start_context)
    evaluate(baseline.parent, run / "analysis/baseline_traffic6000_center", args.workers,
             checkpoint=baseline, levels=(TRAFFIC_VPH,), context_builder=fixed_start_context)
    print(f"Complete: {run}", flush=True)


if __name__ == "__main__":
    main()
