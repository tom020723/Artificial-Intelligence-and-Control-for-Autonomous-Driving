"""Measure fixed-demand IDM trajectory coverage across independent SUMO roads."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import numpy as np
import subprocess
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
from time import monotonic
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_background_data import BackgroundObserver, collect_episode, libsumo, save_json
from env.collision_detection import junction_collisions, sumo_collision_options
from env.control_timing import lane_change_timing_allowed
from env.background_demand import FLOW_SHARES
from env.parallel_traffic import TrafficCurriculum, TrafficSchedule, parse_traffic_flows
from utils.saved_configs import load_saved_configs


def native_expert_worker(pipe, config, state_file, step_length, traffic_options, spawn_protect=None):
    """Run IDM queries on a separate SUMO instance."""
    try:
        libsumo.start(["sumo", "-c", config, "--step-length", str(step_length),
                       "--collision.action", "warn", "--no-warnings", "true",
                       "--no-step-log", "true", *traffic_options, *sumo_collision_options()])
        while request := pipe.recv():
            try:
                libsumo.simulation.loadState(state_file)
                vehicle = libsumo.vehicle
                if spawn_protect is not None:
                    restored = set(vehicle.getIDList())
                    request = [row for row in request if row[0] in restored]
                for vid, edge, _, _, _ in request:
                    if vehicle.getTypeID(vid) != "car_idm":
                        vehicle.setType(vid, "car_idm")
                    vehicle.setSpeedMode(vid, 31)
                    vehicle.setSpeed(vid, -1)
                    allowed = (not edge.startswith(":") if spawn_protect is None else
                               lane_change_timing_allowed(edge, vehicle.getLanePosition(vid),
                                                          vehicle.getRouteIndex(vid), spawn_protect))
                    vehicle.setLaneChangeMode(vid, 1621 if allowed else 0)
                libsumo.simulationStep()
                arrived = set(libsumo.simulation.getArrivedIDList())
                active = set(vehicle.getIDList())
                responses = {}
                for vid, edge, lane, speed, limits in request:
                    accel_limit, brake_limit = limits
                    if vid in arrived:
                        if spawn_protect is None:
                            responses[vid] = (0., 0)
                    elif vid in active:
                        next_speed = vehicle.getSpeed(vid)
                        delta = (vehicle.getLaneIndex(vid) - lane
                                 if vehicle.getRoadID(vid) == edge else 0)
                        acceleration = (next_speed - speed) / step_length
                        if spawn_protect is not None and (
                            (vehicle.getRoadID(vid) != edge and not edge.startswith(":"))
                            or abs(delta) > 1 or not -brake_limit - 1e-5 <= acceleration <= accel_limit + 1e-5
                        ):
                            continue
                        scale = accel_limit if acceleration >= 0 else brake_limit
                        responses[vid] = (float(np.clip(acceleration / scale, -1, 1)),
                                          int(np.clip(delta, -1, 1)))
                    elif spawn_protect is None:
                        raise RuntimeError(f"Expert branch cannot resolve active ego {vid}")
                pipe.send((True, responses))
            except Exception as exc:
                pipe.send((False, repr(exc)))
                break
    finally:
        libsumo.close()
        pipe.close()


class DaggerTeacher:
    """Run the native IDM query from the pre-action state in a separate SUMO."""
    def __init__(self, cfg, model, hparams, step, folder):
        import torch
        from utils.bc_policy import load_actor

        torch.set_num_threads(1)
        self.actor = load_actor(model, hparams, "cpu").eval()
        self.state = str(Path(folder) / "state.xml")
        ctx = get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.worker = ctx.Process(target=native_expert_worker,
                                  args=(child, str(cfg), self.state, step, [], 0.))
        self.worker.start()
        child.close()
        self.query = False

    def begin(self, env, before):
        import torch
        from collect_background_data import libsumo

        self.query = bool(before)
        if not before:
            return
        request = [(vid, row[2], row[3], row[1], (env.max_accel, env.max_decel))
                   for vid, row in before.items()]
        libsumo.simulation.saveState(self.state)
        self.pipe.send(request)
        with torch.inference_mode():
            actions = self.actor.det_action(torch.from_numpy(np.stack([r[0] for r in before.values()]))).numpy()
        for vid, action in zip(before, actions, strict=True):
            libsumo.vehicle.setSpeedMode(vid, 0)
            libsumo.vehicle.setLaneChangeMode(vid, 0)
            env.ego_id = vid
            env._prepare_step(action)

    def finish(self):
        if not self.query:
            return {}
        ok, result = self.pipe.recv()
        if not ok:
            raise RuntimeError(f"IDM query failed: {result}")
        return result

    def close(self):
        try:
            self.pipe.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.pipe.close()
        self.worker.join(3)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join()


def collect(suite, cfg, flows, output, index, observation=None, traffic_vph=4740, dagger_base=None,
            idm_spawn_protect_m=None):
    road, mdp, train = load_saved_configs(str(suite / "reference/model.pt"))
    env = BackgroundObserver(str(cfg), road["ROAD"], road["EGO"], mdp["SIMULATION"],
                             observation or mdp["OBSERVATION"], mdp["ACTION"], train["REWARD"], traffic=road["TRAFFIC"])
    args = SimpleNamespace(traffic_vph=traffic_vph, seconds=3600, closed_trajectories=True,
                           include_vehicle_type="car_idm", spawn_protect_m=idm_spawn_protect_m,
                           spawn_protect_vehicle_type="car_idm")
    kwargs = dict(traffic_schedule=TrafficSchedule(flows))
    if dagger_base is None:
        return collect_episode(env, index, 99000 + index, args, output, **kwargs)
    # One-step commands must not carry over in a restored expert query.
    env.lane_change_duration = 0.
    with TemporaryDirectory(prefix="bc-dagger-", dir="/dev/shm") as scratch:
        teacher = DaggerTeacher(cfg, dagger_base, train["HPARAMS"], env.step_length, scratch)
        try:
            return collect_episode(env, index, 199000 + index, args, output, dagger=teacher, **kwargs)
        finally:
            teacher.close()


def select_dataset(source, destination, ids, assignments, quotas, metadata):
    """Select whole trajectories independently in the two run partitions."""
    rng = np.random.default_rng(20261121)
    groups = {}
    for run, split in assignments.items():
        with np.load(source / f"episode_{run:03d}.npz", allow_pickle=False) as shard:
            _, first = np.unique(shard["vehicle_id"], return_index=True)
            for vid, route, time in zip(shard["vehicle_id"][first], shard["route_id"][first],
                                        shard["entry_time"][first], strict=True):
                band = min(2, int(time // 1200))
                groups.setdefault((split, route, band), []).append((run, vid))
    selected = {}
    histogram = {}
    for split, quota in quotas.items():
        histogram[split] = {}
        for route in ids:
            groups_by_band = [groups.get((split, route, b), []) for b in range(3)]
            allocation = [min(len(group), quota // 3) for group in groups_by_band]
            while sum(allocation) < quota:
                available = [i for i in range(3) if allocation[i] < len(groups_by_band[i])]
                band = min(available, key=allocation.__getitem__)
                allocation[band] += 1
            histogram[split][route] = allocation
            for group, n in zip(groups_by_band, allocation, strict=True):
                for index in rng.choice(len(group), n, replace=False):
                    run, vid = group[index]
                    selected.setdefault(run, set()).add(vid)
    destination.mkdir()
    shards, validation = [], []
    count = 0
    obs_dim = None
    for run, vids in sorted(selected.items()):
        name = f"episode_{run:03d}.npz"
        with np.load(source / name, allow_pickle=False) as shard:
            keep = np.isin(shard["vehicle_id"], list(vids))
            data = {key: shard[key][keep] for key in shard.files}
        if assignments[run] == "validation":
            validation.append(len(shards))
        shards.append(name)
        obs_dim = data["observation"].shape[1]
        count += len(data["observation"])
        np.savez_compressed(destination / name, **data)
    save_json(destination / "manifest.json", dict(shards=shards, samples=count, obs_dim=obs_dim,
                                                   validation_indices=validation))
    if junction_collisions():
        metadata = dict(metadata, junction_collisions=True)
    save_json(destination / "config.json", metadata)
    save_json(destination / "selection.json", dict(assignments=assignments, quotas=quotas,
                                                    by_route_and_time=histogram, pairs=count))
    print(f"Selected {count:,} pairs from whole trajectories", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--train-quota", type=int)
    parser.add_argument("--validation-quota", type=int)
    parser.add_argument("--dagger-base", type=Path)
    gap_mode = parser.add_mutually_exclusive_group()
    gap_mode.add_argument("--leader-gap-geometry", action="store_true",
                          help="Measure clamped front gaps from lane geometry during collection")
    gap_mode.add_argument("--connected-lane-bumper-gaps", action="store_true",
                          help="Measure all five lanes' bumper gaps along connected lane segments")
    parser.add_argument("--signed-adjacent-gaps", action="store_true",
                        help="Preserve signed overlap for adjacent front and rear gaps")
    parser.add_argument("--idm-spawn-protect-m", type=float,
                        help="Block IDM lane changes at entry and append normalized entry progress")
    parser.add_argument("--finetune-lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lane-weight-root", type=int, default=2)
    parser.add_argument("--eval-smoke", type=int, default=0)
    args = parser.parse_args()
    if args.lane_weight_root < 1:
        parser.error("--lane-weight-root must be at least 1")
    if args.signed_adjacent_gaps and not args.connected_lane_bumper_gaps:
        parser.error("--signed-adjacent-gaps requires --connected-lane-bumper-gaps")
    if args.idm_spawn_protect_m is not None and (
            not np.isfinite(args.idm_spawn_protect_m) or args.idm_spawn_protect_m <= 0 or args.dagger_base):
        parser.error("--idm-spawn-protect-m requires a positive finite distance and native IDM collection")
    quotas = {"train": args.train_quota, "validation": args.validation_quota}
    balanced = any(x is not None for x in quotas.values())
    if balanced and any(x is None or x < 1 for x in quotas.values()):
        parser.error("Both trajectory quotas must be positive")
    suite, output = args.suite.resolve(), args.output.resolve()
    dagger_base = args.dagger_base.resolve(strict=True) if args.dagger_base else None
    if dagger_base and not balanced:
        parser.error("DAgger requires both trajectory quotas")
    spec = json.loads((suite / "suite.json").read_text())
    if spec["traffic_vph"] <= 0 or spec.get("adaptive_traffic") or min(args.workers, args.target) < 1:
        parser.error("Use a fixed-demand suite and positive limits")
    output.mkdir(parents=True)
    shutil.copytree(suite / "sumo", output / "sumo")
    cfg = output / "sumo/highway.sumocfg"
    route_file = cfg.with_name("highway.rou.xml")
    tree = ET.parse(route_file)
    root = tree.getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
    nodes = root.findall("flow")
    ids = [f.get("route") or "background_" + f.get("id") for f in nodes]
    rare = [name for f, name in zip(nodes, ids, strict=True)
            if FLOW_SHARES[f.get("id")] == min(FLOW_SHARES.values())]
    existing = set(routes)
    flows = parse_traffic_flows(nodes, routes, cfg.with_name("highway.net.xml"), 800,
                               TrafficCurriculum(**spec["settings"]))
    for f in nodes:
        root.remove(f)
    for name in sorted(routes.keys() - existing):
        ET.SubElement(root, "route", id=name, edges=" ".join(routes[name]))
    tree.write(route_file, encoding="utf-8", xml_declaration=True)
    started = monotonic()
    partitions = {key: Counter() for key in quotas}
    assignments = {}
    split_rng = np.random.default_rng(0)
    validation_runs = {0 if args.eval_smoke else int(split_rng.integers(20))}
    road, mdp, _ = load_saved_configs(str(suite / "reference/model.pt"))
    observation = dict(mdp["OBSERVATION"], ring_merge_scope="direct", junction_neighbor_observation=True) if balanced else None
    if args.leader_gap_geometry:
        observation = dict(observation if observation is not None else mdp["OBSERVATION"],
                           leader_gap_geometry=True)
    if args.connected_lane_bumper_gaps:
        observation = dict(observation if observation is not None else mdp["OBSERVATION"],
                           connected_lane_bumper_gaps=True, leader_gap_geometry=False,
                           signed_adjacent_gaps=args.signed_adjacent_gaps)
    if args.idm_spawn_protect_m is not None:
        observation = dict(observation if observation is not None else mdp["OBSERVATION"],
                           spawn_progress_scale=args.idm_spawn_protect_m)
    def pending():
        if balanced:
            return any(min(partitions[k][r] for r in ids) < quota for k, quota in quotas.items())
        return min(counts[r] for r in rare) < args.target
    counts, cutoff = Counter(), Counter()
    pairs_by_route, observation_counts = Counter(), Counter()
    completed, number = 0, 0
    first_milestone = None
    print(f"Tracking {rare}, target {args.target} each", flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        while pending():
            if shutil.disk_usage(output).free < 8 * 1024**3:
                raise RuntimeError("Insufficient disk space for collection and selected dataset")
            for i in range(number, number + args.workers):
                if i and i % 20 == 0:
                    validation_runs.add(i + int(split_rng.integers(20)))
                assignments[i] = "validation" if i in validation_runs else "train"
            jobs = [pool.submit(collect, suite, cfg, flows, output, i, observation, spec["traffic_vph"],
                                dagger_base, args.idm_spawn_protect_m)
                    for i in range(number, number + args.workers)]
            number += args.workers
            for job in as_completed(jobs):
                report = job.result()
                completed += 1
                counts.update(report["trajectories_by_route"])
                partitions[assignments[report["episode"]]].update(report["trajectories_by_route"])
                cutoff.update(report["cutoff_by_route"])
                pairs_by_route.update(report["routes"])
                observation_counts.update({key: report["counts"].get(key, 0) for key in
                                           ("observed_samples", "leader_gap_geometry_corrected_samples")})
                summary = dict(runs=completed, pairs=sum(pairs_by_route.values()),
                               elapsed_seconds=round(monotonic()-started, 2),
                               rare_routes={r: counts[r] for r in rare},
                               trajectories_by_route=dict(counts), pairs_by_route=dict(pairs_by_route),
                               cutoff_by_route=dict(cutoff),
                               observation_counts=dict(observation_counts),
                               partition_counts={key: dict(c) for key, c in partitions.items()})
                if first_milestone is None and min(counts[r] for r in rare) >= args.target:
                    first_milestone = {k: summary[k] for k in ("runs", "pairs", "elapsed_seconds", "rare_routes")}
                summary["first_milestone"] = first_milestone
                save_json(output / "summary.tmp", summary)
                (output / "summary.tmp").replace(output / "summary.json")
                if balanced:
                    minima = {k: min(c[r] for r in ids) for k, c in partitions.items()}
                    print(f"Saved {summary['pairs']:,} pairs; lowest route counts {minima}; targets {quotas}", flush=True)
                else:
                    print(f"Saved {summary['pairs']:,} pairs; rare routes {summary['rare_routes']}", flush=True)

    if balanced:
        dataset = output / "dataset"
        select_dataset(output, dataset, ids, assignments, quotas,
                       dict(observation=observation, simulation=mdp["SIMULATION"], traffic=road["TRAFFIC"],
                            collision_detection=spec["collision_detection"],
                            idm_spawn_protect_m=args.idm_spawn_protect_m,
                            dagger_base=str(dagger_base) if dagger_base else None,
                            rollout_controller="BC" if dagger_base else "IDM", label_controller="IDM"))
        model = output / "training"
        cmd = [sys.executable, "-u", "scripts/train_bc_shards.py", str(dataset),
               "--out-dir", str(model), "--epochs", str(args.epochs),
               "--lane-weight-root", str(args.lane_weight_root)]
        if dagger_base:
            cmd += ["--init-model", str(dagger_base), "--lr", str(args.finetune_lr)]
        subprocess.run(cmd, check=True)
        from scripts.evaluate_fixed_routes import run
        run(suite, output.name, None, model / "model.pt", 64, args.eval_smoke,
            output=output / "evaluation")


if __name__ == "__main__":
    main()
