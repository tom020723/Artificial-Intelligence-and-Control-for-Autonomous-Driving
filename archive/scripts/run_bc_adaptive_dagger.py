"""Mixed native-IDM/BC rollouts and cumulative supervised updates in adaptive traffic."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_background_data import BackgroundObserver, action_from_speeds
from env.adaptive_traffic_schedule import AdaptiveTrafficSchedule
from env.parallel_traffic import TrafficCurriculum
from scripts.evaluate_fixed_routes import libsumo, run
from scripts.run_bc_adaptive_idm import prepare_collection
from scripts.run_bc51_dagger import load_actor, native_expert_worker
from scripts.run_bc51_fixed_start import fixed_start_context
from scripts.train_bc_shards import run_pass
from test import load_saved_configs
from train_bc_actor import epoch, export_policy, lane_class


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def collect_worker(road_data, context_model, model, output, target, beta, seed, seconds):
    """Collect all active uniform-IDM vehicles from one continuous SUMO road."""
    torch.set_num_threads(1)
    road, mdp, train = load_saved_configs(str(context_model))
    cfg, flows, routes, lights = road_data
    env = BackgroundObserver(str(cfg), road["ROAD"], road["EGO"], mdp["SIMULATION"],
                             mdp["OBSERVATION"], mdp["ACTION"], train["REWARD"],
                             traffic=road["TRAFFIC"])
    env.lane_change_duration = 0.0
    env.lc_spawn_protect = 0.0
    actor = load_actor(model, train["HPARAMS"], "cpu").eval()
    schedule = AdaptiveTrafficSchedule(flows, cfg.with_name("highway.net.xml"), routes, lights, 1800)
    mix_seed, traffic_seed = np.random.SeedSequence(seed).spawn(2)
    mix_rng = np.random.default_rng(mix_seed)
    output.mkdir()
    paths, obs_rows, action_rows = [], [], []
    count = 0
    pairs_by_route, control = Counter(), Counter()

    def flush():
        if obs_rows:
            path = output / f"labels_{len(paths):03d}.npz"
            np.savez_compressed(path, observation=np.asarray(obs_rows, np.float32),
                                action_raw=np.asarray(action_rows, np.float32))
            paths.append(str(path))
            obs_rows.clear()
            action_rows.clear()
            print(f"{output.parent.name}/{output.name}: {count:,}/{target:,} valid labels", flush=True)

    shared_memory = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with TemporaryDirectory(prefix="dagger-idm-", dir=shared_memory) as scratch:
        state_file = str(Path(scratch) / "expert_state.xml")
        context = get_context("spawn")
        pipe, child = context.Pipe()
        traffic_options = ["--max-depart-delay", "125"]
        expert = context.Process(target=native_expert_worker,
            args=(child, str(cfg), state_file, env.step_length, traffic_options, 0.))
        expert.start()
        child.close()
        try:
            simulation_run = 0
            while count < target:
                run_seed = seed + simulation_run
                rng = np.random.default_rng(traffic_seed.spawn(1)[0])
                simulation_run += 1
                libsumo.start(["sumo", "-c", str(cfg), "--seed", str(run_seed),
                               "--step-length", str(env.step_length), "--collision.action", "remove",
                               "--no-warnings", "true", "--no-step-log", "true",
                               "--save-state.precision", "12", "--save-state.rng", "true", *traffic_options])
                try:
                    schedule.reset(libsumo, 0., 6000, rng, set())
                    vehicle = libsumo.vehicle
                    for _ in range(round(seconds / env.step_length)):
                        active = set(vehicle.getIDList())
                        env._active_ids = active
                        ids = sorted(vid for vid in active if vid.startswith("uniform_idm_") and vehicle.getLaneID(vid))
                        control["peak_idm_on_road"] = max(control["peak_idm_on_road"], len(ids))
                        states = [(vid, vehicle.getRoadID(vid), vehicle.getLaneIndex(vid),
                                   vehicle.getSpeed(vid), (env.max_accel, env.max_decel)) for vid in ids]
                        route_ids = [vehicle.getRouteID(vid) for vid in ids]
                        observations = np.asarray([env.observe_vehicle(vid) for vid in ids], dtype=np.float32)
                        native = mix_rng.random(len(ids)) < beta
                        schedule.spawn_due(libsumo, libsumo.simulation.getTime(), str, rng)
                        # Ask the shadow SUMO for IDM actions only when a learner controls at least one car.
                        query = bool(len(ids) and not native.all())
                        if query:
                            libsumo.simulation.saveState(state_file)
                            pipe.send(states)
                            with torch.inference_mode():
                                learner_actions = actor.det_action(torch.from_numpy(observations[~native])).numpy()
                        if query:
                            actions = iter(learner_actions)
                        for vid, use_native in zip(ids, native, strict=True):
                            if use_native:
                                vehicle.setSpeedMode(vid, 31)
                                vehicle.setSpeed(vid, -1)
                                vehicle.setLaneChangeMode(vid, 1621)
                            else:
                                vehicle.setSpeedMode(vid, 0)
                                vehicle.setLaneChangeMode(vid, 0)
                                env.ego_id = vid
                                env._prepare_step(next(actions))
                        libsumo.simulationStep()
                        active = set(vehicle.getIDList())
                        schedule.ids.intersection_update(active | set(libsumo.simulation.getPendingVehicles()))
                        labels = {}
                        if query:
                            ok, labels = pipe.recv()
                            if not ok:
                                raise RuntimeError(f"Native IDM query failed: {labels}")
                        collided = set(libsumo.simulation.getCollidingVehiclesIDList())
                        control.update(expert=int(native.sum()), learner=int((~native).sum()))
                        for state, obs, route_id, use_native in zip(states, observations, route_ids, native, strict=True):
                            vid, edge, lane, speed, _ = state
                            if use_native:
                                # The expert actually drove here, so use its realized action.
                                label = None
                                if vid in active and vid not in collided and vehicle.getRoadID(vid) == edge:
                                    next_speed = vehicle.getSpeed(vid)
                                    accel, _, clipped = action_from_speeds(
                                        speed, next_speed, env.step_length, env.max_accel, env.max_decel)
                                    change = vehicle.getLaneIndex(vid) - lane
                                    if not clipped and abs(change) <= 1:
                                        label = accel, change
                            else:
                                label = labels.get(vid)
                            if label is None:
                                control["unlabelled"] += 1
                                continue
                            if count == target:
                                break
                            control["native_labels" if use_native else "shadow_labels"] += 1
                            obs_rows.append(obs)
                            action_rows.append(label)
                            pairs_by_route[route_id] += 1
                            count += 1
                        if len(obs_rows) >= 100_000:
                            flush()
                        if count == target:
                            break
                finally:
                    libsumo.close()
            flush()
        finally:
            try:
                pipe.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            pipe.close()
            expert.join(timeout=5)
            if expert.is_alive():
                expert.terminate()
                expert.join()
    result = dict(shards=paths, pairs=count, routes=dict(pairs_by_route),
                  control_steps=dict(control), simulation_runs=simulation_run)
    write_json(output / "collection.json", result)
    return result


def route_index(paths):
    index = []
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            names, sizes = np.unique(shard["route_id"], return_counts=True)
        index.append(dict(zip(names, map(int, sizes), strict=True)))
    return index


def base_sample(paths, index, count, rng, obs, actions):
    remaining = sum((Counter(n) for n in index), Counter())
    names = sorted(remaining)
    per_route, remainder = divmod(count, len(names))
    quota = {n: per_route + (i < remainder) for i, n in enumerate(rng.permutation(names))}
    if len(names) != 24 or any(remaining[n] < q for n, q in quota.items()):
        raise ValueError(f"Insufficient expert samples for balanced quotas: {remaining}")
    requested = quota.copy()
    cursor = 0
    for path, counts in zip(paths, index, strict=True):
        grabs = {}
        for name, local in counts.items():
            if quota[name]:
                grabs[name] = rng.hypergeometric(local, remaining[name]-local, quota[name])
                quota[name] -= grabs[name]
            remaining[name] -= local
        if not any(grabs.values()):
            continue
        with np.load(path, allow_pickle=False) as shard:
            route = shard["route_id"]
            selected = np.concatenate([rng.choice(np.flatnonzero(route == n), k, replace=False)
                                       for n, k in grabs.items() if k])
            end = cursor + len(selected)
            obs[cursor:end], actions[cursor:end] = shard["observation"][selected], shard["action_raw"][selected]
            cursor = end
    if cursor != count:
        raise RuntimeError(f"Balanced replay produced {cursor:,}/{count:,} samples")
    return requested


def fit(base, previous, output, config, data_paths, index, validation, dagger_shards,
        dagger_count, args, rng):
    output.mkdir()
    for name in ("road_config.py", "mdp_config.py", "train.py"):
        shutil.copy2(base / name, output / name)
    total = args.base_pairs + dagger_count
    obs = np.empty((total, config["observation_dim"]), np.float32)
    actions = np.empty((total, 2), np.float32)
    quotas = base_sample(data_paths, index, args.base_pairs, rng, obs, actions)
    cursor = args.base_pairs
    for path in dagger_shards:
        with np.load(path, allow_pickle=False) as shard:
            end = cursor + len(shard["action_raw"])
            obs[cursor:end], actions[cursor:end] = shard["observation"], shard["action_raw"]
            cursor = end
    if cursor != total:
        raise RuntimeError("DAgger aggregate size differs from recorded collection")
    write_json(output / "config.json", config | dict(
        init_model=str(previous / "model.pt"), train_samples=total, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, dagger_samples=dagger_count, base_samples=quotas))
    _, _, train = load_saved_configs(str(base / "model.pt"))
    device = args.device
    actor = load_actor(previous / "model.pt", train["HPARAMS"], device)
    state = torch.load(previous / "bc_training_state.pt", map_location=device, weights_only=True)
    actor.load_state_dict(state["state_dict"])
    mean, std = state["obs_mean"], state["obs_std"]
    weights = torch.as_tensor(config["lane_weights"], device=device, dtype=torch.float32)
    params = [*actor.trunk.parameters(), *actor.mu_head.parameters(), *actor.lane_head.parameters()]
    for param in actor.v_head.parameters():
        param.requires_grad_(False)
    actor.log_std.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    fields = ("epoch", "train_loss", "val_loss", "val_accel_mse", "val_lane_ce", "val_lane_macro_f1",
              "val_left_precision", "val_left_recall", "val_right_precision", "val_right_recall")
    best, best_epoch = float("inf"), 0
    with (output / "training_log.csv").open("w") as logfile:
        writer = csv.DictWriter(logfile, fieldnames=fields)
        writer.writeheader()
        for i in range(1, args.epochs + 1):
            order = rng.permutation(total)
            loss = 0.
            for start in range(0, total, 250_000):
                ids = order[start:start+250_000]
                x = (torch.as_tensor(obs[ids], device=device) - mean) / std
                y = torch.as_tensor(actions[ids, 0], device=device)
                z = torch.as_tensor(lane_class(actions[ids]), device=device)
                stats = epoch(actor, x, y, z, weights, args.batch_size, optimizer)
                loss += stats["loss"] * len(ids)
            val = run_pass(actor, validation, mean, std, weights, args.batch_size)
            writer.writerow(dict(epoch=i, train_loss=loss/total,
                                 **{key: val[key[4:]] for key in fields if key.startswith("val_")}))
            logfile.flush()
            if val["loss"] < best:
                best, best_epoch = val["loss"], i
                torch.save(export_policy(actor, mean, std), output / "model.pt")
                torch.save(dict(state_dict=actor.state_dict(), obs_mean=mean, obs_std=std, epoch=i),
                           output / "bc_training_state.pt")
            print(f"{output.parent.name} epoch {i}/{args.epochs}: train={loss/total:.5f} val={val['loss']:.5f}", flush=True)
    write_json(output / "result.json", dict(status="complete", epochs_completed=args.epochs,
                                           best_epoch=best_epoch, best_val_loss=best))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results") / datetime.now().strftime("DAgger_%Y%m%d_%H%M%S"))
    p.add_argument("--betas", type=float, nargs=4, default=(.8, .6, .2, 0.))
    p.add_argument("--pairs", type=int, default=10_000_000)
    p.add_argument("--base-pairs", type=int, default=10_000_000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seconds-per-run", type=int, default=3600)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=61089)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if (args.workers < 1 or args.seconds_per_run < 1 or args.pairs < args.workers or
        args.base_pairs < 24 or args.epochs < 1 or args.batch_size < 1 or
        any(not 0 <= beta <= 1 for beta in args.betas)):
        p.error("Invalid rollout or training settings")
    args.base, args.suite, args.out_dir = (path.resolve() for path in (args.base, args.suite, args.out_dir))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    spec = json.loads((args.suite / "suite.json").read_text())
    if spec["traffic_vph"] != 6000 or not spec.get("adaptive_traffic") or spec.get("uniform_idm_vph"):
        p.error("--suite must use all-adaptive 6,000 vehicles/hour")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    config_path = args.out_dir / "config.json"
    if args.resume:
        if json.loads(config_path.read_text()) != json.loads(json.dumps(settings)):
            p.error("Resume settings must match the original run")
    else:
        args.out_dir.mkdir(parents=True)
        write_json(config_path, settings)
    config = json.loads((args.base / "config.json").read_text())
    if not args.resume:
        fixed_start_context(args.base, args.out_dir / "context", max_vph=6000)
    road, _, _ = load_saved_configs(str(args.suite / "reference/model.pt"))
    road_dir = args.out_dir
    if args.resume:
        road_dir = args.out_dir / datetime.now().strftime("resume_%Y%m%d_%H%M%S")
        road_dir.mkdir()
    road_data = prepare_collection(road_dir, args.suite,
                                   TrafficCurriculum(**spec["settings"]), road["TRAFFIC"]["vehs_per_hour"])
    dataset = Path(config["dataset_directory"])
    paths = [dataset / x for x in json.loads((dataset / "manifest.json").read_text())["shards"]]
    train_paths = [paths[i] for i in config["train_episodes"]]
    validation = [paths[i] for i in config["validation_episodes"]]
    if args.smoke:
        validation = validation[:1]
    index = route_index(train_paths)
    checkpoint = args.base
    accumulated = []
    dagger_count = 0
    betas = args.betas if not args.smoke else args.betas[:1]
    for i, beta in enumerate(betas, 1):
        out = args.out_dir / f"round_{i:02d}"
        out.mkdir(exist_ok=args.resume)
        print(f"Round {i}: beta={beta:.0%}, target={args.pairs:,} pairs", flush=True)
        each, rem = divmod(args.pairs, args.workers)
        collection, pending = {}, []
        for w in range(args.workers):
            worker_dir = out / f"worker_{w}"
            worker_report = worker_dir / "collection.json"
            quota = each + (w < rem)
            if args.resume and worker_report.exists():
                report = json.loads(worker_report.read_text())
                if report["pairs"] != quota or not all(Path(s).is_file() for s in report["shards"]):
                    raise RuntimeError(f"Incomplete saved collection: {worker_report}")
                collection[w] = report
            else:
                if args.resume and worker_dir.exists():
                    worker_dir.rename(worker_dir.with_name(
                        worker_dir.name + datetime.now().strftime("_incomplete_%Y%m%d_%H%M%S")))
                pending.append((w, worker_dir, quota))
        if pending:
            print(f"Round {i}: collecting from {len(pending)} workers, reusing {len(collection)} workers", flush=True)
            with ProcessPoolExecutor(len(pending), mp_context=get_context("spawn")) as pool:
                jobs = {w: pool.submit(collect_worker, road_data, args.out_dir / "context/model.pt",
                                       checkpoint / "model.pt", worker_dir, quota,
                                       beta, args.seed + 1000*i + w, args.seconds_per_run)
                        for w, worker_dir, quota in pending}
                collection.update({w: future.result() for w, future in jobs.items()})
        collection = [collection[w] for w in range(args.workers)]
        dagger_count += sum(report["pairs"] for report in collection)
        accumulated.extend(s for report in collection for s in report["shards"])
        write_json(out / "collection.json", dict(beta=beta, round_pairs=args.pairs,
                                                  cumulative_pairs=dagger_count, workers=collection))
        model = out / "training"
        evaluation = out / "evaluation"
        if not (args.resume and (model / "result.json").is_file() and (model / "model.pt").is_file()
                and (model / "bc_training_state.pt").is_file()):
            if args.resume and model.exists():
                model.rename(model.with_name(model.name + datetime.now().strftime("_incomplete_%Y%m%d_%H%M%S")))
            fit(args.base, checkpoint, model, config, train_paths, index, validation, accumulated,
                dagger_count, args, np.random.default_rng(np.random.SeedSequence([args.seed, i])))
        if not (args.resume and (evaluation / "evaluation_summary.json").is_file()
                and (evaluation / "route_evaluation.csv").is_file() and (evaluation / "summary.png").is_file()):
            if args.resume and evaluation.exists():
                evaluation.rename(evaluation.with_name(
                    evaluation.name + datetime.now().strftime("_incomplete_%Y%m%d_%H%M%S")))
            run(args.suite, f"DAgger round {i}", None, model / "model.pt", 64, 2 if args.smoke else 0,
                output=evaluation)
        else:
            print(f"Round {i}: reusing completed training and evaluation", flush=True)
        checkpoint = model


if __name__ == "__main__":
    main()
