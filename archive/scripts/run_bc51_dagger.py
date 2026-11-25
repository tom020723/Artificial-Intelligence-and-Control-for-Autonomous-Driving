"""DAgger rounds over 89 route/departure pairs with failure-weighted extra episodes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.ppo import PPO
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

COMBOS = 89
TRAFFIC = 6000
EPOCHS = 10


class DaggerSumo(ParallelSumo):
    def _respawn(self, ego, route=None, **kwargs):
        choices = self.departures[route]
        self.departures[route] = tuple(c for c in choices if c[2] == self.start_lanes[ego.replica_index])
        try:
            return super()._respawn(ego, route, **kwargs)
        finally:
            self.departures[route] = choices

    def _advance(self, egos, actions):
        vehicle = libsumo.vehicle
        states = []
        expert_before = {}
        self.expert_actions = {}
        for ego, action in zip(egos, actions, strict=True):
            vid = ego.ego_id
            edge = vehicle.getRoadID(vid)
            if vid not in self.expert_ids or edge.startswith(":"):
                states.append(ego._prepare_step(action))
                continue
            lane = vehicle.getLaneIndex(vid)
            speed = vehicle.getSpeed(vid)
            signals = ego._upcoming_signals()
            connected = bool(signals) and ego._lane_follows_route(vehicle.getLaneID(vid))
            distance = vehicle.getDistance(vid) if signals else None
            states.append((0., 0., 0, False, lane, speed, signals, connected, distance, None))
            allowed = lane_change_timing_allowed(edge, vehicle.getLanePosition(vid),
                                                 vehicle.getRouteIndex(vid), 0)
            vehicle.setType(vid, "car_idm")
            vehicle.setSpeedMode(vid, 31)
            vehicle.setSpeed(vid, -1)
            vehicle.setLaneChangeMode(vid, 1621 if allowed else 0)
            expert_before[vid] = (edge, lane, speed, ego)
        now = libsumo.simulation.getTime()
        for ego in egos:
            ego.background.spawn_due(libsumo, now, ego.prefix, ego.traffic_random)
        libsumo.simulationStep()
        active = set(vehicle.getIDList())
        self._live_background = active | set(libsumo.simulation.getPendingVehicles())
        collided = set(libsumo.simulation.getCollidingVehiclesIDList())
        arrived = set(libsumo.simulation.getArrivedIDList())
        for vid, (edge, lane, speed, ego) in expert_before.items():
            if vid not in active:
                continue
            next_speed = vehicle.getSpeed(vid)
            delta_lane = vehicle.getLaneIndex(vid) - lane
            acceleration = (next_speed - speed) / ego.step_length
            if (vid not in collided and vehicle.getRoadID(vid) == edge and abs(delta_lane) <= 1
                    and next_speed <= ego.vmax + 1e-5
                    and -ego.max_decel - 1e-5 <= acceleration <= ego.max_accel + 1e-5):
                scale = ego.max_accel if acceleration >= 0 else ego.max_decel
                self.expert_actions[vid] = (np.clip(acceleration / scale, -1, 1), delta_lane)
            vehicle.setType(vid, "ego_type")
            vehicle.setSpeedMode(vid, 0)
            vehicle.setLaneChangeMode(vid, 0)
        for ego, state in zip(egos, states, strict=True):
            ego.present = ego.ego_id in active
            yield ego._finish_step(state, collided=ego.ego_id in collided,
                                   arrived=ego.ego_id in arrived)


def load_actor(checkpoint, hparams, device, *, obs_dim=51):
    actor = PPO(obs_dim=obs_dim, act_dim=2, **(hparams | {"device": device})).policy
    actor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return actor


def gate_off_context(run, context, *args, **kwargs):
    saved, curriculum, data = prepare_context(run, context, *args, **kwargs)
    with (context / "mdp_config.py").open("a") as file:
        file.write("\nACTION['lane_change_safety']['enabled'] = False\n")
    saved["ego_lane_change_safety_enabled"] = False
    return saved, curriculum, data


def native_expert_worker(pipe, config, state_file, step_length, traffic_options, spawn_protect=None):
    """Run IDM queries on a separate SUMO instance."""
    try:
        libsumo.start(["sumo", "-c", config, "--step-length", str(step_length),
                       "--collision.action", "warn", "--no-warnings", "true",
                       "--no-step-log", "true", *traffic_options])
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


class StandardDaggerSumo(DaggerSumo):
    def _start(self, config, *, show_gui=False):
        libsumo.start(["sumo", "-c", config, "--step-length", str(self.prototype.step_length),
                       "--seed", str(self.seed), "--collision.action", "remove",
                       "--no-warnings", "true", "--no-step-log", "true",
                       "--save-state.precision", "12", "--save-state.rng", "true",
                       *self.traffic_curriculum.sumo_options()])
        self.started = True

    def start_expert(self, state_file):
        self.expert_state = str(state_file)
        context = get_context("spawn")
        self.expert_pipe, child = context.Pipe()
        self.expert_process = context.Process(target=native_expert_worker,
            args=(child, self.config, self.expert_state, self.prototype.step_length,
                  self.traffic_curriculum.sumo_options()))
        self.expert_process.start()
        child.close()

    def _advance(self, egos, actions):
        vehicle = libsumo.vehicle
        requests = [(ego.ego_id, vehicle.getRoadID(ego.ego_id), vehicle.getLaneIndex(ego.ego_id),
                     vehicle.getSpeed(ego.ego_id), (ego.max_accel, ego.max_decel))
                    for ego in egos]
        libsumo.simulation.saveState(self.expert_state)
        self.expert_pipe.send(requests)
        # The main SUMO executes only the learned policy. Its state is never restored.
        steps = list(ParallelSumo._advance(self, egos, actions))
        ok, response = self.expert_pipe.recv()
        if not ok:
            raise RuntimeError(f"SUMO expert query failed: {response}")
        self.expert_actions = response
        return iter(steps)

    def close(self):
        try:
            if hasattr(self, "expert_process"):
                try:
                    self.expert_pipe.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.expert_pipe.close()
                self.expert_process.join(timeout=3)
                if self.expert_process.is_alive():
                    self.expert_process.terminate()
                    self.expert_process.join()
        finally:
            super().close()


def collect(cases, checkpoint, context_model, cfg, work, settings, snapshot, output,
            standard=False):
    torch.set_num_threads(1)
    road, mdp, train = load_saved_configs(str(context_model))
    actor = load_actor(checkpoint, train["HPARAMS"], "cpu").eval()
    env_class = StandardDaggerSumo if standard else DaggerSumo
    env = env_class(len(cases), dict(
        cfg_path=cfg, road=road["ROAD"], ego=road["EGO"], traffic=road["TRAFFIC"],
        mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
        action=mdp["ACTION"], reward=train["REWARD"]), work, seed=0,
        randomize_signals=True, randomize_departures=True,
        traffic_curriculum=TrafficCurriculum(**settings), fixed_traffic_vph=TRAFFIC,
        traffic_snapshot_path=snapshot)
    env.start_lanes = [case["lane"] for case in cases]
    rows, statuses = [], {}
    try:
        obs = env.reset([case["route"] for case in cases],
                        signal_ticks=[case["tick"] for case in cases],
                        replica_seeds=[np.random.SeedSequence(case["seed"]) for case in cases])
        assert [ego.start_lane for ego in env.egos] == env.start_lanes
        assert obs.shape == (len(cases), 51)
        if standard:
            env.start_expert(work / "expert_state.xml")
        active = list(range(len(cases)))
        for step in range(env.prototype.max_steps):
            if not active:
                break
            if not standard:
                env.expert_ids = {env.egos[i].ego_id for i in active
                                  if step % 5 == cases[i]["expert_phase"]}
            with torch.inference_mode():
                actions = actor.det_action(torch.from_numpy(obs[active])).numpy()
            ongoing = []
            for i, (next_obs, _, terminated, truncated, info) in zip(
                    active, env.step_evaluation(active, actions), strict=True):
                expert = env.expert_actions.get(env.egos[i].ego_id)
                if expert is not None:
                    rows.append((obs[i].copy(), *expert, cases[i]["id"]))
                obs[i] = next_obs
                if terminated or truncated:
                    statuses[i] = next((status for flag, status in (
                        ("red_light_violation", "red_light"), ("collided", "collision"),
                        ("arrived", "arrival"), ("wrong_lane", "wrong_lane"))
                        if info.get(flag)), "timeout")
                else:
                    ongoing.append(i)
            active = ongoing
        if len(statuses) != len(cases) or not rows:
            raise RuntimeError("Incomplete DAgger rollout batch")
        observations, acceleration, lanes, episode = zip(*rows, strict=True)
        np.savez_compressed(output, observation=np.stack(observations),
                            acceleration=np.asarray(acceleration, np.float32),
                            lane=np.asarray(lanes, np.int8), episode=np.asarray(episode, np.int32))
        print(f"{output.stem}: {len(cases)} episodes, {len(rows)} labels, "
              f"outcomes={dict(Counter(statuses.values()))}", flush=True)
        return [case | {"status": statuses[i]} for i, case in enumerate(cases)]
    finally:
        env.close()


def plan_round(combos, history, seed, number, extra_episodes=384):
    rng = np.random.default_rng([seed, number])
    recent = [case for round_cases in history[-3:] for case in round_cases]
    attempts = np.bincount([c["combo"] for c in recent], minlength=COMBOS)
    failures = np.bincount([c["combo"] for c in recent if c["status"] != "arrival"], minlength=COMBOS)
    scores = (failures + 2) / (attempts + 4)
    probabilities = scores / scores.sum()
    ticks = np.setdiff1d(np.arange(250), TICKS)
    quarters = np.array_split(ticks, 4)
    draws = [(i, int(rng.choice(part))) for i in range(COMBOS) for part in quarters]
    rng.shuffle(draws)
    draws.extend((int(i), int(rng.choice(ticks)))
                 for i in rng.choice(COMBOS, extra_episodes, p=probabilities))
    cases = []
    for j, (combo, tick) in enumerate(draws):
        route, lane = combos[combo]
        cases.append(dict(id=(number - 1) * len(draws) + j, combo=combo, route=route,
                          lane=lane, tick=tick, seed=[seed, number, j],
                          expert_phase=int(rng.integers(5)), stratum="base" if j < 356 else "extra"))
    return cases, probabilities.tolist()


def load_replay(dataset, base_config, rng, device):
    manifest = json.loads((dataset / "manifest.json").read_text())
    ids = base_config["train_episodes"]
    observations, actions = [], []
    for index in rng.choice(ids, min(128, len(ids)), replace=False):
        with np.load(dataset / manifest["shards"][index]) as shard:
            action = shard["action_raw"]
            picked = np.concatenate((rng.choice(len(action), min(256, len(action)), replace=False),
                                     np.flatnonzero(action[:, 1])[:128]))
            observations.append(shard["observation"][picked])
            actions.append(action[picked])
    return tuple(torch.as_tensor(np.concatenate(rows), device=device)
                 for rows in (observations, actions))


def fit(actor, replay, teacher, shards, rng, log_file, *, standard=False):
    arrays = {key: [] for key in ("observation", "acceleration", "lane", "episode")}
    for path in shards:
        with np.load(path) as data:
            for key, values in arrays.items():
                values.append(data[key])
    x, a, lane, episode = (np.concatenate(values) for values in arrays.values())
    weights = np.where(lane != 0, 4., 1.)
    weights /= np.bincount(episode, weights=weights)[episode]
    weights /= weights.sum()
    device = next(actor.parameters()).device
    x, a, y = (torch.as_tensor(values, device=device) for values in (x, a, (1 - lane).astype(np.int64)))
    if replay is not None:
        at_x, at_actions = replay
        target_accel, target_probs = teacher
    params = [*actor.trunk.parameters(), *actor.mu_head.parameters(), *actor.lane_head.parameters()]
    optimizer = torch.optim.Adam(params, lr=3e-5)
    actor.train()
    history = []
    for epoch in range(1, EPOCHS + 1):
        totals = np.zeros(3)
        epoch_indices = rng.permutation(len(x)) if standard else rng.choice(len(x), len(x), p=weights)
        for idx in np.array_split(epoch_indices, int(np.ceil(len(x) / 256))):
            mu, _, logits, _ = actor(x[idx])
            expert = F.mse_loss(mu.flatten(), a[idx]) + F.cross_entropy(logits, y[idx])
            bc = anchor = torch.zeros((), device=device)
            if replay is not None:
                j = rng.integers(len(at_x), size=3 * len(idx))
                at_mu, _, at_logits, _ = actor(at_x[j])
                bc = F.mse_loss(at_mu.flatten(), at_actions[j, 0]) + F.cross_entropy(
                    at_logits, (1 - at_actions[j, 1]).long())
                if not standard:
                    anchor = F.mse_loss(at_mu.flatten(), target_accel[j]) + F.kl_div(
                        at_logits.log_softmax(-1), target_probs[j], reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            (expert + bc + 4 * anchor).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            totals += len(idx) * np.asarray([expert.item(), bc.item(), anchor.item()])
        losses = dict(zip(("expert_loss", "bc_replay_loss", "anchor_loss"), (totals / len(x)).tolist(), strict=True))
        history.append(dict(epoch=epoch, **losses))
        print(f"{log_file.parent.name} epoch {epoch}: {losses}", flush=True)
    log_file.write_text(json.dumps(dict(labels=len(x), lane_changes=int(np.count_nonzero(lane)),
                                       epochs=history), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="BC51 selected_model.pt checkpoint")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=61089)
    parser.add_argument("--standard", action="store_true")
    parser.add_argument("--current-round-only", action="store_true",
                        help="Train solely on the current round's expert labels without BC replay")
    parser.add_argument("--extra-episodes", type=int, default=384)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--evaluations", type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 50 or args.rounds < 1 or args.extra_episodes < 0:
        parser.error("Expected 1–50 workers and at least one round")
    if args.budget_seconds is not None and (args.evaluations != 10 or args.budget_seconds <= 1270):
        parser.error("Budget planning requires ten evaluations and time for training")
    if args.evaluations is not None and not 1 <= args.evaluations <= args.rounds:
        parser.error("Evaluations must be between 1 and the initial round estimate")
    if args.current_round_only and args.budget_seconds is not None:
        parser.error("Specify a fixed round count for current-round-only training")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    seed_rng = np.random.default_rng(args.seed)
    checkpoint = args.base.resolve(strict=True)
    base = checkpoint.parent
    output = Path("results").resolve() / datetime.now().strftime("DAgger_%Y%m%d_%H%M%S")
    print(f"Run: {output}", flush=True)
    context = output / "context"
    saved, curriculum, dataset = gate_off_context(base, context, max_vph=TRAFFIC)
    model = context / "model.pt"
    road, mdp, train = load_saved_configs(str(model))
    cfg = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(output / "sumo"))
    set_weighted_od_flow_rates(output / "sumo/highway.rou.xml", TRAFFIC)
    configure_collision_gaps(cfg, ego_gap_m=road["EGO"]["collision_gap_m"],
                             background_gap_m=road["TRAFFIC"]["collision_gap_m"])
    template = ParallelSumo(1, dict(cfg_path=cfg, road=road["ROAD"], ego=road["EGO"],
        traffic=road["TRAFFIC"], mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
        action=mdp["ACTION"], reward=train["REWARD"]), output / "departures", seed=0,
        randomize_signals=True, randomize_departures=True, traffic_curriculum=curriculum)
    template._find_departures()
    combos = [(route, lane[2]) for group in read_ego_route_groups(output / "sumo/highway.ego.rou.xml")
              for route in group for lane in template.departures[route]]
    if len(set(combos)) != COMBOS or len(combos) != COMBOS or template.prototype.obs_dim != 51:
        raise ValueError(f"Expected 89 unique combinations and 51 observations, got {len(combos)} combinations")
    rounds = args.rounds
    evaluation_count = args.evaluations or rounds
    evaluation_rounds = {int(round(i * rounds / evaluation_count)) for i in range(1, evaluation_count + 1)}
    config = dict(
        baseline=str(checkpoint), seed=args.seed, traffic_vph=TRAFFIC, workers=args.workers,
        rounds=args.rounds, combos=combos, base_episodes=356, extra_episodes=args.extra_episodes,
        epochs=EPOCHS, budget_seconds=args.budget_seconds, evaluation_rounds=sorted(evaluation_rounds),
        roll_in=("learner only, native IDM queried every step on a separate SUMO"
                 if args.standard else "BC with one native IDM action every five steps"),
        failure_prior=[2, 2], failure_history_rounds=3, lane_change_safety=False,
        training_data="current_round_only" if args.current_round_only else "aggregate_with_bc_replay")
    config_file = output / "config.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(__file__, output / "run_bc51_dagger.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = load_actor(checkpoint, train["HPARAMS"], device)
    replay = teacher = None
    if not args.current_round_only:
        replay = load_replay(dataset, json.loads((base / "config.json").read_text()), seed_rng, device)
        with torch.no_grad():
            mu, _, logits, _ = actor(replay[0])
            teacher = mu.flatten().clone(), logits.softmax(-1)
    history, shards = [], []
    print(f"Evaluation rounds: {sorted(evaluation_rounds)}", flush=True)
    number = 0
    collection_times, training_times = [], []
    while number < rounds:
        number += 1
        started = perf_counter()
        folder = output / f"round_{number:02d}"
        folder.mkdir()
        cases, probabilities = plan_round(combos, history, args.seed, number, args.extra_episodes)
        (folder / "sampling.json").write_text(json.dumps(dict(probabilities=probabilities, cases=cases), indent=2))
        print(f"Round {number}: {len(cases)} episodes, extra probability range "
              f"{min(probabilities):.4f}–{max(probabilities):.4f}", flush=True)
        bank = TrafficSnapshotBank(cfg, curriculum, road["TRAFFIC"]["vehs_per_hour"],
                                   (TRAFFIC,), STEP, 0, refresh=False)
        round_shards = []
        try:
            scratch_root = "/dev/shm" if args.standard and Path("/dev/shm").is_dir() else None
            with TemporaryDirectory(prefix="dagger-workers-", dir=scratch_root) as scratch, ProcessPoolExecutor(
                    max_workers=args.workers, mp_context=get_context("spawn")) as pool:
                assignments = np.array_split(np.arange(len(cases)), args.workers)
                jobs = []
                for j, indices in enumerate(assignments):
                    shard = folder / f"expert_{j:02d}.npz"
                    round_shards.append(shard)
                    jobs.append(pool.submit(collect, [cases[i] for i in indices], checkpoint,
                                            model, cfg, Path(scratch) / f"worker_{j}",
                                            saved["traffic_curriculum"], bank.path, shard, args.standard))
                outcomes = [case for job in jobs for case in job.result()]
        finally:
            bank.close()
        history.append(outcomes)
        (folder / "outcomes.json").write_text(json.dumps(outcomes, indent=2))
        collection_times.append(perf_counter() - started)
        started = perf_counter()
        shards = round_shards if args.current_round_only else shards + round_shards
        fit(actor, replay, teacher, shards, seed_rng, folder / "training.json", standard=args.standard)
        training_times.append(perf_counter() - started)
        checkpoint = folder / "model.pt"
        torch.save(actor.state_dict(), checkpoint)
        if number == 1 and args.budget_seconds is not None:
            remaining = args.budget_seconds - 10 * 127
            # Each successive round trains on the accumulated labels.
            candidates = list(range(10, 101, 10))
            cost = lambda n: n * collection_times[0] + n * (n + 1) / 2 * training_times[0]
            rounds = min(candidates, key=lambda n: abs(cost(n) - remaining))
            evaluation_rounds = {i * rounds // 10 for i in range(1, 11)}
            config.update(rounds=rounds, evaluation_rounds=sorted(evaluation_rounds),
                          expected_total_seconds=cost(rounds) + 1270,
                          round_one_collection_seconds=collection_times[0],
                          round_one_training_seconds=training_times[0])
            config_file.write_text(json.dumps(config, indent=2) + "\n")
            print(f"Planned {rounds} rounds, estimated {cost(rounds) + 1270:.0f}s, "
                  f"evaluation rounds {sorted(evaluation_rounds)}", flush=True)
        if number in evaluation_rounds:
            evaluate(base, folder / "evaluation", args.workers, checkpoint=checkpoint,
                     levels=(TRAFFIC,), context_builder=gate_off_context,
                     training_plotter=lambda _fig, _grid, _run: None, policy_label="DAgger")
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
