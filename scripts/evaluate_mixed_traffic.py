"""Evaluate simultaneous BC vehicles in independently seeded SUMO traffic."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from multiprocessing import get_context
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_background_data import BackgroundObserver, libsumo
from env.parallel_sumo import _signal_schedule
from scripts.evaluate_fixed_routes import FixedTrafficSchedule, prepare_road, read_json, write_json
from utils.saved_configs import load_saved_configs
from utils.bc_policy import load_actor

MIX = {"krauss": .30, "idm": .25, "eidm": .15, "acc": .10, "ego": .20}
EGO_TYPE = "ego_type"
MAX_TRIP_SECONDS = 500
MAX_DEPART_DELAY = 125
DEFAULT_SEEDS = range(21701, 21713)
DEFAULT_SECONDS = 3600


def run_road(cfg, flows, lanes, suite, model, vph, seconds, seed, *, gui_delay=None,
             leader_gap_geometry=None, spawn_protect_m=0., ego_speed_cap=False,
             unrestricted_ego=False):
    import torch

    torch.set_num_threads(1)
    road, _, _ = load_saved_configs(str(suite / "reference/model.pt"))
    _, mdp, train = load_saved_configs(str(model))
    api = libsumo
    gui = gui_delay is not None
    if gui:
        import traci
        api = traci

    class MixedTrafficObserver(BackgroundObserver):
        @property
        def _api(self):
            return api

    observation = dict(mdp["OBSERVATION"])
    if leader_gap_geometry is not None:
        observation["leader_gap_geometry"] = leader_gap_geometry
    env = MixedTrafficObserver(str(cfg), road["ROAD"], road["EGO"], mdp["SIMULATION"],
                             observation, mdp["ACTION"], train["REWARD"],
                             traffic=road["TRAFFIC"])
    actor = load_actor(model, train["HPARAMS"], "cpu", obs_dim=env.obs_dim).eval()
    # The ego executes the policy's commands without rule-based restrictions.
    # The speed cap and entry-zone ban exist only as explicit comparison options.
    env.disable_ego_rule_barriers()
    env.lc_spawn_protect_only = spawn_protect_m > 0 and not unrestricted_ego
    env.lc_spawn_protect = spawn_protect_m
    env.ego_speed_cap_enabled = ego_speed_cap and not unrestricted_ego
    controllers = road["TRAFFIC"]["controllers"]
    mix = {name: dict(controllers[name], probability=share) if name != "ego"
           else dict(probability=share, speed_dev=0.0) for name, share in MIX.items()}
    schedule = FixedTrafficSchedule(flows, mix, lanes)
    schedule.types[-1] = EGO_TYPE

    traffic_seed, signal_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(traffic_seed)
    signal_ids, signal_ends = _signal_schedule(ET.parse(cfg.with_name("highway.net.xml")).getroot(),
                                              env.step_length, True)
    signal_tick = int(np.random.default_rng(signal_seed).integers(signal_ends[-1]))
    phase = int(np.searchsorted(signal_ends, signal_tick, side="right"))
    remaining = (signal_ends[phase] - signal_tick) * env.step_length
    pending, active, rows = {}, {}, []
    requests, departures = Counter(), Counter()
    observed_failures = defaultdict(set)
    arrivals_with_failures = 0
    peak_ego_speed_mps = 0.0
    ego_steps_above_speed_limit = 0
    steps = round(seconds / env.step_length)
    if not np.isclose(steps * env.step_length, seconds):
        raise ValueError("Simulation duration must be a multiple of the SUMO step length")

    def record(vid, entry, status, now):
        if not gui:
            rows.append(dict(seed=seed, vehicle=vid, route=entry["route"],
                             request_s=entry["request_s"], departure_s=entry.get("departure_s"),
                             start_lane=entry.get("start_lane"), end_s=now, status=status,
                             observed_failures=",".join(sorted(entry.get("observed_failures", ())))))

    command = ["sumo-gui" if gui else "sumo", "-c", str(cfg), "--step-length", str(env.step_length),
               "--seed", str(seed), "--no-step-log", "true", "--no-warnings", "true",
               "--collision.action", "warn" if unrestricted_ego else "remove",
               "--max-depart-delay", "-1" if unrestricted_ego else str(MAX_DEPART_DELAY)]
    if gui:
        command += ["--start", "--delay", str(gui_delay), "--quit-on-end"]
    api.start(command)
    try:
        for signal in signal_ids:
            api.trafficlight.setPhase(signal, phase)
            api.trafficlight.setPhaseDuration(signal, remaining)
        schedule.reset(api, 0, vph, rng, set())
        for _ in range(steps):
            now = api.simulation.getTime()
            for vid, route, vehicle_type in schedule.spawn_due(api, now, str, rng):
                requests[vehicle_type] += 1
                if vehicle_type == EGO_TYPE:
                    pending[vid] = dict(route=route, request_s=now)
            env._active_ids = set(api.vehicle.getIDList())
            before = {}
            if active:
                with torch.inference_mode():
                    obs = np.stack([env.observe_vehicle(vid) for vid in active])
                    actions = actor.det_action(torch.from_numpy(obs)).numpy()
                for vid, action in zip(active, actions, strict=True):
                    env.ego_id = vid
                    before[vid] = env._prepare_step(action)

            api.simulationStep()
            now = api.simulation.getTime()
            live = set(api.vehicle.getIDList())
            env._active_ids = live
            collided = set(api.simulation.getCollidingVehiclesIDList())
            arrived = set(api.simulation.getArrivedIDList())
            departed = api.simulation.getDepartedIDList()
            teleported = set(api.simulation.getStartingTeleportIDList())
            for vid in departed:
                if vid in pending:
                    entry = pending.pop(vid)
                    entry["departure_s"] = now
                    active[vid] = entry
                    if vid in live:
                        entry["start_lane"] = api.vehicle.getLaneIndex(vid)
                        api.vehicle.setSpeedMode(vid, 0)
                        api.vehicle.setLaneChangeMode(vid, 0)
                if vid in live:
                    departures[api.vehicle.getTypeID(vid)] += 1
            for vid, entry in list(active.items()):
                env.ego_id = vid
                if not gui and vid in live:
                    speed = api.vehicle.getSpeed(vid)
                    peak_ego_speed_mps = max(peak_ego_speed_mps, speed)
                    ego_steps_above_speed_limit += speed > env.vmax + 1e-6
                status = None
                if vid in before and vid in live:
                    signals, _, distance = before[vid][6:9]
                    if signals:
                        traveled = api.vehicle.getDistance(vid) - distance
                        if any(traveled > stop + 1e-6 and env._crossing_signal_color(tls, link, color) == "r"
                               for tls, link, stop, color in signals):
                            status = "red_light"
                if unrestricted_ego:
                    failures = entry.setdefault("observed_failures", set())
                    if status == "red_light":
                        failures.add(status)
                    if vid in collided:
                        failures.add("collision")
                    if vid in live and env._stuck_in_wrong_lane():
                        failures.add("wrong_lane")
                    if now - entry["departure_s"] >= MAX_TRIP_SECONDS:
                        failures.add("timeout")
                    for failure in failures:
                        observed_failures[failure].add(vid)
                    if vid in arrived:
                        status = "arrival"
                        arrivals_with_failures += bool(failures)
                    elif vid in teleported:
                        # SUMO manages the vehicle during teleportation.
                        status = "teleport"
                    elif vid not in live:
                        status = "removed"
                    else:
                        status = None
                elif status is None:
                    if vid in collided:
                        status = "collision"
                    elif vid in arrived:
                        status = "arrival"
                    elif vid in teleported:
                        status = "teleport"
                    elif vid not in live:
                        status = "removed"
                    elif env._stuck_in_wrong_lane():
                        status = "wrong_lane"
                    elif now - entry["departure_s"] >= MAX_TRIP_SECONDS:
                        status = "timeout"
                if status:
                    record(vid, entry, status, now)
                    del active[vid]
                    if not unrestricted_ego and (vid in live or vid in teleported):
                        api.vehicle.remove(vid)
            for vid, entry in list(pending.items()):
                if not unrestricted_ego and now - entry["request_s"] > MAX_DEPART_DELAY + 2 * env.step_length:
                    record(vid, entry, "not_inserted", now)
                    del pending[vid]
            schedule.ids.clear()
    finally:
        api.close()
    if gui:
        return
    print(f"Seed {seed}: {dict(Counter(row['status'] for row in rows))}, "
          f"cut off: {len(active)} driving, {len(pending)} awaiting insertion", flush=True)
    if len(rows) + len(active) + len(pending) != requests[EGO_TYPE]:
        raise RuntimeError(f"Seed {seed}: incomplete ego accounting")
    return rows, dict(seed=seed, signal_offset_s=signal_tick * env.step_length,
                      peak_ego_speed_mps=peak_ego_speed_mps,
                      ego_steps_above_speed_limit=ego_steps_above_speed_limit,
                      observed_failure_vehicles={name: len(ids) for name, ids in observed_failures.items()},
                      arrivals_with_failures=arrivals_with_failures,
                      requests=dict(requests), departures_during_window=dict(departures),
                      excluded_at_cutoff=dict(driving=len(active), awaiting_insertion=len(pending)),
                      outcomes=summarize(rows))


def summarize(rows):
    counts = Counter(row["status"] for row in rows)
    inserted = len(rows) - counts["not_inserted"]
    return dict(requested=len(rows), inserted=inserted, statuses=dict(counts),
                success_rate=counts["arrival"] / inserted if inserted else None,
                success_rate_per_request=counts["arrival"] / len(rows) if rows else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="Fixed-demand suite providing the road and calibrated flow weights")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vph", type=int, help="Total demand including ego (defaults to the suite demand)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECONDS,
                        help="Simulation duration per seed (default: 3600 simulated seconds)")
    parser.add_argument("--leader-gap-geometry", action=argparse.BooleanOptionalAction,
                        help="Override the checkpoint's geometry-based leader gap setting")
    parser.add_argument("--spawn-protect-m", type=float, default=0.,
                        help="Initial lane-change restriction in meters (default: 0, disabled)")
    parser.add_argument("--ego-speed-cap", action=argparse.BooleanOptionalAction, default=False,
                        help="Rule-based comparison only: cap ego speed commands at the road speed (default: off)")
    parser.add_argument("--unrestricted-ego", action="store_true",
                        help="Log failures without removing the vehicle, so it keeps driving")
    args = parser.parse_args()
    if not np.isfinite(args.spawn_protect_m) or args.spawn_protect_m < 0:
        parser.error("--spawn-protect-m must be finite and nonnegative")
    suite, model = args.suite.resolve(strict=True), args.model.resolve(strict=True)
    spec = read_json(suite / "suite.json")
    _, mdp, _ = load_saved_configs(str(model))
    geometry = (args.leader_gap_geometry if args.leader_gap_geometry is not None
                else bool(mdp["OBSERVATION"].get("leader_gap_geometry", False)))
    vph = spec["traffic_vph"] if args.vph is None else args.vph
    if (spec.get("adaptive_traffic") or min(vph, args.workers, args.seconds) < 1
            or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Use a fixed-demand suite, positive limits, and distinct nonnegative seeds")
    seconds = args.seconds
    experiment = model.parent.parent if model.parent.name == "training" else model.parent
    out = args.output.resolve() if args.output else experiment / "evaluation_mixed"
    out.mkdir()
    cfg, flows, lanes = prepare_road(suite, out, spec)
    print(f"{vph:,} vehicles/hour, {len(args.seeds)} seeds, {seconds:,} simulated seconds per seed", flush=True)
    rows, runs = [], []
    with ProcessPoolExecutor(min(args.workers, len(args.seeds)), mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(run_road, cfg, flows, lanes, suite, model, vph, seconds, seed,
                            leader_gap_geometry=geometry, spawn_protect_m=args.spawn_protect_m,
                            ego_speed_cap=args.ego_speed_cap, unrestricted_ego=args.unrestricted_ego)
                for seed in args.seeds]
        for job in jobs:
            results, run = job.result()
            rows.extend(results)
            runs.append(run)
    by_route = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)
    fields = ("seed", "vehicle", "route", "request_s", "departure_s", "start_lane", "end_s", "status",
              "observed_failures")
    with (out / "route_evaluation.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    overall = summarize(rows)
    if args.unrestricted_ego:
        driving_at_cutoff = sum(run["excluded_at_cutoff"]["driving"] for run in runs)
        clean_arrivals = sum(row["status"] == "arrival" and not row["observed_failures"] for row in rows)
        overall.update(clean_arrivals=clean_arrivals, driving_at_cutoff=driving_at_cutoff,
                       arrival_rate_including_cutoff=overall["statuses"].get("arrival", 0)
                       / max(1, overall["inserted"] + driving_at_cutoff),
                       clean_arrival_rate_including_cutoff=clean_arrivals
                       / max(1, overall["inserted"] + driving_at_cutoff))
    write_json(out / "evaluation_summary.json",
               dict(checkpoint=str(model), checkpoint_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    suite=str(suite), traffic_vph=vph, requested_type_mix=MIX,
                    warmup_seconds=0, simulation_seconds=seconds, runs=runs, leader_gap_geometry=geometry,
                    spawn_protect_m=args.spawn_protect_m,
                    ego_speed_cap=args.ego_speed_cap and not args.unrestricted_ego,
                    unrestricted_ego=args.unrestricted_ego, ego_lane_change_gate=False,
                    ego_internal_lane_changes=True,
                    connected_lane_bumper_gaps=bool(mdp["OBSERVATION"].get("connected_lane_bumper_gaps")),
                    signed_adjacent_gaps=bool(mdp["OBSERVATION"].get("signed_adjacent_gaps")),
                    spawn_progress_scale=mdp["OBSERVATION"].get("spawn_progress_scale", 0.),
                    overall=overall, by_route={route: summarize(group) for route, group in sorted(by_route.items())}))
    print(f"Overall: {overall} · {out}", flush=True)


if __name__ == "__main__":
    main()
