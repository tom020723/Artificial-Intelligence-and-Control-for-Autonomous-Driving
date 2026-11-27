"""Collect SUMO expert observations/actions at a fixed TOTAL traffic demand.

python collect_background_data.py
Each worker runs an independent libsumo simulation.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time
import shutil
import xml.etree.ElementTree as ET

import numpy as np

# Import the environment first to locate SUMO's Python tools.
from env.sumo_env import SumoHighwayEnv
from env import road_builder
from env.road_config import ROAD, EGO, TRAFFIC, _CENTRAL_BACKGROUND_VPH, _RING_BACKGROUND_VPH
from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD
from env.control_timing import lane_change_timing_allowed
from env.background_demand import sample_background_demand
from env.collision_detection import configure_collision_gaps, junction_collisions, sumo_collision_options
import libsumo


class BackgroundObserver(SumoHighwayEnv):
    @property
    def _api(self):
        return libsumo

    def _ego_is_present(self):
        active_ids = getattr(self, "_active_ids", None)
        return super()._ego_is_present() if active_ids is None else self.ego_id in active_ids


def rebalance_origin_flows(directory, total_vph):
    """Allocate total demand by entry lanes, retaining each origin's route mix."""
    net = ET.parse(Path(directory)/'highway.net.xml').getroot()
    edges = {e.get('id'): e for e in net.findall('edge')}
    path = Path(directory)/'highway.rou.xml';tree=ET.parse(path)
    routes={r.get('id'):r.get('edges').split() for r in tree.getroot().findall('route')}
    grouped={}
    for flow in tree.getroot().findall('flow'):
        origin=flow.get('from') or routes[flow.get('route')][0]
        grouped.setdefault(origin,[]).append(flow)
    counts={edge:sum(1 for lane in edges[edge].findall('lane')
                    if lane.get('disallow')!='all' and
                    (not lane.get('allow') or 'passenger' in lane.get('allow').split()))
            for edge in grouped}
    total_lanes=sum(counts.values());allocation={}
    for edge,flows in grouped.items():
        demand=total_vph*counts[edge]/total_lanes;allocation[edge]=demand
        previous=sum(3600/float(flow.get('period')) for flow in flows)
        for flow in flows:
            rate=demand*(3600/float(flow.get('period')))/previous
            flow.set('period',f'{3600/rate:.10f}')
    tree.write(path,encoding='utf-8',xml_declaration=True)
    return allocation


def build_collection_road(directory, total_vph, origin_lane_proportional=False):
    """Build a private road and return its configuration with the measured demand."""
    traffic = deepcopy(TRAFFIC)
    traffic.update(background_enabled=True, vehs_per_hour=_CENTRAL_BACKGROUND_VPH,
                   keep_lane=False)
    # Enable autonomous/speed-gain changes without adding a keep-right preference.
    traffic.update(lc_speed_gain=1.0, lc_keep_right=0.0)
    for controller in traffic.get('controllers', {}).values():
        controller['keep_lane'] = False
    traffic['prefill']['enabled'] = False
    config = deepcopy(road_builder.NOHYEONG)
    config['all_signals_green'] = False
    for demand, vph in zip(config['ring_segments'] + config['loop_routes'],
                           _RING_BACKGROUND_VPH, strict=True):
        demand['vph'] = vph
    previous = road_builder.NOHYEONG
    try:
        road_builder.NOHYEONG = config
        cfg = road_builder.build(deepcopy(ROAD), deepcopy(EGO), traffic, str(directory))
    finally:
        road_builder.NOHYEONG = previous
    path = directory / 'highway.rou.xml'
    tree = ET.parse(path)
    flows = tree.getroot().findall('flow')
    total_rate = sum(1 / float(f.get('period')) for f in flows)
    if not total_rate:
        raise ValueError('No background traffic templates')
    for flow in flows:
        share = (1 / float(flow.get('period'))) / total_rate
        flow.set('period', f'{3600 / (total_vph * share):.10f}')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    if origin_lane_proportional:
        allocation = rebalance_origin_flows(directory, total_vph)
        traffic['origin_lane_proportional'] = True
        traffic['origin_requested_vph'] = allocation
        flows = ET.parse(path).getroot().findall('flow')
    measured = sum(3600 / float(f.get('period')) for f in flows)
    if not np.isclose(measured, total_vph):
        raise ValueError('Total traffic demand normalization failed')
    return cfg, traffic, measured, config


def lane_change_events(path, step_length):
    """Use SUMO's actual lateral events; edge lane-index resets are not changes."""
    events = {}
    for change in ET.parse(path).getroot().iter('change'):
        direction = int(change.get('dir'))
        tick = round(float(change.get('time')) / step_length)
        key = (change.get('id'), tick)
        if key in events:
            raise ValueError(f'Multiple lane changes in one control step: {key}')
        if direction not in (-1, 1):
            raise ValueError(f'Invalid lane-change direction: {direction}')
        events[key] = direction
    return events


def action_from_speeds(before, after, step_length, max_accel, max_decel):
    acceleration = (after - before) / step_length
    raw = acceleration / (max_accel if acceleration >= 0 else max_decel)
    return np.clip(raw, -1, 1), acceleration, abs(raw) > 1 + 1e-5


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def collect_episode(env, episode, seed, args, directory, extra_routes=None, traffic_schedule=None, dagger=None):
    lc_path = directory / f'lanechanges_{episode:03d}.xml'
    command = ['sumo', '-c', str(env.cfg_path), '--step-length', str(env.step_length),
               '--seed', str(seed), '--no-step-log', 'true', '--no-warnings', 'true',
               '--collision.action', 'remove', '--lanechange-output', str(lc_path),
               '--log', str(directory / f'sumo_{episode:03d}.log'), *sumo_collision_options()]
    rows, bad_vehicles = [], set()
    counts = Counter()
    route_paths = {}
    timing = time.monotonic()
    # Different warmup lengths also diversify the phase at the start of capture.
    warmup = (args.warmup_seconds + int(np.random.default_rng(seed).integers(0, 120))
              if traffic_schedule is None else 0)
    warmup_ticks = round(warmup / env.step_length)
    duration_ticks = round(args.seconds / env.step_length)
    sampled_routes = None
    if getattr(args, "weighted_od", False):
        base = Path(env.cfg_path).parent
        sampled_routes = directory / f"demand_{episode:03d}.rou.xml"
        sample_background_demand(base / "highway.rou.xml", args.traffic_vph,
                                 warmup + args.seconds, seed, output=sampled_routes)
        command += ["--route-files", f"{sampled_routes},{base / 'highway.ego.rou.xml'}"]
    elif extra_routes is not None:
        base = Path(env.cfg_path).parent
        route_file = ET.parse(env.cfg_path).getroot().find("input/route-files").get("value")
        command += ["--route-files", f"{base / route_file},{extra_routes}"]
    closed_only = getattr(args, "closed_trajectories", False)
    expert_entries = Counter()
    entered, endings = {}, {}
    if traffic_schedule is not None and not closed_only:
        command += ["--max-depart-delay", "125"]
    if dagger is not None:
        command += ["--save-state.precision", "12", "--save-state.rng", "true"]
    libsumo.start(command)
    try:
        if traffic_schedule is not None:
            rng = np.random.default_rng(seed)
            traffic_schedule.reset(libsumo, 0., args.traffic_vph, rng, set())
        for tick in range(warmup_ticks + duration_ticks):
            collecting = tick >= warmup_ticks
            before = {}
            allowed = {}
            protect = getattr(args, 'spawn_protect_m', None)
            if protect is not None:
                protected_type = getattr(args, 'spawn_protect_vehicle_type', None)
                for vid in libsumo.vehicle.getIDList():
                    edge = libsumo.vehicle.getRoadID(vid)
                    position = libsumo.vehicle.getLanePosition(vid)
                    index = libsumo.vehicle.getRouteIndex(vid)
                    if protected_type is not None:
                        if libsumo.vehicle.getTypeID(vid) != protected_type:
                            continue
                        allowed[vid] = edge.startswith(':') or index != 0 or position >= protect
                    else:
                        allowed[vid] = lane_change_timing_allowed(edge, position, index, protect)
                    libsumo.vehicle.setLaneChangeMode(vid, 1621 if allowed[vid] else 0)
            if collecting:
                collect_ids = libsumo.vehicle.getIDList()
                env._active_ids = set(collect_ids)
                included = getattr(args, "include_vehicle_type", None)
                excluded = getattr(args, "exclude_vehicle_type", ())
                for vid in sorted(collect_ids):
                    if closed_only and (vid not in entered or vid in endings):
                        continue
                    vehicle_type = libsumo.vehicle.getTypeID(vid)
                    if (included is not None and vehicle_type != included) or vehicle_type in excluded:
                        continue
                    corrections_before = env.leader_gap_geometry_corrections
                    obs = env.observe_vehicle(vid)
                    counts['observed_samples'] += 1
                    counts['leader_gap_geometry_corrected_samples'] += int(
                        env.leader_gap_geometry_corrections > corrections_before)
                    if obs.shape != (env.obs_dim,) or not np.isfinite(obs).all():
                        raise ValueError(f'Invalid observation for {vid}')
                    rid = libsumo.vehicle.getRouteID(vid)
                    route_paths[rid] = list(libsumo.vehicle.getRoute(vid))
                    before[vid] = (obs, libsumo.vehicle.getSpeed(vid),
                                   libsumo.vehicle.getRoadID(vid),
                                   libsumo.vehicle.getLaneIndex(vid), rid,
                                   vehicle_type)
            if traffic_schedule is not None:
                traffic_schedule.spawn_due(libsumo, libsumo.simulation.getTime(), str, rng)
            if dagger is not None:
                dagger.begin(env, before)
            libsumo.simulationStep()
            labels = dagger.finish() if dagger is not None else {}
            if traffic_schedule is not None:
                active = set(libsumo.vehicle.getIDList())
                traffic_schedule.ids.intersection_update(active | set(libsumo.simulation.getPendingVehicles()))
                if not closed_only:
                    for vid in libsumo.simulation.getDepartedIDList():
                        if vid.startswith("uniform_idm_"):
                            expert_entries[traffic_schedule.controller.route_by_id[vid]] += 1
            if closed_only:
                vehicle = libsumo.vehicle
                now = libsumo.simulation.getTime()
                for vid in libsumo.simulation.getDepartedIDList():
                    if vid in active and vehicle.getTypeID(vid) == "car_idm":
                        entered[vid] = (vehicle.getRouteID(vid), now)
                for outcome, ids in (
                    ("arrival", libsumo.simulation.getArrivedIDList()),
                    ("collision", libsumo.simulation.getCollidingVehiclesIDList()),
                    ("teleport", libsumo.simulation.getStartingTeleportIDList()),
                ):
                    for vid in ids:
                        if vid in entered:
                            endings.setdefault(vid, outcome)
            bad_vehicles.update(libsumo.simulation.getCollidingVehiclesIDList())
            bad_vehicles.update(libsumo.simulation.getStartingTeleportIDList())
            if not collecting:
                continue
            counts['departed'] += libsumo.simulation.getDepartedNumber()
            counts['arrived'] += libsumo.simulation.getArrivedNumber()
            counts['collision_events'] += len(libsumo.simulation.getCollidingVehiclesIDList())
            counts['teleport_events'] += len(libsumo.simulation.getStartingTeleportIDList())
            alive = set(libsumo.vehicle.getIDList())
            after_time = libsumo.simulation.getTime()
            for vid, (obs, speed, road, lane, rid, vtype) in before.items():
                if dagger is not None:
                    if vid not in labels:
                        counts["discard_missing_expert_label"] += 1
                        continue
                    accel, direction = labels[vid]
                    acceleration = accel * (env.max_accel if accel >= 0 else env.max_decel)
                    rows.append((obs, vid, after_time - env.step_length, accel, acceleration,
                                 False, speed, speed + acceleration * env.step_length,
                                 road, road, lane, lane, rid, vtype, True, direction))
                    continue
                if vid not in alive:
                    counts['disappeared_transitions'] += 1
                    continue
                accel_raw, acceleration, saturated = action_from_speeds(
                    speed, libsumo.vehicle.getSpeed(vid), env.step_length,
                    env.max_accel, env.max_decel)
                # The lane-change output uses BEGINNING-of-step timestamps;
                # reconciled below against directly observed same-edge changes.
                rows.append((obs, vid, after_time - env.step_length, accel_raw,
                             acceleration, saturated, speed, libsumo.vehicle.getSpeed(vid), road,
                             libsumo.vehicle.getRoadID(vid), lane,
                             libsumo.vehicle.getLaneIndex(vid), rid, vtype,
                             allowed.get(vid, True), None))
            if (tick - warmup_ticks + 1) % max(1, round(60/env.step_length)) == 0:
                if closed_only:
                    live = directory / f'progress_{episode:03d}.json'
                    tmp = live.with_suffix('.tmp')
                    save_json(tmp, dict(seconds=after_time, closed=dict(Counter(entered[v][0] for v in endings))))
                    tmp.replace(live)
                print(f'episode {episode}: {after_time-warmup:.0f}s, '
                      f'{len(rows):,} candidate pairs, {len(alive)} live vehicles', flush=True)
    finally:
        env._active_ids = None
        libsumo.close()
        if sampled_routes is not None:
            sampled_routes.unlink()
    events = lane_change_events(lc_path, env.step_length)
    selected, actions, physical, vehicles, steps, times, rids, types = [], [], [], [], [], [], [], []
    speeds, next_speeds = [], []
    allowed_labels = []
    for obs, vid, t, accel, acceleration, saturated, speed0, speed1, road0, road1, lane0, lane1, rid, vtype, allowed, expert_lane in rows:
        if (closed_only and vid not in endings) or (not closed_only and vid in bad_vehicles):
            counts['discard_bad_vehicle'] += 1
            continue
        # Saturating an infeasible expert action would silently change its label.
        if saturated:
            counts['discard_acceleration_out_of_range'] += 1
            continue
        if speed1 > env.vmax + 1e-5:
            counts['discard_speed_above_ego_limit'] += 1
            continue
        direction = expert_lane if expert_lane is not None else events.get((vid, round(t / env.step_length)), 0)
        if not allowed and direction:
            raise ValueError(f'Expert violated ego lane-change timing: {vid} t={t}')
        if expert_lane is None and road0 == road1 and not road0.startswith(':'):
            observed = lane1 - lane0
            if direction != observed:
                raise ValueError(f'Lane-change timing mismatch: {vid} t={t}: '
                                 f'event={direction}, lane delta={observed}')
            counts['checked_same_edge_labels'] += 1
        selected.append(obs)
        actions.append([accel, direction])
        physical.append(acceleration)
        speeds.append(speed0)
        next_speeds.append(speed1)
        vehicles.append(vid)
        times.append(t)
        steps.append(round(t/env.step_length))
        rids.append(rid)
        types.append(vtype)
        allowed_labels.append(allowed)
    if not selected:
        raise ValueError('No valid expert transitions')
    data = dict(observation=np.asarray(selected, np.float32),
                action_raw=np.asarray(actions, np.float32),
                episode=np.full(len(selected), episode, dtype=np.int32),
                vehicle_id=np.asarray(vehicles), step=np.asarray(steps, np.int32),
                time=np.asarray(times, np.float64), route_id=np.asarray(rids),
                vehicle_type=np.asarray(types), acceleration=np.asarray(physical, np.float32),
                speed=np.asarray(speeds, np.float32), next_speed=np.asarray(next_speeds, np.float32))
    if getattr(args, 'spawn_protect_m', None) is not None:
        data['lane_change_allowed'] = np.asarray(allowed_labels, np.bool_)
    if not env.observation_space.contains(data['observation'][0]):
        raise ValueError('Observation outside policy observation space')
    if closed_only:
        data["entry_time"] = np.asarray([entered[v][1] for v in vehicles])
        data["termination"] = np.asarray([endings[v] for v in vehicles])
    np.savez_compressed(directory / f'episode_{episode:03d}.npz', **data)
    directions = data['action_raw'][:, 1]
    counts.update(samples=len(selected), vehicles=len(set(vehicles)),
                  left_changes=int((directions == 1).sum()),
                  right_changes=int((directions == -1).sum()),
                  keep_lane=int((directions == 0).sum()),
                  red_samples=int((data['observation'][:, 31] == 1).sum()),
                  yellow_samples=int((data['observation'][:, 32] == 1).sum()),
                  green_samples=int((data['observation'][:, 33] == 1).sum()))
    report = dict(episode=episode, seed=seed, warmup_seconds=warmup,
                  collection_seconds=args.seconds, counts=dict(counts),
                  actual_departures_per_hour=counts['departed'] * 3600/args.seconds,
                  routes=dict(Counter(rids)), route_edges=route_paths,
                  vehicle_types=dict(Counter(types)), bad_vehicle_ids=sorted(bad_vehicles),
                  wall_seconds=round(time.monotonic()-timing, 3))
    save_json(directory / f'episode_{episode:03d}.json', report)
    if closed_only:
        report.update(trajectories_by_route=dict(Counter(entered[v][0] for v in set(vehicles))),
                      cutoff_by_route=dict(Counter(route for v, (route, _) in entered.items() if v not in endings)))
        save_json(directory / f'episode_{episode:03d}.json', report)
    if traffic_schedule is not None and not closed_only:
        report.update(idm_entries_by_route=dict(expert_entries),
                      idm_requests_by_route=dict(traffic_schedule.uniform_idm.requests),
                      adaptive_requests=traffic_schedule.controller.stats["requested"])
        save_json(directory / f'episode_{episode:03d}.json', report)
    print(f'episode {episode} complete: {len(selected):,} samples, '
          f'{counts["left_changes"] + counts["right_changes"]} lateral changes, '
          f'{report["wall_seconds"]:.1f}s wall time', flush=True)
    return report


def collection_totals(reports):
    return dict(samples=sum(r['counts']['samples'] for r in reports),
                lane_changes=sum(r['counts']['left_changes'] + r['counts']['right_changes']
                                 for r in reports))


def targets_reached(reports, target_samples, target_lane_changes):
    totals = collection_totals(reports)
    return (len(reports) >= 5 and bool(target_samples or target_lane_changes)
            and (not target_samples or totals['samples'] >= target_samples)
            and (not target_lane_changes or totals['lane_changes'] >= target_lane_changes))


def init_collection_worker(cfg, observation, traffic, args, directory):
    global worker_env, worker_args, worker_directory
    worker_env = BackgroundObserver(cfg, ROAD, EGO, SIMULATION, observation, ACTION, REWARD, traffic=traffic)
    worker_args, worker_directory = args, directory


def collect_worker_episode(episode):
    return collect_episode(worker_env, episode, worker_args.seed + episode, worker_args, worker_directory)


def collect_reports(env, cfg, observation, traffic, args, directory):
    if args.num_workers == 1:
        for ep in range(args.episodes):
            yield collect_episode(env, ep, args.seed + ep, args, directory)
        return

    workers = min(args.num_workers, args.episodes)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=init_collection_worker,
                             initargs=(cfg, observation, traffic, args, directory)) as pool:
        for start in range(0, args.episodes, workers):
            # Keep episode and seed ordering independent of worker completion order.
            yield from pool.map(collect_worker_episode, range(start, min(start + workers, args.episodes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traffic-vph', type=float, default=3000)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--num-workers', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=180)
    parser.add_argument('--warmup-seconds', type=float, default=250)
    parser.add_argument('--target-samples', type=int, default=110000000,
                        help='Stop after a complete episode once this many pairs and >=5 episodes exist; 0 disables')
    parser.add_argument('--target-lane-changes', type=int, default=0,
                        help='Required valid left+right labels, excluding warmup/discarded data; 0 disables')
    parser.add_argument('--origin-lane-proportional', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--layout', choices=['bc43','bc49'], default='bc49')
    parser.add_argument('--guidance-lateral-scale', type=float,
                        help='Add two ego-relative entrance offsets in bc49, divided by this scale in meters')
    parser.add_argument('--sharded', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep episode NPZ shards without a memory-heavy merged dataset')
    parser.add_argument('--min-free-gb', type=float, default=20.0)
    parser.add_argument('--wall-seconds', type=float, default=None, help='Collection wall budget, checked after each episode')
    parser.add_argument('--seed', type=int, default=99000)
    parser.add_argument('--spawn-protect-m', type=float, default=30.0,
                        help='Match ego timing: block native changes on internal edges and before this initial-edge distance')
    parser.add_argument('--exclude-vehicle-type', action='append', default=[],
                        help='Skip expert labels for this type while retaining it in traffic')
    parser.add_argument('--weighted-od', action='store_true',
                        help='Sample OD routes with 5% U-turns and 40.45% ring routes')
    parser.add_argument('--include-acc', action='store_true', help='Include ACC expert labels')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error('--num-workers must be positive')
    if not args.include_acc and 'car_acc' not in args.exclude_vehicle_type:
        args.exclude_vehicle_type.append('car_acc')
    if args.spawn_protect_m is not None and (not np.isfinite(args.spawn_protect_m) or args.spawn_protect_m < 0):
        parser.error('--spawn-protect-m must be finite and nonnegative')
    if (args.episodes < 2 or args.traffic_vph <= 0 or args.seconds <= 0
            or args.warmup_seconds < 0 or args.target_samples < 0 or args.target_lane_changes < 0
            or not all(np.isfinite(v) for v in [args.traffic_vph, args.seconds, args.warmup_seconds])):
        parser.error('Need >=2 episodes, positive finite flow/duration, and nonnegative warmup/target')
    if any(not np.isclose(v/SIMULATION['step_length'], round(v/SIMULATION['step_length']))
           for v in [args.seconds, args.warmup_seconds]):
        parser.error('Durations must be multiples of the simulation step length')
    directory = (args.out_dir or Path('data') / datetime.now().strftime('background_bc_%Y%m%d_%H%M%S')).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cfg, traffic, measured_vph, _ = build_collection_road(
        directory/'sumo', args.traffic_vph, args.origin_lane_proportional and not args.weighted_od)
    if args.weighted_od:
        traffic['demand_profile'] = 'weighted_od'
    collision_settings = configure_collision_gaps(cfg, ego_gap_m=EGO.get('collision_gap_m'),
                                                 background_gap_m=traffic.get('collision_gap_m'))
    observation = dict(OBSERVATION, layout=args.layout, ring_visibility=50.0, ring_distance_scale=50.0)
    if args.guidance_lateral_scale is not None:
        observation['guidance_lateral_scale'] = args.guidance_lateral_scale
    env = BackgroundObserver(cfg, ROAD, EGO, SIMULATION, observation, ACTION, REWARD, traffic=traffic)
    expected_dim = (49 if args.layout == 'bc49' else 43) + 2 * (args.guidance_lateral_scale is not None)
    if env.obs_dim != expected_dim:
        raise ValueError(f'Unexpected observation size for {args.layout}: {env.obs_dim}')
    config = {**vars(args), 'out_dir': str(directory), 'observation': observation,
              'ego': EGO, 'traffic': traffic, 'simulation': SIMULATION,
              'collision_detection': collision_settings,
              'filter': {'excluded_vehicle_types': args.exclude_vehicle_type},
              'sumo_version': libsumo.getVersion(),
              'action': ACTION, 'total_demand_vph': measured_vph,
              'label': 'observation(t) -> realized action during [t,t+dt]; SUMO native lane-change event',
              'train_validation_split': 'whole simulation episode, never individual vehicles or rows',
              'expert': 'native SUMO background controllers; no ego/PPO action injection'}
    if args.spawn_protect_m is not None:
        config['expert_control_timing'] = dict(spawn_protect=args.spawn_protect_m,
                                             block_internal_at_step_start=True)
    config['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                               for name in ['collect_background_data.py', 'env/sumo_env.py',
                                            'env/route_guidance.py', 'env/ring_observation.py', 'env/road_config.py', 'env/mdp_config.py',
                                            'env/road_builder.py', 'env/collision_detection.py']}
    if junction_collisions():
        config['junction_collisions'] = True
    save_json(directory/'config.json', config)
    reports = []
    collection_start = time.monotonic()
    with closing(collect_reports(env, cfg, observation, traffic, args, directory)) as episode_reports:
        for report in episode_reports:
            reports.append(report)
            totals = collection_totals(reports)
            reached = targets_reached(reports, args.target_samples, args.target_lane_changes)
            progress = dict(**totals, targets_reached=reached,
                            collection_wall_seconds=time.monotonic()-collection_start)
            if args.sharded:
                progress.update(episode_count=len(reports), last_episode=reports[-1]['episode'])
            else:
                progress['episodes'] = reports
            save_json(directory/'summary.json', progress)
            print(f'Total: {totals["samples"]:,} pairs, {totals["lane_changes"]:,} valid lane changes', flush=True)
            if shutil.disk_usage(directory).free < args.min_free_gb*1024**3:
                print('Stopping collection to preserve free disk space', flush=True)
                break
            if reached or (args.wall_seconds is not None and len(reports) >= 5
                           and time.monotonic()-collection_start >= args.wall_seconds):
                break
    if args.sharded:
        save_json(directory/'manifest.json', dict(format='episode_npz_shards',
                  shards=[f'episode_{ep:03d}.npz' for ep in range(len(reports))],
                  samples=collection_totals(reports)['samples'], obs_dim=env.obs_dim,
                  collection_wall_seconds=time.monotonic()-collection_start,
                  targets_reached=reached))
        print(f'Sharded dataset: {directory}', flush=True)
        return
    buffers = []
    for ep in range(len(reports)):
        with np.load(directory/f'episode_{ep:03d}.npz', allow_pickle=False) as source:
            buffers.append({k:source[k] for k in source.files})
    merged = {k:np.concatenate([b[k] for b in buffers]) for k in buffers[0]}
    np.savez_compressed(directory/'dataset.npz', **merged)
    print(f'Dataset: {directory / "dataset.npz"} ({len(merged["observation"]):,} pairs)', flush=True)
    if (args.target_samples or args.target_lane_changes) and not reached:
        raise SystemExit('Episode limit reached before targets. Saved partial dataset; see summary.json.')


if __name__ == '__main__':
    main()
