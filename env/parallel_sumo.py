"""Isolated roads stepped together by one libsumo simulation."""

from copy import copy, deepcopy
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import signal
import sqlite3
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from env.collision_detection import sumo_collision_options
from env.ego_routes import group_unique_ego_routes
from env.ring_observation import prefix_geometry
from env.sumo_env import SumoHighwayEnv
from env.parallel_traffic import TrafficFocus, TrafficSchedule, parse_traffic_flows
from env.traffic_snapshots import TrafficSnapshotBank, pick_snapshot

import libsumo


def _replica_id(index, name):
    prefix = f"r{index}_"
    # Internal edges must keep their leading colon.
    return f":{prefix}{name[1:]}" if name.startswith(":") else prefix + name


def _signal_schedule(net, step_length, randomize_signals):
    programs = net.findall("tlLogic")
    if not randomize_signals:
        if any(set(phase.get("state")) - {"G", "g"} for phase in net.iter("phase")):
            raise ValueError("All-green replicas require permanently green traffic lights")
        return (), ()

    schedule = None
    for program in programs:
        phases = program.findall("phase")
        timings = [(phase.get("name"), float(phase.get("duration"))) for phase in phases]
        if (program.get("type") != "static" or float(program.get("offset", "0")) != 0
                or any("next" in phase.attrib for phase in phases)
                or (schedule is not None and timings != schedule)):
            raise ValueError("Signal randomization requires synchronized static signal programs")
        schedule = timings
    if not schedule or not any("r" in phase.get("state").lower() for phase in net.iter("phase")):
        raise ValueError("Signal randomization requires a cycling traffic-light program")
    ticks = [duration / step_length for _, duration in schedule]
    if any(tick <= 0 or not tick.is_integer() for tick in ticks):
        raise ValueError("Signal durations must be positive multiples of the SUMO step length")
    return tuple(program.get("id") for program in programs), tuple(np.cumsum(ticks, dtype=int))


def _write_network(net, destination, count):
    groups = ("edge", "tlLogic", "junction", "connection")
    if {node.tag for node in net} - {*groups, "location"}:
        raise ValueError("Unsupported elements in replicated network")
    location = net.find("location")
    xmin, ymin, xmax, ymax = map(float, location.get("convBoundary").split(","))
    stride_x, stride_y = xmax - xmin + 100, ymax - ymin + 100
    width = min(count, 32)
    offsets = [(i % width * stride_x, i // width * stride_y) for i in range(count)]
    bounds = f"{xmin},{ymin},{xmax + (width - 1) * stride_x},{ymax + ((count - 1) // width) * stride_y}"
    location.set("convBoundary", bounds)
    location.set("origBoundary", bounds)
    header = ET.tostring(ET.Element("net", net.attrib), encoding="utf-8", short_empty_elements=False)
    with destination.open("wb") as file:
        file.write(header.removesuffix(b"</net>") + b"\n")
        file.write(ET.tostring(location, encoding="utf-8"))
        for tag in groups:
            templates = net.findall(tag)
            for index, (dx, dy) in enumerate(offsets):
                for template in templates:
                    element = deepcopy(template)
                    for node in element.iter():
                        for attr, value in node.attrib.items():
                            if attr in {"id", "from", "to", "tl"}:
                                node.set(attr, _replica_id(index, value))
                            elif attr in {"incLanes", "intLanes", "via"}:
                                node.set(attr, " ".join(_replica_id(index, name) for name in value.split()))
                            elif attr in {"x", "y"}:
                                node.set(attr, f"{float(value) + (dx if attr == 'x' else dy):.2f}")
                            elif attr == "shape":
                                points = []
                                for point in value.split():
                                    x, y, *z = point.split(",")
                                    points.append(",".join([f"{float(x) + dx:.2f}", f"{float(y) + dy:.2f}", *z]))
                                node.set(attr, " ".join(points))
                    file.write(ET.tostring(element, encoding="utf-8"))
        file.write(b"</net>\n")
    return offsets


def _build_replicas(config, directory, count, step_length, randomize_signals,
                    traffic_curriculum, reference_vph):
    config = Path(config)
    inputs = ET.parse(config).getroot().find("input")
    net = directory / "parallel.net.xml"
    routes = directory / "parallel.rou.xml"
    types, original_routes, flows = [], {}, []
    for filename in inputs.find("route-files").get("value").split(","):
        for node in ET.parse(config.parent / filename).getroot():
            if node.tag in {"vType", "vTypeDistribution"}:
                types.append(node)
            elif node.tag == "route":
                original_routes[node.get("id")] = node.get("edges").split()
            elif node.tag == "flow" and traffic_curriculum is not None:
                flows.append(node)
            else:
                raise ValueError(f"Replicas require a road without background demand, got {node.tag}")

    directory.mkdir(parents=True, exist_ok=True)
    net_path = config.parent / inputs.find("net-file").get("value")
    source = ET.parse(net_path).getroot()
    ego_routes = {name: edges[0] for name, edges in original_routes.items()
                  if name == "r0" or name.startswith(("ego_d_", "ego_r_"))}
    traffic_flows = (parse_traffic_flows(flows, original_routes, net_path, reference_vph, traffic_curriculum)
                     if traffic_curriculum is not None else ())
    signal_ids, signal_end_ticks = _signal_schedule(source, step_length, randomize_signals)
    offsets = _write_network(source, net, count)
    route_xml = ET.Element("routes")
    route_xml.extend(types)
    for index in range(count):
        for name, edges in original_routes.items():
            ET.SubElement(route_xml, "route", id=_replica_id(index, name),
                          edges=" ".join(_replica_id(index, edge) for edge in edges))
    ET.ElementTree(route_xml).write(routes, encoding="utf-8")

    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=net.name)
    ET.SubElement(inputs, "route-files", value=routes.name)
    result = directory / "parallel.sumocfg"
    ET.ElementTree(root).write(result, encoding="utf-8")
    return (str(result), ego_routes, signal_ids, signal_end_ticks, traffic_flows,
            group_unique_ego_routes(original_routes), offsets)


class _ReplicaEgo(SumoHighwayEnv):
    _api = libsumo

    def _ego_is_present(self):
        return self.present

    def _destination_edge(self):
        return super()._destination_edge().split("_", 1)[1]

    def _neighbor(self, offset, leader, *, physical=False):
        if self.background is not None and self.background.vph:
            if offset and self.use_live_lane_neighbors:
                # getNeighbors can populate SUMO's per-tick leader cache.
                # A snapshot reset removes those cars before the next tick,
                # leaving stale leaders for native movement planning.
                front, rear = self._scan_lane_neighbors(offset)
                neighbor = front if leader else rear
                if (neighbor is None or neighbor[0] > self.visibility
                        or (neighbor[0] < 0 and not physical)):
                    return None
                return (max(neighbor[0], 0), neighbor[1]) if physical else neighbor
            return super()._neighbor(offset, leader, physical=physical)
        return None

    def _scan_lane_neighbors(self, direction):
        if self.background is not None and self.background.vph:
            return super()._scan_lane_neighbors(direction)
        return None, None


class ParallelSumo:
    def __init__(self, num_envs, env_kwargs, directory, seed, *, randomize_signals=False,
                 randomize_departures=False, independent_episode_rng=False,
                 traffic_curriculum=None, fixed_traffic_vph=None, traffic_snapshot_path=None,
                 uniform_route_departures=False,
                 fixed_training_routes=False, focused_envs=0):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.seed = seed
        self.randomize_departures = randomize_departures
        self.independent_episode_rng = independent_episode_rng or randomize_departures
        self.traffic_curriculum = traffic_curriculum
        self.fixed_traffic_vph = fixed_traffic_vph
        self.uniform_route_departures = uniform_route_departures
        self.route_departure_options = ()
        if uniform_route_departures and (not randomize_departures or fixed_training_routes or focused_envs):
            raise ValueError("Uniform route departures require randomized lanes and routes")
        self.fixed_training_routes = fixed_training_routes
        self.focused_envs = focused_envs
        self._snapshot_config = env_kwargs["cfg_path"]
        self._snapshot_path = traffic_snapshot_path
        self._snapshot_bank = None
        self._snapshot_db = None
        if traffic_curriculum and traffic_curriculum.snapshot_warmup_seconds and not randomize_signals:
            raise ValueError("Traffic snapshots require a cycling signal program")
        if fixed_traffic_vph is not None and (
                traffic_curriculum is None or not 0 <= fixed_traffic_vph <= traffic_curriculum.max_vph):
            raise ValueError("Fixed traffic requires a traffic profile and a value within its limits")
        self.prototype = _ReplicaEgo(**env_kwargs)
        self.prototype.background = None
        self.prototype.use_live_lane_neighbors = bool(
            traffic_curriculum and traffic_curriculum.snapshot_warmup_seconds)
        if (self.prototype.road_type != "nohyeong" or self.prototype.warmup_seconds
                or self.prototype.prefill_enabled or self.prototype.gui):
            raise ValueError("ParallelSumo requires a headless intersection without global warm-up")
        if randomize_signals and not self.prototype.signal_enabled:
            raise ValueError("Randomized signals require signal observations")
        (self.config, self.route_entries, self.signal_ids, self.signal_end_ticks,
         self.traffic_flows, self.route_groups, self.network_offsets) = (
            _build_replicas(env_kwargs["cfg_path"], directory, num_envs, self.prototype.step_length,
                            randomize_signals, traffic_curriculum,
                            self.prototype.traffic.get("vehs_per_hour", 0)))
        self.routes = tuple(sorted(self.route_entries))
        self.focus = None
        if focused_envs:
            if not fixed_training_routes or not 0 < focused_envs <= num_envs or traffic_curriculum is None:
                raise ValueError("Focused replicas require Phase 2 fixed-route training")
            if any(vph not in traffic_curriculum.snapshot_levels(None) for vph in TrafficFocus.levels):
                raise ValueError("Missing traffic snapshot levels for focused replicas")
            unique_routes = sum(self.route_groups, ())
            if (num_envs - focused_envs) % len(unique_routes):
                raise ValueError("Each worker's fixed replica count must be divisible by its route count")
            self.focus = TrafficFocus(unique_routes)
        self.departures = {}
        self.egos = []
        self.started = False
        self._live_background = set()

    def _start(self, config, *, show_gui=False):
        traffic_options = self.traffic_curriculum.sumo_options() if self.traffic_curriculum else []
        gui_options = ["--start"] if show_gui else []
        libsumo.start(["sumo-gui" if show_gui else "sumo", "-c", config,
                       "--step-length", str(self.prototype.step_length),
                       "--seed", str(self.seed), "--collision.action", "remove",
                       "--no-warnings", "true", "--no-step-log", "true", *traffic_options, *gui_options,
                       *sumo_collision_options()])
        self.started = True

    def _find_departures(self):
        if self.independent_episode_rng:
            self._start(self.config)
            try:
                for route, edge in self.route_entries.items():
                    replica_edge = _replica_id(0, edge)
                    lanes = []
                    for index in range(libsumo.edge.getLaneNumber(replica_edge)):
                        lane = f"{edge}_{index}"
                        allowed = libsumo.lane.getAllowed(_replica_id(0, lane))
                        disallowed = libsumo.lane.getDisallowed(_replica_id(0, lane))
                        if (not allowed or "passenger" in allowed) and "passenger" not in disallowed:
                            lanes.append((lane, 0.0, index))
                    if not lanes:
                        raise ValueError(f"No passenger departure lanes on {edge}")
                    if not self.randomize_departures:
                        configured = self.prototype._depart_lane_for(replica_edge)
                        if not configured.isdigit():
                            raise ValueError("Fixed departure requires a numeric or center lane")
                        fixed_lane = int(configured)
                        lanes = [lane for lane in lanes if lane[2] == fixed_lane]
                        if not lanes:
                            raise ValueError(f"Configured departure lane is unavailable on {edge}")
                    self.departures[route] = tuple(lanes)
            finally:
                self._close_simulation()
            return
        # Query the exact lane SUMO picks for every route under the baseline's
        # departLane setting. moveTo can then insert there without a global tick.
        ego = self.prototype
        self._start(ego.cfg_path)
        try:
            for route in self.routes:
                ego._spawn_ego(route)
                libsumo.simulationStep()
                lane = libsumo.vehicle.getLaneID(ego.ego_id)
                if not lane:
                    raise RuntimeError(f"SUMO did not insert the ego for route {route}")
                position = libsumo.vehicle.getLanePosition(ego.ego_id)
                self.departures[route] = ((lane, position, libsumo.vehicle.getLaneIndex(ego.ego_id)),)
                libsumo.vehicle.remove(ego.ego_id)
        finally:
            self._close_simulation()

    def _start_traffic_snapshots(self):
        if self.traffic_curriculum and self.traffic_curriculum.snapshot_warmup_seconds:
            levels = self.traffic_curriculum.snapshot_levels(self.fixed_traffic_vph)
            if levels:
                return TrafficSnapshotBank(
                    self._snapshot_config, self.traffic_curriculum,
                    self.prototype.traffic["vehs_per_hour"], levels,
                    self.prototype.step_length, self.seed, refresh=self.fixed_traffic_vph is None)
        return None

    def training_route_assignments(self, num_envs):
        """Balance all replicas globally before splitting them among workers."""
        if self.focus is not None:
            result = []
            for worker in range(num_envs // self.num_envs):
                routes = self.focus.routes
                fixed = list(routes) * ((self.num_envs - self.focused_envs) // len(routes))
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, num_envs, worker]))
                rng.shuffle(fixed)
                result.extend(fixed + [None] * self.focused_envs)
            return result
        direct, ring = self.route_groups
        if not direct or not ring:
            group = direct or ring
            if not group:
                raise ValueError("Fixed training routes require ego routes")
            routes = [group[i % len(group)] for i in range(num_envs)]
        else:
            n_ring = round(num_envs * self.prototype.ego_ring_prob)
            routes = [direct[i % len(direct)] for i in range(num_envs - n_ring)]
            routes.extend(ring[i % len(ring)] for i in range(n_ring))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, num_envs]))
        rng.shuffle(routes)
        return routes

    def _respawn(self, ego, route=None, traffic_vph=None, signal_tick=None, snapshot_tick=None, start_lane=None):
        if ego.present:
            libsumo.vehicle.remove(ego.ego_id)
        if route is None:
            route = ego.fixed_training_route
        departure = None
        if route is None and self.uniform_route_departures:
            route, departure = self.route_departure_options[
                ego.np_random.integers(len(self.route_departure_options))]
        if route is None:
            route = ego._sample_ego_route(self.route_groups) if ego.ego_route_random else "r0"
        ego.signal_offset_s = None
        offset = None
        if self.signal_ids:
            # When using the snapshot bank, this tick selects the paired
            # traffic and signal state. The signal is never randomized again.
            offset = ego.signal_random.integers(self.signal_end_ticks[-1]) if signal_tick is None else signal_tick
            ego.signal_offset_s = float(offset * ego.step_length)
            phase = int(np.searchsorted(self.signal_end_ticks, offset, side="right"))
            remaining = float((self.signal_end_ticks[phase] - offset) * ego.step_length)
            for signal in self.signal_ids:
                tls_id = _replica_id(ego.replica_index, signal)
                libsumo.trafficlight.setPhase(tls_id, phase)
                libsumo.trafficlight.setPhaseDuration(tls_id, remaining)
        candidates = self.departures[route]
        if start_lane is not None:
            departure = next((candidate for candidate in candidates if candidate[2] == start_lane), None)
            if departure is None:
                raise ValueError(f"Unavailable departure lane {start_lane} for {route}")
        if departure is None:
            departure = candidates[ego.lane_random.integers(len(candidates))]
        lane, position, lane_index = departure
        ego.start_lane = lane_index
        if ego.background is not None:
            vph = ego.background.vph if traffic_vph is None else traffic_vph
            ego.background.reset(libsumo, libsumo.simulation.getTime(), vph,
                                 ego.traffic_random, self._live_background)
            if vph and self._snapshot_db is not None:
                clearance = (libsumo.vehicletype.getLength("ego_type")
                             + libsumo.vehicletype.getMinGap("ego_type"))
                snapshot_key = int(offset) if snapshot_tick is None else snapshot_tick
                vehicles = pick_snapshot(self._snapshot_db, vph, snapshot_key,
                                         ego.traffic_random, lane, position, clearance)
                ego.background.restore(libsumo, ego.prefix, vehicles)
        ego.episode += 1
        ego.ego_id = _replica_id(ego.replica_index, f"ego_{ego.episode}")
        ego._spawn_ego(_replica_id(ego.replica_index, route), str(lane_index))
        libsumo.vehicle.moveTo(ego.ego_id, _replica_id(ego.replica_index, lane), position)
        ego.present = True
        return ego._get_obs()

    def reset(self, routes=None, *, replica_seeds=None, signal_ticks=None, snapshot_ticks=None, start_lanes=None):
        if routes is None and self.fixed_training_routes:
            routes = self.training_route_assignments(self.num_envs)
        valid_routes = set(self.routes) | ({None} if self.focus is not None else set())
        if routes is not None and (len(routes) != self.num_envs or set(routes) - valid_routes):
            raise ValueError("Expected one valid route per replica")
        if signal_ticks is not None and (
                not self.signal_ids or len(signal_ticks) != self.num_envs
                or any(not isinstance(tick, (int, np.integer)) or
                       not 0 <= tick < self.signal_end_ticks[-1] for tick in signal_ticks)):
            raise ValueError("Expected one valid signal tick per replica")
        if snapshot_ticks is not None and len(snapshot_ticks) != self.num_envs:
            raise ValueError("Expected one snapshot key per replica")
        if start_lanes is not None and len(start_lanes) != self.num_envs:
            raise ValueError("Expected one departure lane per replica")
        self._close_simulation()
        if not self.departures:
            self._find_departures()
        if self.uniform_route_departures:
            self.route_departure_options = tuple(
                (route, departure)
                for group in self.route_groups for route in group
                for departure in self.departures[route]
            )
        if self._snapshot_path is None:
            self._snapshot_bank = self._start_traffic_snapshots()
            if self._snapshot_bank is not None:
                self._snapshot_path = self._snapshot_bank.path
        if self._snapshot_path is not None and self._snapshot_db is None:
            self._snapshot_db = sqlite3.connect(f"file:{self._snapshot_path}?mode=ro", uri=True)
        self._start(self.config)
        self._live_background.clear()
        self.egos = []
        observations = []
        if replica_seeds is None:
            replica_seeds = np.random.SeedSequence(self.seed).spawn(self.num_envs)
        for index, seed in enumerate(replica_seeds):
            ego = copy(self.prototype)
            ego.replica_index = index
            ego.episode = 0
            ego.present = False
            # Recreating the parent keeps repeated resets with the same seed reproducible.
            seed = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key,
                                          pool_size=seed.pool_size)
            if self.independent_episode_rng:
                route_seed, lane_seed, signal_seed, traffic_seed = seed.spawn(4)
                ego.np_random = np.random.default_rng(route_seed)
                ego.signal_random = np.random.default_rng(signal_seed)
            else:
                ego.np_random = ego.signal_random = np.random.default_rng(seed)
                lane_seed, traffic_seed = seed.spawn(2)
            ego.lane_random = np.random.default_rng(lane_seed)
            ego.traffic_random = np.random.default_rng(traffic_seed)
            ego.prefix = partial(_replica_id, index)
            ego.background = TrafficSchedule(self.traffic_flows) if self.traffic_flows else None
            ego._rewarded_green_signals = set()
            ego._signal_stop_offsets = {
                (_replica_id(index, tid), link): offset
                for (tid, link), offset in self.prototype._signal_stop_offsets.items()
            }
            ego._guidance_geometry = {
                (_replica_id(index, a), _replica_id(index, b)): geometry
                for (a, b), geometry in self.prototype._guidance_geometry.items()
            }
            if ego.guidance_lateral_scale is not None:
                dx, dy = self.network_offsets[index]
                ego._guidance_geometry = {
                    key: (length, sine, cosine, (x + dx, y + dy))
                    for key, (length, sine, cosine, (x, y)) in ego._guidance_geometry.items()
                }
            ego._ring_geometry = prefix_geometry(self.prototype._ring_geometry, ego.prefix)
            ego._signal_stop_lines = {
                (_replica_id(index, tid), link): dict(line, lane_id=_replica_id(index, line["lane_id"]))
                for (tid, link), line in self.prototype._signal_stop_lines.items()
            }
            ego._signal_route_movements = {
                (ego.prefix(incoming), ego.prefix(outgoing)): [
                    (ego.prefix(tls), link, ego.prefix(lane), lane_index, position)
                    for tls, link, lane, lane_index, position in movements
                ]
                for (incoming, outgoing), movements in self.prototype._signal_route_movements.items()
            }
            ego.no_lane_change_edges = {_replica_id(index, edge) for edge in ego.no_lane_change_edges}
            self.egos.append(ego)
            route = routes[index] if routes is not None else None
            ego.fixed_training_route = route if self.fixed_training_routes else None
            vph = self.fixed_traffic_vph
            if vph is None and self.traffic_curriculum is not None:
                vph = self.traffic_curriculum.initial_vph
            ego.focused = self.focus is not None and route is None
            if ego.focused:
                ego.focus_random = np.random.default_rng(seed.spawn(1)[0])
                route, vph = self.focus.sample(ego.focus_random)
            tick = signal_ticks[index] if signal_ticks is not None else None
            snapshot = {} if snapshot_ticks is None else {"snapshot_tick": snapshot_ticks[index]}
            if start_lanes is not None:
                snapshot["start_lane"] = start_lanes[index]
            observations.append(self._respawn(ego, route, traffic_vph=vph, signal_tick=tick, **snapshot))
        return np.stack(observations)

    def _advance(self, egos, actions):
        states = [ego._prepare_step(action) for ego, action in zip(egos, actions, strict=True)]
        if self.traffic_flows:
            now = libsumo.simulation.getTime()
            for ego in egos:
                ego.background.spawn_due(libsumo, now, ego.prefix, ego.traffic_random)
        libsumo.simulationStep()
        active = set(libsumo.vehicle.getIDList())
        if self.traffic_flows:
            self._live_background = active | set(libsumo.simulation.getPendingVehicles())
        collided = set(libsumo.simulation.getCollidingVehiclesIDList())
        arrived = set(libsumo.simulation.getArrivedIDList())
        for ego, state in zip(egos, states, strict=True):
            ego.present = ego.ego_id in active
            yield ego._finish_step(
                state, collided=ego.ego_id in collided, arrived=ego.ego_id in arrived)

    def step(self, actions):
        transitions = []
        for ego, (obs, reward, terminated, truncated, info) in zip(
                self.egos, self._advance(self.egos, actions), strict=True):
            # The BC-to-PPO objective explicitly treats the deadline as a failure.
            timeout_obs = obs if truncated and not terminated and not ego.rw.get("bc_ppo_reward") else None
            if terminated or truncated:
                vph = self.fixed_traffic_vph
                route = None
                success = info["arrived"] and not (info["red_light_violation"] or info["collided"]
                                                   or info["wrong_lane"])
                if ego.focused:
                    self.focus.record(ego.ego_route_id.removeprefix(ego.prefix("")), ego.background.vph, success)
                    route, vph = self.focus.sample(ego.focus_random)
                elif vph is None and ego.background is not None:
                    vph = self.traffic_curriculum.after_episode(ego.background.vph, success)
                obs = self._respawn(ego, route=route, traffic_vph=vph)
            transitions.append((obs, reward, terminated, truncated, timeout_obs))
        return transitions

    def step_evaluation(self, indices, actions):
        """Advance only active evaluation egos without starting another episode."""
        egos = [self.egos[index] for index in indices]
        locations = [
            (libsumo.vehicle.getRoadID(ego.ego_id), libsumo.vehicle.getLanePosition(ego.ego_id))
            for ego in egos
        ]
        for ego, transition, (edge, position) in zip(
                egos, self._advance(egos, actions), locations, strict=True):
            transition[4].update(start_lane=ego.start_lane, signal_offset_s=ego.signal_offset_s,
                                 traffic_vph=ego.background.vph if ego.background is not None else 0)
            prefix = ego.prefix("")
            transition[4].update(lane_change_edge=edge.removeprefix(prefix),
                                 lane_change_position_m=position)
            clearance = ego.get_stop_clearance()
            if clearance is not None:
                transition[4].update(clearance)
            if transition[2] or transition[3]:
                if ego.present:
                    libsumo.vehicle.remove(ego.ego_id)
                    ego.present = False
                if ego.background is not None:
                    ego.background.reset(libsumo, libsumo.simulation.getTime(), 0,
                                         ego.traffic_random, self._live_background)
            yield transition

    def _traffic_levels(self):
        if self._snapshot_bank is not None:
            self._snapshot_bank.check()
        return [ego.background.vph for ego in self.egos] if self.traffic_flows else []

    def traffic_levels(self):
        if self.focus is not None:
            self.focus.update(self.focus.drain())
        return self._traffic_levels()

    def _close_simulation(self):
        if self.started:
            libsumo.close()
            self.started = False

    def close(self):
        self._close_simulation()
        if self._snapshot_db is not None:
            self._snapshot_db.close()
            self._snapshot_db = None
        if self._snapshot_bank is not None:
            self._snapshot_bank.close()
            self._snapshot_bank = None
            self._snapshot_path = None


def _run_sumo_worker(conn, env, replica_seeds):
    # The main process handles Ctrl+C and saves the current model.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while True:
            command, data = conn.recv()
            if command == "reset":
                conn.send((True, env.reset(data, replica_seeds=replica_seeds)))
            elif command == "step":
                rows = env.step(data)
                # One observation array instead of thousands of pickled arrays.
                conn.send((True, (np.stack([row[0] for row in rows]),
                                  [row[1:] for row in rows])))
            elif command == "close":
                break
            elif command == "traffic_levels":
                conn.send((True, env.traffic_levels()))
            elif command == "focus_report":
                conn.send((True, (env._traffic_levels(), env.focus.drain())))
            elif command == "focus_weights":
                env.focus.weights = data
                conn.send((True, None))
            else:
                raise ValueError(f"Unknown SUMO worker command: {command}")
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        conn.send((False, traceback.format_exc()))
    finally:
        conn.close()
        env.close()


class ShardedParallelSumo:
    """Collect synchronous rollouts from one independent SUMO per worker."""

    def __init__(self, template, num_workers):
        self.template = template
        self.prototype = template.prototype
        self.num_envs = template.num_envs * num_workers
        self.num_workers = num_workers
        self._workers = []
        self._snapshot_bank = None

    def _start(self):
        context = get_context("spawn")
        seeds = np.random.SeedSequence(self.template.seed).spawn(self.num_envs)
        size = self.template.num_envs
        for i in range(self.num_workers):
            conn, child = context.Pipe()
            process = context.Process(target=_run_sumo_worker,
                                      args=(child, self.template, seeds[i * size:(i + 1) * size]))
            try:
                process.start()
            except BaseException:
                conn.close()
                raise
            finally:
                child.close()
            self._workers.append((process, conn))

    def _exchange(self, command, payloads):
        for (_, conn), data in zip(self._workers, payloads, strict=True):
            conn.send((command, data))
        for i, (_, conn) in enumerate(self._workers):
            try:
                ok, result = conn.recv()
            except (EOFError, ConnectionError) as exc:
                process = self._workers[i][0]
                process.join(timeout=0.2)
                raise RuntimeError(f"SUMO worker {i} disconnected (exit code {process.exitcode})") from exc
            if not ok:
                raise RuntimeError(f"SUMO worker {i} failed:\n{result}")
            yield result

    def reset(self, routes=None):
        if routes is None and self.template.fixed_training_routes:
            routes = self.template.training_route_assignments(self.num_envs)
        if routes is not None and len(routes) != self.num_envs:
            raise ValueError("Expected one route per environment")
        if not self._workers:
            self._snapshot_bank = self.template._start_traffic_snapshots()
            if self._snapshot_bank is not None:
                self.template._snapshot_path = self._snapshot_bank.path
            self._start()
        size = self.template.num_envs
        batches = (None if routes is None else routes[i:i + size]
                   for i in range(0, self.num_envs, size))
        return np.concatenate(list(self._exchange("reset", batches)))

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError("Expected one action per environment")
        size = self.template.num_envs
        batches = (actions[i:i + size] for i in range(0, self.num_envs, size))
        return [(obs, *rest) for observations, rows in self._exchange("step", batches)
                for obs, rest in zip(observations, rows, strict=True)]

    def traffic_levels(self):
        if self.template.traffic_curriculum is None:
            return []
        if self._snapshot_bank is not None:
            self._snapshot_bank.check()
        if self.template.focus is not None:
            reports = list(self._exchange("focus_report", [None] * self.num_workers))
            levels, counts = zip(*reports, strict=True)
            self.template.focus.update(np.sum(counts, axis=0))
            list(self._exchange("focus_weights", [self.template.focus.weights] * self.num_workers))
            return [vph for shard in levels for vph in shard]
        return [value for shard in self._exchange("traffic_levels", [None] * self.num_workers)
                for value in shard]

    def close(self):
        for _, conn in self._workers:
            try:
                conn.send(("close", None))
            except (EOFError, ConnectionError):
                pass
            finally:
                conn.close()
        for process, _ in self._workers:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
        self._workers.clear()
        if self._snapshot_bank is not None:
            self._snapshot_bank.close()
            self._snapshot_bank = None
            self.template._snapshot_path = None
