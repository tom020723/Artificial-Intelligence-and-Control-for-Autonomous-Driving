"""Isolated Phase 1 roads stepped together by one libsumo simulation."""

from copy import copy, deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from env.sumo_env import SumoHighwayEnv

import libsumo


def _replica_id(index, name):
    prefix = f"r{index}_"
    # Internal edges must keep their leading colon.
    return f":{prefix}{name[1:]}" if name.startswith(":") else prefix + name


def _write_network(source, destination, count):
    net = ET.parse(source).getroot()
    groups = ("edge", "tlLogic", "junction", "connection")
    if {node.tag for node in net} - {*groups, "location"}:
        raise ValueError("Unsupported elements in Phase 1 network")
    if any(set(phase.get("state")) - {"G", "g"} for phase in net.iter("phase")):
        raise ValueError("Replicas require permanently green traffic lights")
    location = net.find("location")
    xmin, ymin, xmax, ymax = map(float, location.get("convBoundary").split(","))
    stride_x, stride_y = xmax - xmin + 100, ymax - ymin + 100
    width = min(count, 32)
    bounds = f"{xmin},{ymin},{xmax + (width - 1) * stride_x},{ymax + ((count - 1) // width) * stride_y}"
    location.set("convBoundary", bounds)
    location.set("origBoundary", bounds)
    header = ET.tostring(ET.Element("net", net.attrib), encoding="utf-8", short_empty_elements=False)
    with destination.open("wb") as file:
        file.write(header.removesuffix(b"</net>") + b"\n")
        file.write(ET.tostring(location, encoding="utf-8"))
        for tag in groups:
            templates = net.findall(tag)
            for index in range(count):
                dx, dy = index % width * stride_x, index // width * stride_y
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


def _build_replicas(config, directory, count):
    config = Path(config)
    inputs = ET.parse(config).getroot().find("input")
    net = directory / "parallel.net.xml"
    routes = directory / "parallel.rou.xml"
    types, original_routes = [], {}
    for filename in inputs.find("route-files").get("value").split(","):
        for node in ET.parse(config.parent / filename).getroot():
            if node.tag in {"vType", "vTypeDistribution"}:
                types.append(node)
            elif node.tag == "route":
                original_routes[node.get("id")] = node.get("edges").split()
            else:
                raise ValueError(f"Replicas require a road without background demand, got {node.tag}")

    directory.mkdir(parents=True, exist_ok=True)
    _write_network(config.parent / inputs.find("net-file").get("value"), net, count)
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
    return str(result), tuple(sorted(original_routes))


class _ReplicaEgo(SumoHighwayEnv):
    _api = libsumo

    def _ego_is_present(self):
        return self.present

    def _destination_edge(self):
        return super()._destination_edge().split("_", 1)[1]

    def _neighbor(self, offset, leader):
        # One vehicle per disconnected road and no background demand.
        return None

    def _scan_lane_neighbors(self, direction):
        return None, None


class ParallelSumo:
    def __init__(self, num_envs, env_kwargs, directory, seed):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.seed = seed
        self.prototype = _ReplicaEgo(**env_kwargs)
        if (self.prototype.road_type != "nohyeong" or self.prototype.warmup_seconds
                or self.prototype.prefill_enabled or self.prototype.gui):
            raise ValueError("ParallelSumo requires the headless, traffic-free Phase 1 road")
        self.config, self.routes = _build_replicas(env_kwargs["cfg_path"], directory, num_envs)
        self.departures = {}
        self.egos = []
        self.started = False

    def _start(self, config):
        libsumo.start(["sumo", "-c", config, "--step-length", str(self.prototype.step_length),
                       "--seed", str(self.seed), "--collision.action", "remove",
                       "--no-warnings", "true", "--no-step-log", "true"])
        self.started = True

    def _find_departures(self):
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
                self.departures[route] = (lane, position, libsumo.vehicle.getLaneIndex(ego.ego_id))
                libsumo.vehicle.remove(ego.ego_id)
        finally:
            self.close()

    def _respawn(self, ego, route=None):
        if ego.present:
            libsumo.vehicle.remove(ego.ego_id)
        if route is None:
            route = ego._sample_ego_route(self.routes) if ego.ego_route_random else "r0"
        lane, position, lane_index = self.departures[route]
        ego.episode += 1
        ego.ego_id = _replica_id(ego.replica_index, f"ego_{ego.episode}")
        ego._spawn_ego(_replica_id(ego.replica_index, route), str(lane_index))
        libsumo.vehicle.moveTo(ego.ego_id, _replica_id(ego.replica_index, lane), position)
        ego.present = True
        return ego._get_obs()

    def reset(self):
        self.close()
        if not self.departures:
            self._find_departures()
        self._start(self.config)
        self.egos = []
        observations = []
        for index, seed in enumerate(np.random.SeedSequence(self.seed).spawn(self.num_envs)):
            ego = copy(self.prototype)
            ego.replica_index = index
            ego.episode = 0
            ego.present = False
            ego.np_random = np.random.default_rng(seed)
            ego._rewarded_green_signals = set()
            ego._signal_stop_offsets = {
                (_replica_id(index, tid), link): offset
                for (tid, link), offset in self.prototype._signal_stop_offsets.items()
            }
            ego._signal_stop_lines = {
                (_replica_id(index, tid), link): dict(line, lane_id=_replica_id(index, line["lane_id"]))
                for (tid, link), line in self.prototype._signal_stop_lines.items()
            }
            ego.no_lane_change_edges = {_replica_id(index, edge) for edge in ego.no_lane_change_edges}
            self.egos.append(ego)
            observations.append(self._respawn(ego))
        return np.stack(observations)

    def step(self, actions):
        states = [ego._prepare_step(action) for ego, action in zip(self.egos, actions, strict=True)]
        libsumo.simulationStep()
        active = set(libsumo.vehicle.getIDList())
        collided = set(libsumo.simulation.getCollidingVehiclesIDList())
        arrived = set(libsumo.simulation.getArrivedIDList())
        transitions = []
        for ego, state in zip(self.egos, states, strict=True):
            ego.present = ego.ego_id in active
            obs, reward, terminated, truncated, _ = ego._finish_step(
                state, collided=ego.ego_id in collided, arrived=ego.ego_id in arrived)
            timeout_obs = obs if truncated and not terminated else None
            if terminated or truncated:
                obs = self._respawn(ego)
            transitions.append((obs, reward, terminated, truncated, timeout_obs))
        return transitions

    def close(self):
        if self.started:
            libsumo.close()
            self.started = False
