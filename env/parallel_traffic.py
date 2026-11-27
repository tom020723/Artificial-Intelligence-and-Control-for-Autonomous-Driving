"""Per-road background traffic for the shared SUMO clock."""

from dataclasses import dataclass, replace
from math import gcd

import numpy as np


@dataclass(frozen=True)
class TrafficCurriculum:
    initial_vph: int = 0
    max_vph: int = 1500
    success_increment: int = 50
    failure_decrement: int = 150
    ring_traffic: bool = True
    total_vph: bool = False
    snapshot_warmup_seconds: int = 0
    max_depart_delay_seconds: int | None = None
    ring_reference_vph: int | None = None

    def sumo_options(self):
        if self.max_depart_delay_seconds is None:
            return []
        return ["--max-depart-delay", str(self.max_depart_delay_seconds)]

    def after_episode(self, vph, success):
        delta = self.success_increment if success else -self.failure_decrement
        return min(self.max_vph, max(0, vph + delta))

    def snapshot_levels(self, fixed_vph):
        if fixed_vph is not None:
            return (fixed_vph,) if fixed_vph else ()
        spacing = gcd(self.initial_vph, self.max_vph, self.success_increment, self.failure_decrement)
        return tuple(range(spacing, self.max_vph + 1, spacing)) if spacing else ()


@dataclass(frozen=True)
class TrafficFlow:
    route: str
    vehicle_type: str
    lane: str
    speed: str
    rate_per_central_vph: float


def parse_traffic_flows(flows, routes, net_path, central_vph, curriculum):
    """Convert XML demand into route templates; no static flow is loaded in SUMO."""
    import sumolib

    net = None
    result = []
    for flow in flows:
        route = flow.get("route")
        ring = route is None
        if ring:
            if not curriculum.ring_traffic:
                continue
            if net is None:
                net = sumolib.net.readNet(str(net_path))
            route = "background_" + flow.get("id")
            waypoints = [flow.get("from"), *flow.get("via", "").split(), flow.get("to")]
            edges = []
            for start, end in zip(waypoints, waypoints[1:]):
                path, _ = net.getShortestPath(net.getEdge(start), net.getEdge(end), vClass="passenger")
                if path is None:
                    raise ValueError(f"No background route for {flow.get('id')}: {start} -> {end}")
                edges.extend(edge.getID() for edge in (path if not edges else path[1:]))
            routes[route] = edges
        # Keep route proportions stable when the total-demand limit changes.
        ring_reference = curriculum.ring_reference_vph or curriculum.max_vph
        reference = ring_reference if ring else central_vph
        result.append(TrafficFlow(route, flow.get("type"), flow.get("departLane", "best"),
                                  flow.get("departSpeed", "0"),
                                  3600 / float(flow.get("period")) / reference))
    if not result:
        raise ValueError("Background traffic requires a road with demand templates")
    if curriculum.total_vph:
        total = sum(flow.rate_per_central_vph for flow in result)
        result = [replace(flow, rate_per_central_vph=flow.rate_per_central_vph / total)
                  for flow in result]
    return tuple(result)


class TrafficSchedule:
    """One replica's spawn schedule and ownership of its background vehicles."""

    def __init__(self, flows):
        self.flows = flows
        self.cumulative = np.cumsum([flow.rate_per_central_vph for flow in flows])
        self.ids = set()
        self.next_departure = float("inf")
        self.vph = 0
        self.serial = 0

    _second_stage_lanes = None

    @classmethod
    def second_stage_internal_lanes(cls, api):
        """Internal lanes that continue another internal lane (after a junction's waiting point)."""
        if cls._second_stage_lanes is None:
            # getLinks reports the junction lane used to reach the next lane in field 4 ("via").
            cls._second_stage_lanes = frozenset(
                link[4] for lane in api.lane.getIDList() if lane.startswith(":")
                for link in api.lane.getLinks(lane) if link[4])
        return cls._second_stage_lanes

    def restore(self, api, prefix, vehicles):
        # A vehicle placed by moveTo on the second stage of a two-stage junction lane (left turn or
        # U-turn past its waiting point) never moves again and blocks every crossing movement, which
        # gridlocks the junction for the whole episode. Leave those few vehicles out of the restore.
        skipped = self.second_stage_internal_lanes(api)
        restored = {}
        for route, vehicle_type, lane, position, speed, speed_factor, _ in vehicles:
            if lane.startswith(":") and prefix(lane) in skipped:
                continue
            vehicle = prefix(f"bg_{self.serial}")
            self.serial += 1
            api.vehicle.add(vehicle, prefix(route), typeID=vehicle_type)
            api.vehicle.moveTo(vehicle, prefix(lane), position)
            api.vehicle.setSpeedFactor(vehicle, speed_factor)
            api.vehicle.setPreviousSpeed(vehicle, speed)
            self.ids.add(vehicle)
            restored[vehicle] = route
        return restored

    def reset(self, api, now, vph, rng, live_ids):
        if self.ids:
            for vehicle in sorted(self.ids & live_ids):
                api.vehicle.remove(vehicle)
            self.ids.clear()
        self.vph = vph
        self.next_departure = now + self._headway(rng) if vph else float("inf")

    def _headway(self, rng):
        return rng.exponential(3600 / (self.vph * self.cumulative[-1]))

    def spawn_due(self, api, now, prefix, rng):
        while self.next_departure <= now:
            idx = self.cumulative.searchsorted(rng.random() * self.cumulative[-1], side="right")
            flow = self.flows[idx]
            vehicle = prefix(f"bg_{self.serial}")
            self.serial += 1
            api.vehicle.add(vehicle, prefix(flow.route), typeID=flow.vehicle_type,
                            departLane=flow.lane, departSpeed=flow.speed)
            self.ids.add(vehicle)
            self.next_departure += self._headway(rng)


class TrafficFocus:
    """Sample route/demand pairs in proportion to their recent failure estimates."""

    levels = (1500, 2000, 2500, 3000)

    def __init__(self, routes):
        self.routes = tuple(routes)
        self.index = {pair: i for i, pair in enumerate(
            (route, vph) for route in self.routes for vph in self.levels)}
        self.pairs = tuple(self.index)
        self.pending = np.zeros((len(self.pairs), 2), dtype=np.int64)
        self.total = self.pending.copy()
        self.recent = self.pending.astype(float)
        self.weights = np.full(len(self.pairs), 1 / len(self.pairs))

    def sample(self, rng):
        return self.pairs[rng.choice(len(self.pairs), p=self.weights)]

    def record(self, route, vph, success):
        self.pending[self.index[route, vph], int(not success)] += 1

    def drain(self):
        result = self.pending.copy()
        self.pending.fill(0)
        return result

    def update(self, counts):
        self.total += counts
        self.recent *= .98
        self.recent += counts
        # Eight virtual episodes at 50% failure stabilize small samples.
        failure = (self.recent[:, 1] + 4) / (self.recent.sum(axis=1) + 8)
        self.weights = failure / failure.sum()

    def summary(self):
        return [{"route": route, "traffic_vph": vph, "episodes": int(counts.sum()),
                 "failures": int(counts[1]), "sampling_probability": float(weight)}
                for (route, vph), counts, weight in zip(self.pairs, self.total, self.weights, strict=True)]
