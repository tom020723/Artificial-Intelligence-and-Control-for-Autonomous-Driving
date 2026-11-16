"""Per-road background traffic for the shared SUMO clock."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrafficCurriculum:
    initial_vph: int = 0
    max_vph: int = 1500
    success_increment: int = 50
    failure_decrement: int = 150
    ring_traffic: bool = True

    def after_episode(self, vph, success):
        delta = self.success_increment if success else -self.failure_decrement
        return min(self.max_vph, max(0, vph + delta))


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
        # Ring traffic scales from zero to its road-config demand at the curriculum cap.
        reference = curriculum.max_vph if ring else central_vph
        result.append(TrafficFlow(route, flow.get("type"), flow.get("departLane", "best"),
                                  flow.get("departSpeed", "0"),
                                  3600 / float(flow.get("period")) / reference))
    if not result:
        raise ValueError("Background traffic requires a road with demand templates")
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
