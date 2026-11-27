"""Independent uniform demand for IDM background vehicles."""
from collections import Counter
import xml.etree.ElementTree as ET


def separate_idm_type(tree):
    """Keep IDM available as an explicit type and out of adaptive draws."""
    distribution = tree.find("./vTypeDistribution[@id='carDist']")
    idm = distribution.find("./vType[@id='car_idm']")
    distribution.remove(idm)
    idm.attrib.pop("probability", None)
    tree.getroot().insert(0, idm)
    return idm


class UniformIDM:
    """Independent equal-rate IDM demand with at most one pending request per route."""

    def __init__(self, flows, routes, lanes, rate, rng):
        self.flows = {flow.route: flow for flow in flows}
        if len(self.flows) != 24 or len({tuple(routes[r]) for r in self.flows}) != 24:
            raise ValueError("Uniform IDM requires 24 distinct background paths")
        self.rng = rng
        self.delay = 3600 * len(self.flows) / rate
        self.next_due = {r: rng.exponential(self.delay) for r in self.flows}
        self.entry = {r: [lane for lane, (edge, _) in lanes.items() if edge == routes[r][0]]
                      for r in self.flows}
        self.pending = {}
        self.requests = Counter()
        self.entered = Counter()

    def spawn(self, api, now, states, route_by_id, *, prefix=str):
        blocked = {lane for lane, pos, speed, _ in states.values() if pos < 10 and speed < 1}
        active_pending = set(api.simulation.getPendingVehicles())
        for route, flow in self.flows.items():
            if self.next_due[route] > now or self.pending.get(route) in active_pending:
                continue
            lanes = [lane for lane in self.entry[route] if lane not in blocked]
            if not lanes:
                continue
            vid = prefix(f"uniform_idm_{sum(self.requests.values())}")
            lane = str(self.rng.choice(lanes)).rsplit("_", 1)[1]
            api.vehicle.add(vid, prefix(route), typeID="car_idm", departLane=lane, departSpeed=flow.speed)
            self.pending[route] = vid
            self.requests[route] += 1
            route_by_id[vid] = route
            self.next_due[route] = now + self.rng.exponential(self.delay)
