"""Lane discharge measurements and queue-dependent admission for the road viewer."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import math
import heapq
import xml.etree.ElementTree as ET


@dataclass
class LaneMeter:
    lane: str
    cycle_seconds: float
    bootstrap_vph: float = 0.
    short_ratio: float = .3
    long_ratio: float = .6
    queue_ratio: float = 0.
    state: str = 'short'
    green: bool | None = None
    departures: int = 0
    demand_seen: bool = False
    complete_green: bool = False
    samples: deque = field(default_factory=lambda: deque(maxlen=5))
    completed_greens: int = 0
    cycle_queue_peak: float = 0.
    last_cycle_queue_peak: float = 0.
    saw_green: bool = False
    control_queue_ratio: float | None = None
    queue_length_m: float = 0.
    approach_length_m: float = 0.
    queued_vehicles: int = 0
    bootstrap_cap_vph: float = math.inf
    residual_queued: int = 0
    nonclearing_greens: int = 0
    draining: bool = False
    shared_drain_factor: float = 1.
    committed_before_stop: int = 0

    def observe(self, green, departed, queued, ratio, committed=None):
        if green and not self.green:
            if self.saw_green:
                self.last_cycle_queue_peak = self.cycle_queue_peak
            self.saw_green = True
            self.cycle_queue_peak = ratio
        else:
            self.cycle_queue_peak = max(self.cycle_queue_peak, ratio)
        if self.green:
            self.departures += departed
        if self.green and not green:
            if self.demand_seen:
                self.samples.append(self.departures)
                self.completed_greens += 1
            if self.complete_green:
                # A moving vehicle still upstream at green end has not been
                # discharged and will join the next queue; include that pipeline.
                remaining = queued if committed is None else committed
                self.nonclearing_greens = (self.nonclearing_greens + 1
                                           if remaining >= 2 and remaining >= self.residual_queued else 0)
                self.residual_queued = remaining
                service = self.baseline_vph * self.cycle_seconds / 3600
                if remaining <= 1:
                    self.draining = False
                    self.nonclearing_greens = 0
                elif remaining > service or self.nonclearing_greens >= 2:
                    self.draining = True
        if green and not self.green:
            self.departures = 0
            # The first observed green could be partial: do not train on it.
            self.demand_seen = False
            self.complete_green = self.green is False
        if green and self.complete_green and queued >= 2:
            self.demand_seen = True
        self.green = green
        self.queue_ratio = ratio

    @property
    def mean_departures(self):
        return sum(self.samples) / len(self.samples) if self.samples else None

    @property
    def baseline_vph(self):
        if not self.samples:
            return min(self.bootstrap_vph, self.bootstrap_cap_vph)
        # A small probe allows recovery after a genuinely blocked green.
        return max(.2, self.mean_departures) * 3600 / self.cycle_seconds

    def budget(self):
        ratio = self.queue_ratio if self.control_queue_ratio is None else self.control_queue_ratio
        if ratio <= self.short_ratio:
            self.state = 'short'
        elif ratio >= self.long_ratio:
            self.state = 'long'
        if ratio >= .9:
            return 0.
        factor = min(self.drain_factor, self.shared_drain_factor)
        normal = 1.1 if self.state == 'short' else .9
        return self.baseline_vph * (min(normal, factor) if factor < 1. else normal)

    @property
    def drain_factor(self):
        if not self.draining:
            return 1.
        service = self.baseline_vph * self.cycle_seconds / 3600
        # Reserve half the residual backlog per green before adding new demand.
        # At >= two greens of backlog, pause this physical lane's new arrivals.
        return max(0., 1. - self.residual_queued / (2 * max(.2, service)))


class LaneInflowController:
    """Allocate each signal lane's observed budget among compatible routes.

    Controls requested arrivals, never removes live vehicles. Unsignalized slip
    routes retain their nominal demand. Pending insertion and entry occupancy
    guard all routes, including unsignalized ones.
    """
    def __init__(self, net_path, flows, routes, max_vph, rng, short_ratio=.3, long_ratio=.6):
        root = ET.parse(net_path).getroot()
        self.flows, self.routes = tuple(flows), routes
        self.max_vph, self.rng = max_vph, rng
        self.lanes = {lane.get('id'): (edge.get('id'), float(lane.get('length')))
                      for edge in root.findall('edge') if not edge.get('id').startswith(':')
                      for lane in edge.findall('lane')}
        self.all_lanes = {lane.get('id'): (edge.get('id'), float(lane.get('length')))
                          for edge in root.findall('edge') for lane in edge.findall('lane')}
        all_connections = [c.attrib for c in root.findall('connection')]
        self.internal_links = defaultdict(list)
        for c in all_connections:
            if c['from'].startswith(':'):
                self.internal_links[f"{c['from']}_{c['fromLane']}"].append(c)
        self.connections = [c.attrib for c in root.findall('connection')
                            if not c.get('from').startswith(':')]
        cycles = {t.get('id'): sum(float(p.get('duration')) for p in t.findall('phase'))
                  for t in root.findall('tlLogic')}
        self.controlled = defaultdict(list)
        for c in self.connections:
            if c.get('tl'):
                self.controlled[f"{c['from']}_{c['fromLane']}"].append(c)
        self.meters = {lane: LaneMeter(lane, cycles[cs[0]['tl']], short_ratio=short_ratio,
                                       long_ratio=long_ratio)
                       for lane, cs in self.controlled.items()}
        self.options = {}
        self.shares = defaultdict(dict)
        for i, flow in enumerate(flows):
            pairs = set(zip(routes[flow.route], routes[flow.route][1:]))
            options = [lane for lane, cs in self.controlled.items()
                       if any((c['from'], c['to']) in pairs for c in cs)]
            self.options[i] = options
            nominal = max_vph * flow.rate_per_central_vph
            for lane in options:
                share = nominal / len(options)
                self.shares[lane][i] = share
                self.meters[lane].bootstrap_vph += share
        # Treat every approach as continuous to the central stop line, including
        # intermediate junction internals and every legal upstream lane branch.
        central = {lane: self._corridor(lane) for lane in self.lanes
                   if self.lanes[lane][0] in {f'A{i}_in' for i in range(1, 6)}}
        self.queue_roots = {}
        self.corridors = {}
        for lane in self.meters:
            roots = [name for name, segments in central.items()
                     if any(part[0] == lane for part in segments)]
            if not roots:
                raise ValueError(f'Signal lane has no continuous central approach: {lane}')
            merged = {}
            for name in roots:
                for segment, offset, length in central[name]:
                    if segment not in merged or offset < merged[segment][0]:
                        merged[segment] = (offset, length)
            self.queue_roots[lane] = roots
            self.corridors[lane] = [(name, *value) for name, value in sorted(merged.items(),
                                                                           key=lambda item: item[1][0])]
        corridor_lanes = {lane: {part[0] for part in corridor}
                          for lane, corridor in self.corridors.items()}
        self.shared = {lane: [other for other in self.meters
                              if corridor_lanes[lane] & corridor_lanes[other]]
                       for lane in self.meters}
        # Bootstrap conservatively until a route has a measured green. Multiple
        # signal meters on one physical approach lane share this initial budget.
        # The 2.5 s headway is an initial tuning assumption, not measured capacity.
        programs = {t.get('id'): t.findall('phase') for t in root.findall('tlLogic')}
        self.service_routes = {lane: {flows[i].route for i in self.shares[lane]}
                               for lane in self.meters}
        for lane, meter in self.meters.items():
            cs = self.controlled[lane]
            green_seconds = sum(float(p.get('duration')) for p in programs[cs[0]['tl']]
                                if any(p.get('state', '')[int(c['linkIndex']):int(c['linkIndex']) + 1]
                                       in ('G', 'g') for c in cs))
            if green_seconds:
                initial_capacity = green_seconds / 2.5 * 3600 / meter.cycle_seconds
                shared_demand = sum(self.meters[other].bootstrap_vph for other in self.shared[lane])
                meter.bootstrap_cap_vph = initial_capacity * meter.bootstrap_vph / shared_demand if shared_demand else 0.
        self.rates = [max_vph * f.rate_per_central_vph for f in flows]
        self.next_due = [self._headway(rate) for rate in self.rates]
        self.entry_choices = {(i, lane): self._departure_lanes(flow.route, lane)
                              for i, flow in enumerate(flows) for lane in self.options[i]}
        self.previous = {}
        self.previous_signals = {}
        self.serial = 0
        self.stats = Counter()
        self.last_update = -10.
        self.route_by_id = {}
        self.gate_weights = {}
        self.update_rates(0.)

    def _corridor(self, lane):
        arm = self.lanes[lane][0].split('_')[0]
        result, distances, pending = {}, {lane: 0.}, [(0., lane)]
        while pending:
            offset, current = heapq.heappop(pending)
            if offset != distances[current]:
                continue
            edge, length = self.lanes[current]
            result[current] = (offset, length)
            for c in self.connections:
                if c['to'] != edge or c['toLane'] != current.rsplit('_', 1)[1]:
                    continue
                if not c['from'].startswith(arm + '_in'):
                    continue
                via = self._internal_path(c)
                upstream_offset = offset + length
                for internal in reversed(via):
                    internal_length = self.all_lanes[internal][1]
                    if internal not in result or upstream_offset < result[internal][0]:
                        result[internal] = (upstream_offset, internal_length)
                    upstream_offset += internal_length
                source = f"{c['from']}_{c['fromLane']}"
                if upstream_offset < distances.get(source, math.inf):
                    distances[source] = upstream_offset
                    heapq.heappush(pending, (upstream_offset, source))
        return [(name, *value) for name, value in sorted(result.items(), key=lambda item: item[1][0])]

    def _internal_path(self, connection):
        path, current = [], connection.get('via')
        while current and current not in path:
            path.append(current)
            links = [c for c in self.internal_links[current]
                     if c['to'] == connection['to'] and c['toLane'] == connection['toLane']]
            if len(links) != 1:
                raise ValueError(f'Ambiguous internal continuation at {current}')
            current = links[0].get('via')
        return path

    def _departure_lanes(self, route, stop_lane):
        edges = self.routes[route]
        target_edge, _ = self.lanes[stop_lane]
        index = edges.index(target_edge)
        costs = {int(stop_lane.rsplit('_', 1)[1]): 0}
        for source, dest in reversed(list(zip(edges[:index], edges[1:index + 1]))):
            previous = {}
            for c in self.connections:
                if c['from'] != source or c['to'] != dest:
                    continue
                src, dst = int(c['fromLane']), int(c['toLane'])
                cost = min(abs(dst - lane) + value for lane, value in costs.items())
                previous[src] = min(previous.get(src, math.inf), cost)
            if not previous:
                raise ValueError(f'No departure path to {stop_lane} on {route}')
            costs = previous
        best = min(costs.values())
        return [lane for lane, cost in costs.items() if cost == best]

    def _headway(self, rate):
        return self.rng.exponential(3600 / rate) if rate > 0 else math.inf

    def observe(self, states, signals, teleported=()):
        """states maps vehicle id to (lane, position, speed, length)."""
        halted = defaultdict(list)
        for lane, pos, speed, length in states.values():
            if speed <= 1.:
                halted[lane].append((pos, length))
        ignored = set(teleported)
        for lane, meter in self.meters.items():
            cs = self.controlled[lane]
            green = any(signals[c['tl']][int(c['linkIndex'])] in 'Gg' for c in cs)
            departed = 0
            for vehicle, old in self.previous.items():
                if old[0] != lane or vehicle in ignored or vehicle not in states:
                    continue
                new_lane = states[vehicle][0]
                if any(self.previous_signals.get(c['tl'], '')[int(c['linkIndex']):int(c['linkIndex']) + 1]
                       in ('G', 'g') and
                       (new_lane == c.get('via') or self.lanes.get(new_lane, (None,))[0] == c['to'])
                       for c in cs):
                    departed += 1
            candidates = []
            corridor = self.corridors[lane]
            for upstream, offset, length in corridor:
                candidates.extend((offset + length - pos, size)
                                  for pos, size in halted[upstream])
            # Keep upstream slow traffic visible even when a green or a minor
            # junction opens a gap ahead of it. Branch lengths are not summed.
            end = max((distance + size for distance, size in candidates), default=0.)
            queued = len(candidates)
            length = max(offset + size for _, offset, size in corridor)
            meter.queue_length_m = min(length, end)
            meter.approach_length_m = length
            meter.queued_vehicles = queued
            corridor_ids = {part[0] for part in corridor}
            # Queue pressure is shared, but service-learning demand is specific
            # to the movement at this stop line (e.g. U-turn versus left turn).
            service_queued = sum(
                speed <= 1. and old_lane in corridor_ids and
                (vehicle not in self.route_by_id or
                 self.route_by_id[vehicle] in self.service_routes[lane])
                for vehicle, (old_lane, _, speed, _) in states.items())
            committed = sum(old_lane in corridor_ids and
                            (vehicle not in self.route_by_id or
                             self.route_by_id[vehicle] in self.service_routes[lane])
                            for vehicle, (old_lane, _, _, _) in states.items())
            meter.committed_before_stop = committed
            meter.observe(green, departed, service_queued, min(1., end / length), committed)
        self.previous = states
        self.previous_signals = signals

    def update_rates(self, now):
        if now - self.last_update < 10:
            return False
        self.last_update = now
        for lane, meter in self.meters.items():
            # A green opening must not erase a long queue or let another route
            # keep filling the same physical U-turn/left-turn approach lane.
            meter.control_queue_ratio = max(
                max(self.meters[other].queue_ratio, self.meters[other].cycle_queue_peak,
                    self.meters[other].last_cycle_queue_peak) for other in self.shared[lane])
            meter.shared_drain_factor = min(self.meters[other].drain_factor
                                            for other in self.shared[lane])
        budgets = {lane: meter.budget() for lane, meter in self.meters.items()}
        rates = []
        for i, flow in enumerate(self.flows):
            if not self.options[i]:
                rates.append(self.max_vph * flow.rate_per_central_vph)
            else:
                rates.append(sum(budgets[lane] * self.shares[lane][i] /
                                 self.meters[lane].bootstrap_vph for lane in self.options[i]))
        scale = min(1., self.max_vph / sum(rates)) if sum(rates) else 1.
        self.rates = [rate * scale for rate in rates]
        self.gate_weights = {
            i: [budgets[lane] * self.shares[lane][i] / self.meters[lane].bootstrap_vph
                for lane in self.options[i]] for i in self.options if self.options[i]}
        # Resample on rate changes: no backlog of missed generation events.
        self.next_due = [now + self._headway(rate) for rate in self.rates]
        return True

    def spawn_due(self, api, now, states, *, prefix=str):
        pending = Counter(self.route_by_id.get(v) for v in api.simulation.getPendingVehicles())
        entries = defaultdict(list)
        for lane, pos, speed, length in states.values():
            edge = self.lanes.get(lane, (None,))[0]
            if edge:
                entries[edge].append((pos, speed))
        for i, (flow, rate) in enumerate(zip(self.flows, self.rates)):
            if self.next_due[i] > now:
                continue
            self.next_due[i] = now + self._headway(rate)
            entry = self.routes[flow.route][0]
            lane = flow.lane
            if self.options[i]:
                weights = self.gate_weights.get(i, [1.] * len(self.options[i]))
                weight_sum = sum(weights)
                if not weight_sum:
                    continue
                gate = self.rng.choice(self.options[i], p=[w / weight_sum for w in weights])
                lane = str(self.rng.choice(self.entry_choices[i, gate]))
            # One pending vehicle per route prevents an uncontrolled inlet reservoir.
            blocked = (any(old_lane == f'{entry}_{lane}' and pos < 10 and speed < 1
                           for old_lane, pos, speed, _ in states.values()) if lane.isdigit()
                       else bool(entries[entry]) and all(speed < 1 for _, speed in entries[entry]))
            if pending[flow.route] or blocked:
                self.stats['admissions_withheld'] += 1
                continue
            vehicle = prefix(f'adaptive_{self.serial}')
            self.serial += 1
            api.vehicle.add(vehicle, prefix(flow.route), typeID=flow.vehicle_type,
                            departLane=lane, departSpeed=flow.speed)
            self.route_by_id[vehicle] = flow.route
            self.stats['requested'] += 1

    def rows(self, now):
        return [dict(time=now, lane=lane, queue_ratio=m.queue_ratio, state=m.state,
                     queue_length_m=m.queue_length_m, approach_length_m=m.approach_length_m,
                     queued_vehicles=m.queued_vehicles, queue_roots=' '.join(self.queue_roots[lane]),
                     residual_after_green=m.residual_queued, nonclearing_greens=m.nonclearing_greens,
                     draining=m.draining, drain_factor=m.drain_factor,
                     shared_drain_factor=m.shared_drain_factor,
                     committed_before_stop=m.committed_before_stop,
                     cycle_queue_peak=m.cycle_queue_peak, last_cycle_queue_peak=m.last_cycle_queue_peak,
                     control_queue_ratio=m.control_queue_ratio,
                     mean_green_departures=m.mean_departures, eligible_greens=m.completed_greens,
                     baseline_vph=m.baseline_vph, lane_budget_vph=m.budget(),
                     total_requested_vph=sum(self.rates)) for lane, m in sorted(self.meters.items())]
