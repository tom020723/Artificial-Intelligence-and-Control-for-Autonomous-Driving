"""Physical ring-merge observations; no SUMO priority/foe/open-link decisions."""
from collections import defaultdict
import math


EMPTY = [0.0, -1.0, 0.0, -1.0, 0.0, -1.0]


def ring_geometry(net, horizon=50.0, *, scope="legacy"):
    """Map each ring lane to its merge road and the other incoming lanes.

    The merge point is the start of the shared outgoing external road.
    Direct scope observes only the straight mainline approach lane.
    Connectivity is used only for physical sensing, not route/lane advice.
    """
    if scope not in ("legacy", "direct"):
        raise ValueError(f"Unknown ring merge scope: {scope}")
    lanes = {l.get('id'): l for e in net.findall('edge') for l in e.findall('lane')}
    external = {e.get('id'): e for e in net.findall('edge') if e.get('function') != 'internal'}
    conns = net.findall('connection')
    parents = defaultdict(set)
    for c in conns:
        source = c.get('from') + '_' + c.get('fromLane')
        target = c.get('via') or c.get('to') + '_' + c.get('toLane')
        parents[target].add(source)
    result = {}
    for edge in external:
        if 'RING_' not in edge:
            continue
        for lane in external[edge].findall('lane'):
            links = [c for c in conns if c.get('from') == edge and c.get('fromLane') == lane.get('index')]
            if not links:
                continue
            goals = {(c.get('to'), c.get('toLane')) for c in links}
            # Current ring roads each merge into exactly one external road.
            roads = {goal for goal, _ in goals}
            if len(roads) != 1:
                raise ValueError(f'Ambiguous ring merge: {edge}')
            goal = next(iter(roads))
            if scope == "direct":
                through = [c for c in conns if (c.get("to"), c.get("toLane")) in goals
                           and c.get("from") in external and "RING_" not in c.get("from")
                           and c.get("dir") == "s" and not c.get("tl")]
                # A1's second ring lane can feed two outgoing lanes. Prefer the
                # lane-preserving connection when choosing its mainline neighbor.
                through.sort(key=lambda c: abs(int(c.get("toLane")) - int(lane.get("index"))))
                mainline = (f'{through[0].get("from")}_{through[0].get("fromLane")}',) if through else ()
                result[lane.get("id")] = dict(goal=goal, lanes=mainline)
                continue
            approach = set()
            for c in conns:
                if (c.get('to'), c.get('toLane')) not in goals or c.get('from') == edge:
                    continue
                if c.get('from') not in external:
                    continue
                source = c.get('from') + '_' + c.get('fromLane')
                # Walk downstream internal lanes to the shared road.
                current = c.get('via')
                seen = set()
                while current and current not in seen:
                    seen.add(current); approach.add(current)
                    successors = [x for x in conns if x.get('from') + '_' + x.get('fromLane') == current]
                    current = next((x.get('via') for x in successors if x.get('via')), None)
                pending = [(source, 0.0)]
                visited = set()
                while pending:
                    current, distance = pending.pop()
                    if current in visited or current not in lanes:
                        continue
                    visited.add(current)
                    if current.startswith(edge + '_'):
                        continue
                    approach.add(current)
                    distance += float(lanes[current].get('length'))
                    if distance <= horizon:
                        pending.extend((parent, distance) for parent in parents[current])
            result[lane.get('id')] = dict(goal=goal, lanes=tuple(sorted(approach)))
    return result


def prefix_geometry(geometry, prefix):
    return {prefix(lane): dict(goal=prefix(plan['goal']), lanes=tuple(prefix(x) for x in plan['lanes']))
            for lane, plan in geometry.items()}


def ring_observation(api, vehicle_id, geometry, vmax, distance_scale=50.0, visibility=50.0):
    vehicle = api.vehicle
    plan = geometry.get(vehicle.getLaneID(vehicle_id))
    if plan is None:
        return list(EMPTY)
    own_distance = vehicle.getDrivingDistance(vehicle_id, plan['goal'], 0.0)
    result = [1.0, own_distance / distance_scale if own_distance >= 0 else -1.0]
    x, y = vehicle.getPosition(vehicle_id)
    candidates = {}
    for lane in plan['lanes']:
        for other in api.lane.getLastStepVehicleIDs(lane):
            if other == vehicle_id or other in candidates:
                continue
            ox, oy = vehicle.getPosition(other)
            if math.hypot(ox - x, oy - y) > visibility:
                continue
            distance = vehicle.getDrivingDistance(other, plan['goal'], 0.0)
            if distance < 0:
                continue
            candidates[other] = (distance, vehicle.getSpeed(other))
    for distance, speed in sorted(candidates.values())[:2]:
        result.extend((speed / vmax, distance / distance_scale))
    while len(result) < 6:
        result.extend((0.0, -1.0))
    return result
