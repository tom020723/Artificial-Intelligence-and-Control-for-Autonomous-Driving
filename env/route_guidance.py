"""Road-level navigation, independent of the vehicle's lane choice."""
import math


def choice_geometry(net, include_entrances=False):
    """Map (incoming, outgoing) roads to (incoming length, sin, cos).

    Only true road choices are included, not forced merges or edge splits.
    Lane 0 supplies a fixed road reference; the ego's lane is never consulted.
    """
    edges = {e.get('id'): e for e in net.findall('edge')
             if e.get('function') != 'internal'}
    outgoing = {}
    for c in net.findall('connection'):
        a, b = c.get('from'), c.get('to')
        if a in edges and b in edges:
            outgoing.setdefault(a, set()).add(b)

    def heading(edge, end):
        lane = edges[edge].find('lane')
        points = [tuple(map(float, p.split(',')[:2])) for p in lane.get('shape').split()]
        pairs = list(zip(points, points[1:]))
        if end:
            pairs.reverse()
        a, b = next((a, b) for a, b in pairs if math.dist(a, b) > 1e-6)
        return math.atan2(b[1] - a[1], b[0] - a[0])

    result = {}
    for a, targets in outgoing.items():
        if len(targets) < 2:
            continue
        for b in targets:
            angle = heading(b, False) - heading(a, True)
            result[a, b] = (float(edges[a].find('lane').get('length')),
                            math.sin(angle), math.cos(angle))
            if include_entrances:
                starts = [tuple(map(float, lane.get('shape').split()[0].split(',')[:2]))
                          for lane in edges[b].findall('lane')]
                entrance = tuple(sum(axis) / len(starts) for axis in zip(*starts))
                result[a, b] += (entrance,)
    return result


def navigation_observation(vehicle, vehicle_id, geometry, distance_scale=650.0,
                           lateral_scale=None):
    """Next two choices, with distances along the live route, including junctions."""
    route = vehicle.getRoute(vehicle_id)
    index = vehicle.getRouteIndex(vehicle_id)
    if index < 0:
        return [0.0] * (8 if lateral_scale is None else 10)
    # SUMO retains the incoming route index while inside its internal junction.
    if vehicle.getRoadID(vehicle_id).startswith(':'):
        index += 1
    obs = []
    lateral = []
    for a, b in zip(route[index:], route[index + 1:]):
        choice = geometry.get((a, b))
        if choice is None:
            continue
        length, sine, cosine = choice[:3]
        distance = vehicle.getDrivingDistance(vehicle_id, a, length)
        if distance < 0:  # Passed/unreachable choices must not become valid zero-distance guidance.
            continue
        obs.extend([1.0, distance / distance_scale, sine, cosine])
        if lateral_scale is not None:
            if not lateral:
                x, y = vehicle.getPosition(vehicle_id)
                heading = math.radians(vehicle.getAngle(vehicle_id))
                left_x, left_y = -math.cos(heading), math.sin(heading)
            entrance_x, entrance_y = choice[3]
            lateral.append(((x - entrance_x) * left_x + (y - entrance_y) * left_y)
                           / lateral_scale)
        if len(obs) == 8:
            break
    obs += [0.0] * (8 - len(obs))
    if lateral_scale is not None:
        obs += lateral + [0.0] * (2 - len(lateral))
    return obs
