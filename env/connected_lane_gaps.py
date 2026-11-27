"""Measure bumper clearance on the directed SUMO lane network."""

from collections import defaultdict
from heapq import heappop, heappush


class ConnectedLaneGaps:
    def __init__(self, visibility, max_vehicle_length, *, signed_adjacent=False):
        self.visibility = visibility
        self.max_vehicle_length = max_vehicle_length
        self.signed_adjacent = signed_adjacent
        self.lengths = {}
        self.forward = {}
        self.backward = defaultdict(list)
        self.frame = None
        self.vehicles = {}

    def _load_network(self, api):
        for lane in api.lane.getIDList():
            self.lengths[lane] = api.lane.getLength(lane)
            successors = tuple(dict.fromkeys(link[4] or link[0] for link in api.lane.getLinks(lane)))
            self.forward[lane] = successors
            for successor in successors:
                self.backward[successor].append(lane)

    def _segments(self, lane, position, ego_length):
        """Yield nearby lane intervals in coordinates relative to the reference lane."""
        yield lane, 0.0
        for forward, links in ((True, self.forward), (False, self.backward)):
            start = self.lengths[lane] - position if forward else position
            limit = self.visibility + (self.max_vehicle_length if forward else ego_length)
            pending = [(start, other) for other in links.get(lane, ())]
            pending.sort()
            seen = {lane}
            while pending:
                distance, other = heappop(pending)
                if distance > limit:
                    break
                if other in seen:
                    continue
                seen.add(other)
                length = self.lengths[other]
                yield other, position + distance if forward else position - distance - length
                for successor in links.get(other, ()):
                    if successor not in seen:
                        heappush(pending, (distance + length, successor))

    def _vehicles_on(self, api, lane, cache):
        if not cache or lane not in self.vehicles:
            vehicle = api.vehicle
            snapshot = [(vid, vehicle.getLanePosition(vid), vehicle.getLength(vid), vehicle.getSpeed(vid))
                        for vid in api.lane.getLastStepVehicleIDs(lane) if vehicle.getLaneID(vid) == lane]
            if not cache:
                return snapshot
            self.vehicles[lane] = snapshot
        return self.vehicles[lane]

    def around(self, api, ego, offsets, *, cache=False):
        if not self.forward:
            self._load_network(api)
        if cache:
            frame = api.simulation.getTime()
            if frame != self.frame:
                self.vehicles.clear()
                self.frame = frame

        vehicle = api.vehicle
        own_lane = vehicle.getLaneID(ego)
        own_position = vehicle.getLanePosition(ego)
        ego_length = vehicle.getLength(ego)
        reference_lane, reference_position = own_lane, own_position
        # Inside a junction, index the adjacent lanes at the outgoing edge.
        # The projected ego position is upstream of that edge's start.
        seen = set()
        while reference_lane.startswith(":"):
            next_lanes = self.forward.get(reference_lane, ())
            if len(next_lanes) != 1 or reference_lane in seen:
                reference_lane = None
                break
            seen.add(reference_lane)
            reference_position -= self.lengths[reference_lane]
            reference_lane = next_lanes[0]

        neighbors = []
        for offset in offsets:
            if offset == 0:
                lane, position = own_lane, own_position
            elif reference_lane is not None:
                edge, index = reference_lane.rsplit("_", 1)
                lane, position = f"{edge}_{int(index) + offset}", reference_position
            else:
                neighbors.append((None, None))
                continue
            if lane not in self.lengths:
                neighbors.append((None, None))
                continue

            front = rear = None
            signed = self.signed_adjacent and offset != 0
            for segment, origin in self._segments(lane, position, ego_length):
                for vid, front_pos, length, speed in self._vehicles_on(api, segment, cache):
                    if vid == ego:
                        continue
                    relative_front = origin + front_pos - position
                    if relative_front >= 0:
                        gap = relative_front - length
                        if not signed:
                            gap = max(0.0, gap)
                        if gap <= self.visibility and (front is None or gap < front[0]):
                            front = (gap, speed)
                    else:
                        gap = -relative_front - ego_length
                        if not signed:
                            gap = max(0.0, gap)
                        if gap <= self.visibility and (rear is None or gap < rear[0]):
                            rear = (gap, speed)
            neighbors.append((front, rear))
        return neighbors
