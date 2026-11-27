"""Connect lane-based admission to a replicated SUMO road."""
from collections import deque
from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from env.uniform_idm import UniformIDM
from env.lane_inflow import LaneInflowController
from env.parallel_traffic import TrafficSchedule


def adaptive_schedule_factory(config, directory, flows, lights, uniform_idm_vph=0):
    routes = {node.get("id")[3:]: [edge[3:] for edge in node.get("edges").split()]
              for node in ET.parse(Path(directory) / "parallel.rou.xml").iter("route")
              if node.get("id").startswith("r0_")}
    return lambda: AdaptiveTrafficSchedule(
        flows, Path(config).with_name("highway.net.xml"), routes, lights, uniform_idm_vph)


class AdaptiveTrafficSchedule(TrafficSchedule):
    def __init__(self, flows, net_path, routes, lights, uniform_idm_vph):
        super().__init__(flows)
        self.net_path, self.routes, self.lights = net_path, routes, lights
        self.controller = None
        self.uniform_idm_vph = uniform_idm_vph
        self.uniform_idm = None

    def reset(self, api, now, vph, rng, live_ids):
        super().reset(api, now, vph, rng, live_ids)
        self.uniform_idm = None
        self.controller = (LaneInflowController(
            self.net_path, self.flows, self.routes, vph - self.uniform_idm_vph, rng)
                           if vph else None)
        if self.controller is not None and self.uniform_idm_vph:
            idm_rng = np.random.default_rng(rng.integers(2**63))
            self.uniform_idm = UniformIDM(
                self.flows, self.routes, self.controller.lanes, self.uniform_idm_vph, idm_rng)
            self.uniform_idm.next_due = {
                route: now + delay for route, delay in self.uniform_idm.next_due.items()
            }
        if self.controller is not None:
            self.controller.last_update = now
            self.controller.next_due = [now + self.controller._headway(rate)
                                        for rate in self.controller.rates]

    def restore(self, api, prefix, vehicles):
        restored = super().restore(api, prefix, vehicles)
        self.controller.route_by_id.update(restored)

    def spawn_due(self, api, now, prefix, rng):
        if self.controller is None:
            return
        base = prefix("")
        vehicle = api.vehicle
        states = {}
        for vid in sorted(self.ids):
            lane = vehicle.getLaneID(vid)
            if not lane:
                continue
            lane = ":" + lane.removeprefix(":" + base) if lane.startswith(":") else lane.removeprefix(base)
            states[vid] = (lane, vehicle.getLanePosition(vid), vehicle.getSpeed(vid), vehicle.getLength(vid))
        signals = {light: api.trafficlight.getRedYellowGreenState(prefix(light)) for light in self.lights}
        controller = self.controller
        controller.observe(states, signals, api.simulation.getStartingTeleportIDList())
        controller.update_rates(now)
        controller.route_by_id = {vid: route for vid, route in controller.route_by_id.items() if vid in self.ids}
        controller.spawn_due(api, now, states, prefix=prefix)
        if self.uniform_idm is not None:
            self.uniform_idm.spawn(api, now, states, controller.route_by_id, prefix=prefix)
        self.ids.update(controller.route_by_id)

    def checkpoint(self, now):
        controller = self.controller
        meters = {lane: asdict(meter) | {"samples": list(meter.samples)}
                  for lane, meter in controller.meters.items()}
        return dict(meters=meters, rates=controller.rates, gate_weights=list(controller.gate_weights.items()),
                    since_update=now - controller.last_update)

    def restore_checkpoint(self, checkpoint, now):
        controller = self.controller
        for lane, state in checkpoint["meters"].items():
            state["samples"] = deque(state["samples"], maxlen=5)
            vars(controller.meters[lane]).update(state)
        controller.rates = checkpoint["rates"]
        controller.gate_weights = dict(checkpoint["gate_weights"])
        controller.last_update = now - checkpoint["since_update"]
        controller.next_due = [now + controller._headway(rate) for rate in controller.rates]
