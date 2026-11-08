"""주차장 폐쇄와 우회전 도로의 출구 제한을 실제 생성 도로/교통류로 검사한다."""
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from env.road_builder import build
from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG
from sumolib import checkBinary


class RoadRestrictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sumo-road-restrictions-")
        cls.folder = Path(cls.tmp.name)
        # Exercise positive demand too: closure must not rely only on vph=0.
        config = copy.deepcopy(NOHYEONG)
        config["all_signals_green"] = False
        for segment in config["ring_segments"]:
            segment["vph"] = 30
        config["loop_routes"] = [{"from": "A2", "to": "A1",
                                  "via": ["OUTER_N", "OUTER_W"], "vph": 30}]
        traffic = copy.deepcopy(TRAFFIC)
        traffic.update(background_enabled=True, vehs_per_hour=200)
        with patch("env.road_builder.NOHYEONG", config):
            cls.cfg = build(ROAD, EGO, traffic, cls.tmp.name)
        cls.net = ET.parse(cls.folder / "highway.net.xml").getroot()
        cls.connections = cls.net.findall("connection")
        cls.ego_routes = ET.parse(cls.folder / "highway.ego.rou.xml").getroot()
        cls.traffic = ET.parse(cls.folder / "highway.rou.xml").getroot()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parking_geometry_remains_without_permissions_or_connections(self):
        for name in ("OUTER_N", "OUTER_W"):
            with self.subTest(edge=name):
                edge = self.net.find(f"edge[@id='{name}']")
                self.assertIsNotNone(edge)
                self.assertTrue(edge.findall("lane"))
                self.assertTrue(all(lane.get("disallow") == "all" for lane in edge.findall("lane")))
                self.assertFalse(any(name in (c.get("from"), c.get("to")) for c in self.connections))

    def test_three_slip_roads_exit_only_to_the_outbound_arm(self):
        for source, target in (("RING_A1_A3", "A3_out1"),
                               ("RING_A3_A5", "A5_out1"),
                               ("RING_A5_A2", "A2_out2")):
            with self.subTest(edge=source):
                targets = {c.get("to") for c in self.connections if c.get("from") == source}
                self.assertEqual(targets, {target})

    def test_all_generated_routes_avoid_parking_and_forbidden_through_connections(self):
        connections = {(c.get("from"), c.get("to")) for c in self.connections}
        closed = {"OUTER_N", "OUTER_W"}
        for route in self.ego_routes.findall("route") + self.traffic.findall("route"):
            edges = route.get("edges").split()
            self.assertFalse(closed.intersection(edges), route.get("id"))
            self.assertTrue(all(pair in connections for pair in zip(edges, edges[1:])), route.get("id"))
        direct = [r for r in self.ego_routes.findall("route") if r.get("id").startswith("ego_d_")]
        self.assertEqual(len(direct), 22)
        for flow in self.traffic.findall("flow"):
            self.assertFalse(closed.intersection(flow.get("via", "").split()), flow.get("id"))
            self.assertNotIn("OUTER_", flow.get("id"))
        self.assertFalse(any(f.get("id").startswith("lr_") for f in self.traffic.findall("flow")))
        self.assertEqual(len([f for f in self.traffic.findall("flow") if f.get("id").startswith("lf_")]), 5)

    def test_A1_straight_lanes_reach_A2_and_A5_without_lane3_left_turn(self):
        for lane in range(4):
            links = [c for c in self.connections if c.get("from") == "A1_in"
                     and c.get("fromLane") == str(lane)]
            self.assertEqual({(c.get("to"), int(c.get("toLane"))) for c in links},
                             {("A2_out", lane), ("A5_out", min(lane, 2))})
        left = [c for c in self.connections if c.get("from") == "A1_in" and c.get("fromLane") == "4"]
        self.assertEqual({(c.get("to"), c.get("toLane")) for c in left},
                         {("A4_out", "1"), ("A1_out", "0")})
        green = self.net.find("tlLogic[@id='nC']/phase[@name='A1_green']").get("state")
        merging = [c for c in self.connections if c.get("from") == "A1_in"
                   and c.get("to") == "A5_out" and c.get("toLane") == "2"]
        self.assertEqual(sorted(green[int(c.get("linkIndex"))] for c in merging), ["G", "g"])

    def test_A2_A5_upstream_uturns_share_their_central_signal_phase(self):
        nodes = {e.get("id"): e.get("to") for e in self.net.findall("edge")
                 if e.get("function") != "internal"}
        program = self.net.find("tlLogic[@id='nC']")
        self.assertEqual([t.get("id") for t in self.net.findall("tlLogic")], ["nC"])
        for arm, source, lane, target in (("A2", "A2_in2", "3", "A2_out2"),
                                          ("A5", "A5_in1", "2", "A5_out1")):
            with self.subTest(arm=arm):
                turns = [c for c in self.connections if c.get("from") == source and c.get("to") == target]
                self.assertEqual(len(turns), 1)
                turn = turns[0]
                self.assertEqual((turn.get("fromLane"), turn.get("toLane"), turn.get("tl")), (lane, "0", "nC"))
                index = int(turn.get("linkIndex"))
                for phase in program.findall("phase"):
                    expected = "g" if phase.get("name") == arm + "_green" else ("y" if phase.get("name") == arm + "_yellow" else "r")
                    self.assertEqual(phase.get("state")[index], expected)
                for link in self.connections:
                    if nodes.get(link.get("from")) == nodes[source] and link is not turn:
                        self.assertIsNone(link.get("tl"), link.attrib)
                route = self.ego_routes.find(f"route[@id='ego_d_{arm}_{arm}']")
                self.assertEqual(route.get("edges").split(), [source, target])

    def test_background_traffic_runs_without_using_closed_parking_roads(self):
        trajectory = self.folder / "traffic.fcd.xml"
        trips = self.folder / "trips.xml"
        result = subprocess.run(
            [checkBinary("sumo"), "-c", self.cfg, "--end", "180", "--seed", "0",
             "--no-step-log", "true", "--time-to-teleport", "-1",
             "--fcd-output", str(trajectory), "--tripinfo-output", str(trips)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(ET.parse(trips).getroot().findall("tripinfo"))
        vehicles = ET.parse(trajectory).getroot().findall("timestep/vehicle")
        self.assertTrue(vehicles)
        self.assertFalse(any(v.get("lane", "").startswith(("OUTER_N_", "OUTER_W_")) for v in vehicles))


if __name__ == "__main__":
    unittest.main()
