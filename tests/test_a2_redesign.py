"""A2 실험 8 명세의 차선 연결, 신호, 차선 변경 허용을 실제 SUMO로 검사한다."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import traci
from env import road_builder
from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.sumo_env import SumoHighwayEnv
from train import REWARD


class A2RedesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sumo-a2-redesign-test-")
        config = copy.deepcopy(NOHYEONG)
        config["all_signals_green"] = False
        with patch("env.road_builder.NOHYEONG", config):
            cls.cfg = road_builder.build(ROAD, EGO, TRAFFIC, cls.tmp.name)
        cls.net = ET.parse(Path(cls.tmp.name)/"highway.net.xml").getroot()
        cls.connections = cls.net.findall("connection")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def links(self, edge, lane):
        return {(c.get("to"), int(c.get("toLane"))) for c in self.connections
                if c.get("from") == edge and int(c.get("fromLane")) == lane}

    def test_topology_removes_short_edges_and_merges_A2_junction_into_nC(self):
        self.assertIsNone(self.net.find("edge[@id='A2_in2']"))
        self.assertIsNone(self.net.find("edge[@id='A2_out2']"))
        self.assertIsNone(self.net.find("junction[@id='nM_A2_0']"))
        self.assertIsNone(self.net.find("junction[@id='nM_A2_1']"))
        self.assertIsNotNone(self.net.find("junction[@id='nM_A2']"))
        for edge, count in [("A2_in1",4),("A2_in",5),("A2_out",3),("A2_out1",3)]:
            node=self.net.find(f"edge[@id='{edge}']")
            self.assertEqual(len(node.findall("lane")),count)
            self.assertGreater(min(float(x.get("length")) for x in node.findall("lane")),10.)
        self.assertEqual(self.net.find("edge[@id='A2_in']").get("to"),"nC")
        self.assertEqual(self.net.find("edge[@id='RING_A2_A4']").get("from"),"nC")
        center=self.net.find("junction[@id='nC']")
        self.assertEqual((float(center.get("x")),float(center.get("y"))),(0.,0.))
        polygon=[tuple(map(float,p.split(","))) for p in center.get("shape").split()]
        # The deleted branch location must really be inside the new junction.
        px,py=56.29,32.50
        inside=False
        for (ax,ay),(bx,by) in zip(polygon,polygon[1:]+polygon[:1]):
            if (ay>py)!=(by>py) and px < (bx-ax)*(py-ay)/(by-ay)+ax:
                inside=not inside
        self.assertTrue(inside)

    def test_upstream_lane_connections_and_uturn(self):
        expected={0:{("A2_in",0),("A2_in",1)},1:{("A2_in",2)},
                  2:{("A2_in",3)},3:{("A2_in",4),("A2_out1",0)}}
        for lane, targets in expected.items():
            self.assertEqual(self.links("A2_in1",lane),targets)
        self.assertEqual(self.links("RING_A5_A2",0),{("A2_out1",0)})

    def test_five_incoming_lanes_have_exact_final_connections(self):
        expected={0:{("RING_A2_A4",0)},1:{("A1_out",0)},2:{("A1_out",1)},
                  3:{("A1_out",2)},4:{("A3_out",3)}}
        for lane, targets in expected.items():
            self.assertEqual(self.links("A2_in",lane),targets)
        for lane in range(3):
            self.assertEqual(self.links("A2_out",lane),{("A2_out1",lane)})

    def test_three_lane_exit_receives_requested_approaches(self):
        self.assertEqual([self.links("A1_in",i) & {(f"A2_out",j) for j in range(3)}
                          for i in range(4)], [set(),{("A2_out",0)},{("A2_out",1)},{("A2_out",2)}])
        self.assertIn(("A3_out",0),self.links("A1_in",0))
        self.assertIn(("A2_out",0),self.links("A3_in",1))
        self.assertIn(("A2_out",2),self.links("A3_in",2))
        self.assertIn(("A2_out",2),self.links("A4_in",2))

    def test_ring_exit_remains_uncontrolled_and_main_lanes_use_A2_signal(self):
        ring=next(c for c in self.connections if c.get("from")=="A2_in" and c.get("to")=="RING_A2_A4")
        self.assertIsNone(ring.get("tl"))
        phase=self.net.find("tlLogic[@id='nC']/phase[@name='A2_green']").get("state")
        for c in self.connections:
            if c.get("from")=="A2_in" and c is not ring:
                self.assertEqual(c.get("tl"),"nC")
                self.assertEqual(phase[int(c.get("linkIndex"))],"G")

    def test_dashed_lanes_allow_passenger_changes_in_network_and_RL_commands(self):
        env=SumoHighwayEnv(self.cfg,ROAD,EGO,SIMULATION,OBSERVATION,ACTION,REWARD,traffic=TRAFFIC)
        env._choose_ego_route=lambda:"ego_d_A2_A1"
        try:
            env.reset(seed=0)
            for edge,lane in [("A2_in",2),("A2_out",1)]:
                for direction in (-1,1):
                    self.assertIn("passenger",traci.lane.getChangePermissions(f"{edge}_{lane}",direction))
                node=self.net.find(f"edge[@id='{edge}']/lane[@index='{lane}']")
                self.assertIsNone(node.get("changeLeft"))
                self.assertIsNone(node.get("changeRight"))
            traci.vehicle.moveTo("ego","A2_in_2",5.)
            self.assertTrue(env._apply_lane_change(-1))
            self.assertTrue(env._apply_lane_change(1))
            self.assertTrue(env._apply_lane_change(0))
            traci.vehicle.moveTo("ego","A2_in1_2",100.)
            self.assertTrue(env._apply_lane_change(-1))
            env._choose_ego_route=lambda:"ego_d_A1_A2"
            env.reset(seed=0)
            traci.vehicle.moveTo("ego","A2_out_1",5.)
            self.assertTrue(env._apply_lane_change(-1))
            self.assertTrue(env._apply_lane_change(1))
        finally:
            env.close()


if __name__=="__main__":
    unittest.main()
