"""실제 SUMO로 신호 관측/정지선 통과/보상 면제를 검증한다.

실행: conda run -n sumo python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import traci

from env.road_builder import build
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.sumo_env import SumoHighwayEnv
from train import REWARD
from utils.evaluator import evaluate_policy


class SignalExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sumo-signal-test-")
        cls.cfg = build(ROAD, EGO, TRAFFIC, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION,
                                  {**OBSERVATION, "signal_enabled": True},
                                  ACTION, REWARD, traffic=TRAFFIC)
        self.env.ego_route_random = False
        self.env.reset(seed=0)
        self.lane = next(lane for lane in traci.lane.getIDList()
                         if lane.startswith("A1_in_") and any(
                             traci.lane.getEdgeID(link[0]) == "A2_out"
                             for link in traci.lane.getLinks(lane)))
        self.length = traci.lane.getLength(self.lane)
        self.offset = 8.0

    def tearDown(self):
        self.env.close()

    def place(self, distance, color, speed=0):
        traci.vehicle.moveTo("ego", self.lane, self.length - self.offset - distance)
        traci.vehicle.setSpeed("ego", speed)
        # Set the actual speed too, since setSpeed is applied in the next step.
        traci.vehicle.setPreviousSpeed("ego", speed)
        tid = self.env._next_signal()[0]
        n = len(traci.trafficlight.getRedYellowGreenState(tid))
        traci.trafficlight.setRedYellowGreenState(tid, color * n)

    def test_observation_colors_range_and_offset(self):
        self.assertEqual(self.env.obs_dim, 36)
        np.testing.assert_array_equal(self.env._signal_observation(None), [1, 0, 0, 0, 1])
        np.testing.assert_array_equal(self.env._signal_observation(("x", 0, 101, "r")), [1, 0, 0, 0, 1])
        for index, color in enumerate(("G", "y", "r"), start=1):
            self.place(10, color)
            obs = self.env._get_obs()
            self.assertEqual(obs[31 + index], 1)
            self.assertAlmostEqual(float(obs[35]), 0.1, places=5)
            self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(set(traci.vehicle.getIDList()), {"ego"})
        self.assertFalse(ET.parse(os.path.join(self.tmp.name, "highway.rou.xml")).getroot().findall("flow"))

    def test_red_crossing_at_offset_stopline_terminates(self):
        self.place(2, "r", speed=10)
        _, reward, terminated, _, info = self.env.step([1, 0])
        self.assertTrue(info["red_light_violation"])
        self.assertTrue(terminated)
        self.assertEqual(reward, -10)
        # Must detect before the junction edge, at the 8m offset line.
        self.assertEqual(traci.vehicle.getRoadID("ego"), "A1_in")

    def test_green_and_yellow_crossing_not_red_violation(self):
        for color in ("G", "y"):
            self.env.reset(seed=0)
            self.place(2, color, speed=10)
            _, _, _, _, info = self.env.step([1, 0])
            self.assertFalse(info["red_light_violation"])

    def test_red_stop_and_reward_exemptions(self):
        self.place(10, "r", speed=0)
        with patch.object(self.env, "_neighbor", return_value=(2.0, 0.0)):
            _, reward, terminated, _, info = self.env.step([0, 0])
        self.assertFalse(terminated)
        self.assertFalse(info["red_light_violation"])
        self.assertEqual(reward, 0)  # both close-gap and blocked exemptions
        self.place(10, "G", speed=0)
        with patch.object(self.env, "_neighbor", return_value=(2.0, 0.0)):
            _, reward, _, _, _ = self.env.step([0, 0])
        self.assertAlmostEqual(reward, -0.15)
        self.place(15, "r", speed=2)
        with patch.object(self.env, "_neighbor", return_value=(2.0, 0.0)):
            _, reward, _, _, _ = self.env.step([0, 0])
        self.assertAlmostEqual(reward, 0.1 * 2 / ROAD["speed_limit"] - 0.1)

    def test_red_violation_evaluation_metric(self):
        class FakeEnv:
            visibility = 50
            def reset(self): return np.zeros(36), {}
            def step(self, action):
                return np.zeros(36), -10, True, False, {"red_light_violation": True}
        class Policy:
            def predict(self, *args, **kwargs): return [0, 0]
        metrics = evaluate_policy(Policy(), FakeEnv(), n_episodes=2)
        self.assertEqual(metrics.red_light_violation_rate, 1)
        self.assertEqual(metrics.success_rate, 0)
        self.assertEqual(metrics.timeout_rate, 0)


if __name__ == "__main__":
    unittest.main()
