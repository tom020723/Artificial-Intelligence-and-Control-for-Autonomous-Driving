"""실제 SUMO로 신호 관측/정지선 통과/보상 면제를 검증한다.

실행: conda run -n sumo python -m unittest discover -s tests -v
"""
import os
import copy
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import traci

from env.road_builder import build
from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.sumo_env import SumoHighwayEnv
from train import REWARD
from utils.evaluator import evaluate_policy


class SignalExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sumo-signal-test-")
        # Retain regression coverage for saved experiments with upstream U-turns.
        legacy = copy.deepcopy(NOHYEONG)
        legacy["inbound_end_offsets"] = {}
        legacy["stop_offsets"] = {"A1": 8.0, "A3": 8.0, "A4": 8.0}
        legacy["upstream_uturns"] = {"A3": {"radius": 48.0, "from_lane": 4,
                                             "to_lane": 0, "signal_control": "central"}}
        legacy["inbound_lanes"]["A3"] = {0: 5, 1: 5, 2: 4}
        legacy["inbound_lane_links"]["A3"] = {
            2: [(0, 1), (1, 2), (2, 3), (3, 4)],
            1: [(i, i) for i in range(5)],
        }
        legacy["central_lane_movements"]["A3"][4] = [("A1", 2)]
        cls.legacy = legacy
        with patch("env.road_builder.NOHYEONG", legacy):
            cls.cfg = build(ROAD, EGO, TRAFFIC, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION,
                                  {**OBSERVATION, "signal_enabled": True, "signal_count": 1},
                                  ACTION, {**REWARD, "red_light_penalty": 10.0,
                                           "green_pass_bonus": 0.0,
                                           "red_stop_penalty_coef": 0.0,
                                           "lane_change_penalty": 0.0,
                                           "invalid_action_penalty": 0.0}, traffic=TRAFFIC)
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

    def test_experiment4_red_violation_penalty(self):
        self.env.rw["red_light_penalty"] = 5.0
        self.place(2, "r", speed=10)
        _, reward, terminated, _, info = self.env.step([1, 0])
        self.assertTrue(terminated and info["red_light_violation"])
        self.assertEqual(reward, -5.0)

    def test_green_bonus_once_per_signal_and_reset(self):
        self.env.rw["green_pass_bonus"] = 10.0
        self.place(2, "G", speed=10)
        _, reward, _, _, info = self.env.step([1, 0])
        self.assertEqual(info["green_pass_reward"], 10.0)
        self.assertGreaterEqual(reward, 10.0)
        self.place(2, "G", speed=10)
        _, _, _, _, info = self.env.step([1, 0])
        self.assertEqual(info["green_pass_reward"], 0.0)
        self.env.reset(seed=0)
        self.place(2, "G", speed=10)
        _, _, _, _, info = self.env.step([1, 0])
        self.assertEqual(info["green_pass_reward"], 10.0)

    def test_red_stop_room_penalty_and_queue_safety(self):
        self.env.rw.update(red_stop_penalty_coef=.05,
                           red_stop_safe_gap=3., red_stop_margin=3.,
                           red_stop_distance_scale=100.)
        signal = ("nC", 0, 20., "r")
        for leader, expected in [(None, .0085), (("front", 3.), 0.),
                                  (("front", 5.), .001)]:
            with self.subTest(leader=leader):
                with patch("env.sumo_env.traci.vehicle.getLeader", return_value=leader):
                    self.assertAlmostEqual(self.env._red_stop_penalty(signal), expected)
        for signal in [("nC", 0, 2., "r"), ("nC", 0, 20., "G"),
                       ("nC", 0, 101., "r"), None]:
            self.assertEqual(self.env._red_stop_penalty(signal), 0.)
        # 실험 5: 속도와 무관하게 동일한 거리 식 적용.
        with patch("env.sumo_env.traci.vehicle.getLeader", return_value=None):
            for speed in (0., .6, 5., 13.89):
                traci.vehicle.setPreviousSpeed("ego", speed)
                self.assertAlmostEqual(self.env._red_stop_penalty(("nC", 0, 20., "r")), .0085)
        self.place(20, "r", speed=0)
        _, reward, _, _, info = self.env.step([0, 0])
        self.assertAlmostEqual(info["red_stop_penalty"], .0085)
        self.assertAlmostEqual(reward, -.0085)

    def test_green_bonus_without_stop_offset(self):
        self.env.rw["green_pass_bonus"] = 10.0
        with patch.object(self.env, "_choose_ego_route", return_value="ego_d_A2_A1"):
            self.env.reset(seed=0)
        lane = next(lane for lane in traci.lane.getIDList()
                    if lane.startswith("A2_in_") and any(
                        traci.lane.getEdgeID(link[0]) == "A1_out"
                        for link in traci.lane.getLinks(lane)))
        traci.vehicle.moveTo("ego", lane, traci.lane.getLength(lane) - 2)
        traci.vehicle.setPreviousSpeed("ego", 10)
        signal = self.env._next_signal()
        self.assertAlmostEqual(signal[2], 2.)
        n = len(traci.trafficlight.getRedYellowGreenState(signal[0]))
        traci.trafficlight.setRedYellowGreenState(signal[0], "G" * n)
        _, _, _, _, info = self.env.step([1, 0])
        self.assertEqual(info["green_pass_reward"], 10.0)

    def test_lane_change_costs(self):
        self.env.rw.update(lane_change_penalty=.1, invalid_action_penalty=.2)
        self.place(20, "G", speed=0)
        self.assertTrue(self.lane.endswith("_0"))
        _, reward, _, _, info = self.env.step([0, -1])
        self.assertFalse(info["lane_change_applied"])
        self.assertAlmostEqual(reward, -.2)
        _, reward, _, _, info = self.env.step([0, 1])
        self.assertTrue(info["lane_change_applied"])
        self.assertAlmostEqual(reward, -.1)

    def use_two_signals(self, route):
        reward = self.env.rw.copy()
        self.env.close()
        self.env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION,
                                  {**OBSERVATION, "signal_enabled": True, "signal_count": 2},
                                  ACTION, reward, traffic=TRAFFIC)
        with patch.object(self.env, "_choose_ego_route", return_value=route):
            self.env.reset(seed=0)

    def test_A3_through_observes_only_central_signal_and_correct_stopline(self):
        self.use_two_signals("ego_d_A3_A2")
        lane = "A3_in1_1"
        traci.vehicle.moveTo("ego", lane, traci.lane.getLength(lane) - 20.)
        traci.vehicle.setPreviousSpeed("ego", 5.)
        for tid in traci.trafficlight.getIDList():
            n = len(traci.trafficlight.getRedYellowGreenState(tid))
            traci.trafficlight.setRedYellowGreenState(tid, ("r" if tid == "nC" else "G") * n)
        signals = self.env._upcoming_signals()
        self.assertEqual([s[0] for s in signals], ["nC"])
        raw_center = next(s for s in traci.vehicle.getNextTLS("ego") if s[0] == "nC")
        self.assertAlmostEqual(signals[0][2], raw_center[2] - 8.)
        stop = self.env._signal_stop_lines[(signals[0][0], signals[0][1])]
        self.assertTrue(stop["lane_id"].startswith("A3_in_"))
        self.assertAlmostEqual(stop["lane_position"], traci.lane.getLength(stop["lane_id"]) - 8.)
        obs = self.env._get_obs()
        self.assertEqual(obs.shape, (41,))
        np.testing.assert_array_equal(obs[31:35], [0, 0, 0, 1])
        np.testing.assert_array_equal(obs[36:41], [1, 0, 0, 0, 1])
        self.assertAlmostEqual(float(obs[35]), signals[0][2] / 100., places=5)
        self.assertEqual(self.env._nearest_red_signal(signals), signals[0])
        self.env.rw["red_stop_penalty_coef"] = .05
        _, _, _, _, info = self.env.step([0, 0])
        self.assertTrue(info["signal_red_visible"])
        self.assertGreater(info["red_stop_penalty"], 0.)
        self.assertEqual(len(info["observed_signals"]), 1)
        self.assertAlmostEqual(info["red_stop_penalty"], .05 * (info["red_signal_distance"] - 3.) / 100.)

    def test_A3_uturn_branch_keeps_left_turn_and_uses_central_phases(self):
        net = ET.parse(os.path.join(self.tmp.name, "highway.net.xml")).getroot()
        self.assertEqual([t.get("id") for t in net.findall("tlLogic")], ["nC"])
        connections = net.findall("connection")
        through = [c for c in connections if c.get("from") == "A3_in1" and c.get("to") == "A3_in"]
        self.assertEqual(len(through), 5)
        self.assertTrue(all(c.get("tl") is None for c in through))
        self.assertTrue(any(c.get("fromLane") == "4" and c.get("toLane") == "4" for c in through))
        self.assertTrue(any(c.get("from") == "A3_in" and c.get("fromLane") == "4"
                            and c.get("to") == "A1_out" for c in connections))
        self.assertFalse(any(c.get("from") == "A3_in" and c.get("to") == "A3_out" for c in connections))
        turn = next(c for c in connections if c.get("from") == "A3_in1" and c.get("to") == "A3_out1")
        self.assertEqual((turn.get("fromLane"), turn.get("toLane"), turn.get("tl")), ("4", "0", "nC"))
        idx = int(turn.get("linkIndex"))
        for phase in net.find("tlLogic[@id='nC']").findall("phase"):
            expected = "g" if phase.get("name") == "A3_green" else ("y" if phase.get("name") == "A3_yellow" else "r")
            self.assertEqual(phase.get("state")[idx], expected)

    def test_A3_through_branch_crossing_has_no_bonus_or_red_violation(self):
        self.use_two_signals("ego_d_A3_A4")
        self.env.rw["green_pass_bonus"] = 10.
        lane = "A3_in1_2"
        traci.vehicle.moveTo("ego", lane, traci.lane.getLength(lane) - 1.)
        traci.vehicle.setPreviousSpeed("ego", 5.)
        n = len(traci.trafficlight.getRedYellowGreenState("nC"))
        traci.trafficlight.setRedYellowGreenState("nC", "r" * n)
        _, _, terminated, _, info = self.env.step([0, 0])
        self.assertFalse(terminated)
        self.assertFalse(info["red_light_violation"])
        self.assertEqual(info["green_pass_reward"], 0.)
        self.assertEqual([s["tls_id"] for s in info["observed_signals"]], ["nC"])
        self.assertEqual(traci.vehicle.getRoadID("ego"), "A3_in")

    def test_A3_uturn_red_wait_and_crossing_use_upstream_stopline(self):
        self.use_two_signals("ego_d_A3_A3")
        lane = "A3_in1_4"
        length = traci.lane.getLength(lane)
        traci.vehicle.moveTo("ego", lane, length - 2.)
        traci.vehicle.setPreviousSpeed("ego", 0.)
        n = len(traci.trafficlight.getRedYellowGreenState("nC"))
        traci.trafficlight.setRedYellowGreenState("nC", "r" * n)
        signal = self.env._next_signal()
        self.assertEqual(signal[0], "nC")
        self.assertAlmostEqual(signal[2], 2.)
        self.assertEqual(self.env._signal_stop_lines[signal[:2]]["lane_id"], lane)
        _, _, terminated, _, info = self.env.step([0, 0])
        self.assertFalse(terminated)
        self.assertFalse(info["red_light_violation"])
        traci.vehicle.setPreviousSpeed("ego", 10.)
        _, reward, terminated, _, info = self.env.step([0, 0])
        self.assertTrue(terminated and info["red_light_violation"])
        self.assertEqual(reward, -10.)

    def test_legacy_config_keeps_separate_upstream_controller(self):
        legacy = copy.deepcopy(self.legacy)
        legacy["upstream_uturns"]["A3"].pop("signal_control")
        with tempfile.TemporaryDirectory(prefix="sumo-legacy-uturn-") as folder:
            with patch("env.road_builder.NOHYEONG", legacy):
                build(ROAD, EGO, TRAFFIC, folder)
            net = ET.parse(os.path.join(folder, "highway.net.xml")).getroot()
            self.assertEqual({t.get("id") for t in net.findall("tlLogic")}, {"nC", "nM_A3_0"})

    def test_uncontrolled_lane_end_is_not_a_red_stopline(self):
        self.use_two_signals("ego_d_A4_A2")
        lane = "A4_in1_0"
        traci.vehicle.moveTo("ego", lane, traci.lane.getLength(lane) - 1.)
        traci.vehicle.setPreviousSpeed("ego", 5.)
        n = len(traci.trafficlight.getRedYellowGreenState("nC"))
        traci.trafficlight.setRedYellowGreenState("nC", "r" * n)
        signal = self.env._next_signal()
        self.assertEqual(signal[0], "nC")
        self.assertGreater(signal[2], 5.)  # 1m to unrelated lane-end marker, actual light farther ahead
        self.assertNotEqual(self.env._signal_stop_lines[(signal[0], signal[1])]["lane_id"], lane)
        _, _, _, _, info = self.env.step([0, 0])
        self.assertFalse(info["red_light_violation"])

    def test_two_signal_padding_and_violation_of_second_signal(self):
        self.use_two_signals("ego_d_A1_A2")
        self.place(2, "r", speed=10.)
        obs = self.env._get_obs()
        np.testing.assert_array_equal(obs[36:41], [1, 0, 0, 0, 1])
        real_signal = self.env._next_signal()
        # Both stop lines crossed in one simulation step: red second signal must not be missed.
        self.env._signal_stop_lines[("test_green", 0)] = {"lane_id": self.lane, "lane_position": 0.}
        original = self.env._upcoming_signals
        calls = [0]
        def upcoming():
            calls[0] += 1
            if calls[0] == 1:
                return [("test_green", 0, .1, "G"), real_signal]
            return original()
        with patch.object(self.env, "_upcoming_signals", side_effect=upcoming):
            _, reward, terminated, _, info = self.env.step([1, 0])
        self.assertTrue(terminated and info["red_light_violation"])
        self.assertEqual(reward, -10.)
        self.assertEqual(info["green_pass_reward"], 0.)

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
