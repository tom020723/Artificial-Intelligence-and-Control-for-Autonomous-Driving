"""실험 8의 경로 목적지와 넓어진 차선 연결성을 실제 SUMO에서 검증한다."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import traci

from env.road_builder import build
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.sumo_env import SumoHighwayEnv
from train import REWARD


class DestinationObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sumo-destination-test-")
        cls.cfg = build(ROAD, EGO, TRAFFIC, cls.tmp.name)
        cls.observation = {**OBSERVATION, "signal_enabled": True, "signal_count": 2,
                           "connectivity_half_width": 4, "destination_enabled": True,
                           "destination_arms": ("A1", "A2", "A3", "A4", "A5")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION, self.observation,
                                  ACTION, REWARD, traffic=TRAFFIC)
        self.env._choose_ego_route = lambda: "ego_d_A3_A5"
        self.env.reset(seed=0)

    def tearDown(self):
        self.env.close()

    def move_near_A3_stopline(self, lane_index):
        lane = f"A3_in_{lane_index}"
        traci.vehicle.moveTo("ego", lane, traci.lane.getLength(lane) - 20.)
        return self.env._get_obs()

    def test_A3_to_A5_target_three_lanes_right_is_observed(self):
        obs = self.move_near_A3_stopline(3)
        self.assertEqual(obs.shape, (50,))
        self.assertTrue(self.env.observation_space.contains(obs))
        # A5 is reached from lane 0, three lanes to the right of lane 3.
        self.assertEqual(obs[21 + self.env.connectivity_offsets.index(-3)], 1.)
        self.assertAlmostEqual(float(obs[21 + self.env.connectivity_offsets.index(0)]), .4)
        self.assertEqual(obs[21 + self.env.connectivity_offsets.index(4)], -1.)
        np.testing.assert_array_equal(obs[-5:], [0, 0, 0, 0, 1])

    def test_leftmost_lane_sees_target_four_lanes_right(self):
        obs = self.move_near_A3_stopline(4)
        self.assertEqual(obs[21 + self.env.connectivity_offsets.index(-4)], 1.)

    def test_each_destination_is_encoded_from_actual_route(self):
        for index, arm in enumerate(self.observation["destination_arms"]):
            self.env._choose_ego_route = lambda arm=arm: f"ego_d_A1_{arm}"
            obs, _ = self.env.reset(seed=0)
            expected = np.eye(5, dtype=np.float32)[index]
            np.testing.assert_array_equal(obs[-5:], expected)
            obs, _, _, _, _ = self.env.step(np.array([0., 0.], dtype=np.float32))
            np.testing.assert_array_equal(obs[-5:], expected)

    def test_A2_destinations_are_distinguishable_at_departure(self):
        observations = []
        for destination in ("A1", "A3"):
            self.env._choose_ego_route = lambda destination=destination: f"ego_d_A2_{destination}"
            obs, _ = self.env.reset(seed=0)
            observations.append(obs)
        np.testing.assert_array_equal(observations[0][:-5], observations[1][:-5])
        np.testing.assert_array_equal(observations[0][-5:], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(observations[1][-5:], [0, 0, 1, 0, 0])

    def test_destination_does_not_depend_on_route_id(self):
        traci.route.add("renamed_test_route", traci.vehicle.getRoute("ego"))
        traci.vehicle.setRouteID("ego", "renamed_test_route")
        np.testing.assert_array_equal(self.env._get_obs()[-5:], [0, 0, 0, 0, 1])

    def test_old_snapshot_keys_preserve_41_dimensional_observation(self):
        legacy = self.observation.copy()
        for key in ("connectivity_half_width", "destination_enabled", "destination_arms"):
            legacy.pop(key)
        old_env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION, legacy,
                                 ACTION, REWARD, traffic=TRAFFIC)
        old = old_env._get_obs()
        current = self.env._get_obs()
        self.assertEqual(old.shape, (41,))
        np.testing.assert_array_equal(old[:21], current[:21])
        np.testing.assert_array_equal(old[21:26], current[23:28])
        np.testing.assert_array_equal(old[26:31], current[30:35])
        np.testing.assert_array_equal(old[31:41], current[35:45])

    def test_terminal_observation_keeps_new_shape(self):
        with patch("traci.vehicle.getIDList", return_value=[]):
            np.testing.assert_array_equal(self.env._get_obs(), np.zeros(50, dtype=np.float32))

    def test_unknown_destination_is_reported(self):
        with patch("traci.vehicle.getRoute", return_value=("unknown_edge",)):
            with self.assertRaisesRegex(ValueError, "Unknown destination edge"):
                self.env._destination_observation()


if __name__ == "__main__":
    unittest.main()
