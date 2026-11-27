"""Event-reward speeding penalty and AnchoredPPO outcome bookkeeping.

Run: python -m unittest tests/test_bc_ppo_reward.py
"""
import tempfile
import unittest

import numpy as np
import traci

from algorithms.anchored_ppo import OUTCOMES, TerminalRewardRecorder, outcome
from env.bc_ppo_reward import OVERSPEED_TOLERANCE, STEP_REWARD, TERMINAL_REWARDS, overspeed_penalty
from env.mdp_config import ACTION, OBSERVATION, SIMULATION
from env.road_builder import build
from env.road_config import EGO, ROAD, TRAFFIC
from env.sumo_env import SumoHighwayEnv
from train import REWARD


class OverspeedPenaltyTests(unittest.TestCase):
    def test_zero_at_or_below_the_limit(self):
        for speed in (0.0, 8.0, 13.89):
            self.assertEqual(overspeed_penalty(speed, 13.89, 0.02, tolerance=0.0), 0.0)

    def test_linear_in_relative_excess(self):
        self.assertAlmostEqual(overspeed_penalty(1.2 * 13.89, 13.89, 0.02, tolerance=0.0), 0.02 * 0.2)
        self.assertAlmostEqual(overspeed_penalty(1.5 * 8.33, 8.33, 0.02, tolerance=0.0), 0.02 * 0.5)
        # Same relative excess on a slower lane costs the same.
        self.assertAlmostEqual(overspeed_penalty(1.3 * 3.0, 3.0, 0.01),
                               overspeed_penalty(1.3 * 13.89, 13.89, 0.01))

    def test_default_tolerance_matches_the_expert_speed_spread(self):
        self.assertEqual(OVERSPEED_TOLERANCE, 0.1)
        self.assertEqual(overspeed_penalty(1.1 * 13.89, 13.89, 0.02), 0.0)
        self.assertAlmostEqual(overspeed_penalty(1.3 * 13.89, 13.89, 0.02), 0.02 * 0.2)

    def test_tolerance_shifts_the_threshold(self):
        self.assertEqual(overspeed_penalty(1.1 * 13.89, 13.89, 0.02, tolerance=0.1), 0.0)
        self.assertAlmostEqual(overspeed_penalty(1.3 * 13.89, 13.89, 0.02, tolerance=0.1), 0.02 * 0.2)

    def test_zero_coefficient_disables_it(self):
        self.assertEqual(overspeed_penalty(40.0, 13.89, 0.0), 0.0)

    def test_rejects_invalid_arguments(self):
        for args in ((10, 13.89, -0.1), (10, 13.89, 0.1, -0.1), (10, 0.0, 0.1), (10, 13.89, float("nan"))):
            with self.assertRaises(ValueError):
                overspeed_penalty(*args)


class OutcomeTests(unittest.TestCase):
    def test_terminal_step_rewards_map_to_events(self):
        for name, value in OUTCOMES.items():
            self.assertEqual(outcome(value + STEP_REWARD), name)
            # One step of speeding penalty does not change the event.
            self.assertEqual(outcome(value + STEP_REWARD - 0.05), name)
        self.assertEqual(outcome(TERMINAL_REWARDS["red_light"]), "collision_or_red_light")
        self.assertEqual(outcome(TERMINAL_REWARDS["timeout"]), "wrong_lane_or_timeout")
        self.assertEqual(outcome(0.3), "other")

    def test_recorder_keeps_terminal_rewards_in_transition_order(self):
        class FakeEnv:
            num_envs = 3

            def __init__(self):
                self.batches = iter([
                    [(None, -0.001, False, False, None), (None, 0.9999, True, False, None),
                     (None, -1.9, True, False, None)],
                    [(None, -1.0001, False, True, None), (None, 0.0, False, False, None),
                     (None, 0.0, False, False, None)],
                ])

            def step(self, actions):
                return next(self.batches)

        env = TerminalRewardRecorder(FakeEnv())
        self.assertEqual(env.num_envs, 3)
        env.step(None)
        env.step(None)
        self.assertEqual([outcome(r) for r in env.drain()],
                         ["arrival", "collision_or_red_light", "wrong_lane_or_timeout"])
        self.assertEqual(env.drain(), [])


class EnvOverspeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sumo-overspeed-test-")
        cls.cfg = build(ROAD, EGO, TRAFFIC, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_env(self, coef, tolerance=0.0):
        reward = dict(REWARD, bc_ppo_reward=True, overspeed_penalty=coef, overspeed_tolerance=tolerance)
        env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION, OBSERVATION, ACTION, reward, traffic=TRAFFIC)
        env.reset(seed=0)
        self.addCleanup(env.close)
        return env

    def drive_one_step(self, env, speed):
        traci.vehicle.setSpeed(env.ego_id, speed)
        traci.vehicle.setPreviousSpeed(env.ego_id, speed)
        _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0], np.float32))
        self.assertFalse(terminated or truncated)
        return reward, info

    def test_speeding_step_pays_the_lane_limit_penalty(self):
        env = self.make_env(0.02)
        self.assertFalse(env.ego_speed_cap_enabled)
        reward, info = self.drive_one_step(env, 20.0)
        speed = traci.vehicle.getSpeed(env.ego_id)
        limit = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(env.ego_id))
        self.assertGreater(speed, limit)  # no cap: the commanded speed is kept
        self.assertAlmostEqual(info["lane_speed_limit"], limit)
        expected = 0.02 * (speed - limit) / limit
        self.assertAlmostEqual(info["overspeed_penalty"], expected)
        self.assertAlmostEqual(reward, STEP_REWARD - expected)

    def test_legal_speed_and_disabled_penalty_cost_nothing(self):
        env = self.make_env(0.02)
        reward, info = self.drive_one_step(env, 8.0)
        self.assertEqual(info["overspeed_penalty"], 0.0)
        self.assertAlmostEqual(reward, STEP_REWARD)
        env.close()
        env = self.make_env(0.0)
        reward, info = self.drive_one_step(env, 20.0)
        self.assertEqual(info["overspeed_penalty"], 0.0)
        self.assertIsNone(info["lane_speed_limit"])
        self.assertAlmostEqual(reward, STEP_REWARD)

    def test_tolerance_is_applied_in_the_env(self):
        env = self.make_env(0.02, tolerance=1.0)
        _, info = self.drive_one_step(env, 20.0)
        self.assertEqual(info["overspeed_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
