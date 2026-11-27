"""PPO from scratch (no BC) and the optional lane-change penalty.

Run: python -m unittest tests/test_ppo_scratch.py
"""
from copy import deepcopy
import tempfile
import unittest

import numpy as np
import torch
import traci

from algorithms.anchored_ppo import AnchoredPPO
from env.bc_ppo_reward import STEP_REWARD
from env.mdp_config import ACTION, OBSERVATION, SIMULATION
from env.road_builder import build
from env.road_config import EGO, ROAD, TRAFFIC
from env.sumo_env import SumoHighwayEnv
from train import HPARAMS, REWARD


def make_agent(obs_dim=8, seed=0):
    rng = np.random.default_rng(seed)
    normalization = dict(mean=rng.normal(size=obs_dim).astype(np.float32),
                         std=rng.uniform(.5, 3, size=obs_dim).astype(np.float32))
    hparams = dict(HPARAMS, n_steps=1024, minibatch_size=1024, device="cpu", seed=seed)
    agent = AnchoredPPO(obs_dim=obs_dim, act_dim=2, normalization=normalization, anchor_coef=0.1, **hparams)
    return agent, normalization


class ScratchInitTests(unittest.TestCase):
    def test_raw_input_matches_standardized_input_of_the_random_network(self):
        agent, normalization = make_agent()
        before = deepcopy(agent.policy)
        agent.init_from_scratch(normalization, init_std=0.5, lane_keep_prob=0.9)
        raw = torch.as_tensor(np.random.default_rng(1).normal(size=(64, 8)) * 3, dtype=torch.float32)
        standardized = (raw - torch.as_tensor(normalization["mean"])) / torch.as_tensor(normalization["std"])
        with torch.no_grad():
            expected = before.trunk(standardized)
            actual = agent.policy.trunk(raw)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_starts_mostly_keeping_the_lane_with_the_given_std_and_no_anchor(self):
        agent, normalization = make_agent()
        agent.init_from_scratch(normalization, init_std=0.5, lane_keep_prob=0.9)
        obs = torch.as_tensor(np.random.default_rng(2).normal(size=(256, 8)), dtype=torch.float32)
        with torch.no_grad():
            _, _, logits, _ = agent.policy(obs)
        keep = torch.softmax(logits, -1)[:, 1]
        self.assertAlmostEqual(float(keep.mean()), 0.9, delta=0.02)
        self.assertAlmostEqual(agent.policy.current_std, 0.5, places=5)
        self.assertIsNone(agent.anchor)
        self.assertEqual(agent.anchor_coef, 0.0)
        accel_kl, lane_kl = agent._anchor_kl(obs, torch.zeros(256, 1), logits)
        self.assertEqual(float(accel_kl.abs().sum() + lane_kl.abs().sum()), 0.0)

    def test_rejects_invalid_keep_probability(self):
        agent, normalization = make_agent()
        for value in (0.0, 1.0):
            with self.assertRaises(ValueError):
                agent.init_from_scratch(normalization, init_std=0.5, lane_keep_prob=value)


class LaneChangePenaltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="sumo-lane-penalty-test-")
        cls.cfg = build(ROAD, EGO, TRAFFIC, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_env(self, penalty):
        reward = dict(REWARD, bc_ppo_reward=True, bc_lane_change_penalty=penalty)
        env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION, OBSERVATION, ACTION, reward, traffic=TRAFFIC)
        env.reset(seed=0)
        env.disable_ego_rule_barriers()
        self.addCleanup(env.close)
        return env

    def change_lane(self, env):
        """Command a lane change into a lane that exists; return (reward, info)."""
        lane = traci.vehicle.getLaneIndex(env.ego_id)
        lanes = traci.edge.getLaneNumber(traci.vehicle.getRoadID(env.ego_id))
        command = 1.0 if lane + 1 < lanes else -1.0
        _, reward, terminated, truncated, info = env.step(np.array([0.0, command], np.float32))
        self.assertFalse(terminated or truncated)
        return reward, info

    def test_executed_lane_change_pays_the_penalty_and_keeping_does_not(self):
        env = self.make_env(0.05)
        reward, info = self.change_lane(env)
        self.assertNotEqual(info["lane_change"], 0)
        self.assertTrue(info["lane_change_applied"])
        self.assertAlmostEqual(reward, STEP_REWARD - 0.05)
        _, reward, _, _, info = env.step(np.array([0.0, 0.0], np.float32))
        self.assertEqual(info["lane_change"], 0)
        self.assertAlmostEqual(reward, STEP_REWARD)

    def test_default_has_no_penalty(self):
        env = self.make_env(0.0)
        reward, info = self.change_lane(env)
        self.assertTrue(info["lane_change_applied"])
        self.assertAlmostEqual(reward, STEP_REWARD)


if __name__ == "__main__":
    unittest.main()
