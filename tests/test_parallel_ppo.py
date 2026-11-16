"""Regression coverage for GAE across simultaneous environment transitions."""

import unittest

import numpy as np
import torch

from algorithms.parallel_ppo import ParallelPPO
from utils.buffer import RolloutBuffer
from utils.evaluator import evaluate_parallel_policy


class ScriptedWorkers:
    num_envs = 2

    def __init__(self):
        self.rounds = iter([
            [(100, 2, False, True, 5), (20, 3, False, False, None)],
            [(200, 4, True, False, None), (30, 6, False, False, None)],
        ])

    def step(self, actions):
        return [
            (np.array([obs]), reward, terminated, truncated,
             None if timeout_obs is None else np.array([timeout_obs]))
            for obs, reward, terminated, truncated, timeout_obs in next(self.rounds)
        ]


class ParallelPPOTests(unittest.TestCase):
    def test_gae_uses_own_worker_and_bootstraps_timeout_before_reset(self):
        agent = ParallelPPO.__new__(ParallelPPO)
        agent.n_steps, agent.num_timesteps = 4, 0
        agent.gamma, agent.gae_lambda = 0.9, 0.8
        agent.buffer = RolloutBuffer()
        agent._values = lambda obs: obs[:, 0]
        agent._act_batch = lambda obs: (np.zeros((len(obs), 2)), np.zeros(len(obs)), agent._values(obs))
        ep_returns, ep_lengths = [0.0, 0.0], [0, 0]

        obs, adv, returns, finished = agent._collect(
            ScriptedWorkers(), np.array([[1], [10]]), ep_returns, ep_lengths, logger=None)

        np.testing.assert_array_equal(obs[:, 0], [200, 30])
        np.testing.assert_array_equal(np.asarray(agent.buffer.obs)[:, 0], [1, 100, 10, 20])
        np.testing.assert_allclose(adv, [5.5, -96, 20.36, 13])
        np.testing.assert_allclose(returns, [6.5, 4, 30.36, 33])
        self.assertEqual(finished, [2, 4])
        self.assertEqual(ep_returns, [0, 9])
        self.assertEqual(ep_lengths, [0, 2])
        self.assertEqual(agent.num_timesteps, 4)

    def test_parallel_evaluation_counts_one_episode_per_environment(self):
        class Policy(torch.nn.Module):
            def det_action(self, obs):
                return obs

        class EvaluationEnv:
            num_envs = 2
            prototype = type("Prototype", (), {"visibility": 50.0})()

            def __init__(self):
                self.batches = []
                self.closed = False

            def reset(self):
                return np.array([[0.0], [1.0]], dtype=np.float32)

            def step_evaluation(self, indices, actions):
                self.batches.append(indices)
                np.testing.assert_array_equal(actions[:, 0], indices)
                if len(self.batches) == 1:
                    return [
                        (np.array([0.0]), 4.0, True, False, {"arrived": True}),
                        (np.array([1.0]), 1.0, False, False, {"speed": 8.0}),
                    ]
                return [(np.array([1.0]), 2.0, False, True, {"speed": 10.0})]

            def close(self):
                self.closed = True

        env = EvaluationEnv()
        metrics = evaluate_parallel_policy(Policy(), env, "cpu")
        self.assertEqual(env.batches, [[0, 1], [1]])
        self.assertTrue(env.closed)
        self.assertEqual(metrics.success_rate, 0.5)
        self.assertEqual(metrics.timeout_rate, 0.5)
        self.assertEqual(metrics.ep_return, 3.5)
        self.assertEqual(metrics.ep_length, 1.5)
        self.assertEqual(metrics.mean_speed, 9.0)


if __name__ == "__main__":
    unittest.main()
