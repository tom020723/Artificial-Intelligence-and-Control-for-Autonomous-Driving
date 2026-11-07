"""PPO rollout 경계의 bootstrap 및 GAE 에피소드 분리 회귀 테스트."""
import unittest
import numpy as np
from algorithms.ppo import PPO
from utils.buffer import RolloutBuffer


class ScriptedEnv:
    def __init__(self, transitions, reset_value):
        self.transitions = iter(transitions)
        self.reset_value = reset_value
        self.reset_count = 0

    def step(self, action):
        value, reward, terminated, truncated = next(self.transitions)
        return np.array([value], dtype=np.float32), reward, terminated, truncated, {}

    def reset(self):
        self.reset_count += 1
        return np.array([self.reset_value], dtype=np.float32), {}


class PPOTimeLimitTests(unittest.TestCase):
    def collect(self, transitions, reset_value=100., gae_lambda=.95):
        # Use the production collector with exact, known critic values.
        agent = PPO.__new__(PPO)
        agent.n_steps, agent.num_timesteps, agent.gamma = len(transitions), 0, .99
        agent.buffer = RolloutBuffer()
        agent.get_value = lambda obs: float(obs[0])
        agent.act = lambda obs: (np.zeros(2, dtype=np.float32), 0., agent.get_value(obs))
        env = ScriptedEnv(transitions, reset_value)
        obs, last_value, returns = agent.collect_rollout(
            env, np.array([1.], dtype=np.float32), ep_stats=[0., 0])
        advantages, _ = agent.buffer.compute_gae(last_value, .99, gae_lambda)
        return agent.buffer, advantages, returns, env

    def test_timeout_at_rollout_end_bootstraps_final_state_once(self):
        buffer, adv, returns, env = self.collect([(5., 2., False, True)])
        self.assertAlmostEqual(buffer.rewards[0], 2 + .99 * 5)
        self.assertAlmostEqual(float(adv[0]), 2 + .99 * 5 - 1, places=5)
        self.assertEqual(returns, [2.])  # episode logging uses unmodified reward
        self.assertEqual(env.reset_count, 1)

    def test_timeout_does_not_leak_next_episode_reward_or_value(self):
        for reward in (0., 10., -20.):
            for reset_value in (0., 100.):
                with self.subTest(reward=reward, reset_value=reset_value):
                    _, adv, _, _ = self.collect([
                        (5., 2., False, True), (0., reward, True, False)], reset_value)
                    self.assertAlmostEqual(float(adv[0]), 5.95, places=5)

    def test_timeout_with_lambda_zero_does_not_bootstrap_reset_state(self):
        _, adv, _, _ = self.collect([
            (5., 2., False, True), (0., 10., True, False)], gae_lambda=0.)
        self.assertAlmostEqual(float(adv[0]), 5.95, places=5)

    def test_true_terminal_does_not_bootstrap(self):
        buffer, adv, _, _ = self.collect([(5., 2., True, False)])
        self.assertEqual(buffer.rewards, [2.])
        self.assertAlmostEqual(float(adv[0]), 1.)

    def test_terminated_and_truncated_together_do_not_bootstrap(self):
        buffer, adv, _, _ = self.collect([(5., 2., True, True)])
        self.assertEqual(buffer.rewards, [2.])
        self.assertAlmostEqual(float(adv[0]), 1.)

    def test_ongoing_rollout_bootstraps_without_reset(self):
        _, adv, _, env = self.collect([(5., 2., False, False)])
        self.assertAlmostEqual(float(adv[0]), 5.95, places=5)
        self.assertEqual(env.reset_count, 0)

    def test_gae_still_propagates_within_episode(self):
        _, adv, _, _ = self.collect([
            (5., 2., False, False), (7., 3., False, False)])
        delta_last = 3 + .99 * 7 - 5
        self.assertAlmostEqual(float(adv[1]), delta_last, places=5)
        self.assertAlmostEqual(float(adv[0]), 2 + .99 * 5 - 1 + .99 * .95 * delta_last, places=5)


if __name__ == '__main__':
    unittest.main()
