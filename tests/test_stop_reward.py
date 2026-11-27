"""The red-light stop bonus is peaked at its target (Laplace shape)."""
from math import exp
import unittest

from env.bc_ppo_reward import STOP_AMPLITUDE, STOP_SIGMA_M, stop_reward


class StopRewardTests(unittest.TestCase):
    def test_full_bonus_only_at_the_target_and_symmetric(self):
        self.assertAlmostEqual(stop_reward(0.0), STOP_AMPLITUDE)
        self.assertAlmostEqual(stop_reward(0.5), stop_reward(-0.5))

    def test_decays_exponentially_with_the_absolute_error(self):
        for error in (0.25, 1.0, 3.0):
            self.assertAlmostEqual(stop_reward(error), STOP_AMPLITUDE * exp(-error / STOP_SIGMA_M))
        # Peaked: a quarter metre already costs over a fifth of the bonus.
        self.assertLess(stop_reward(0.25), 0.8 * STOP_AMPLITUDE)


if __name__ == "__main__":
    unittest.main()
