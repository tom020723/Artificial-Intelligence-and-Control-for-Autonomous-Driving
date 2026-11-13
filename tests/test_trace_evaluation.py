"""Regression tests for the trace CSV reward assembly."""

import unittest

from scripts.trace_evaluation import _reward_component_values


class TraceRewardTests(unittest.TestCase):
    def test_component_values_excludes_total_reward_key(self):
        info = {"reward": None, "red_stop_penalty": 0.0,
                "red_stop_clearance_reward": 0.02, "green_stop_penalty": 0.0}
        values = _reward_component_values(info)
        self.assertNotIn("reward", values)
        self.assertEqual(values["red_stop_clearance_reward"], 0.02)

    def test_missing_components_default_to_none(self):
        values = _reward_component_values({})
        self.assertNotIn("reward", values)
        self.assertIsNone(values["red_stop_penalty"])

    def test_component_keys_end_in_penalty_or_reward(self):
        values = _reward_component_values({})
        for key in values:
            self.assertTrue(key.endswith("penalty") or key.endswith("reward"))
            self.assertNotEqual(key, "reward")


if __name__ == "__main__":
    unittest.main()
