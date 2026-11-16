"""Regression tests for the Phase 1 retraining validation tooling."""

import unittest

import numpy as np

from scripts.evaluate_quick import (
    EVAL_SEED, OFFSET_SETS, STEP, _departure_seed, _check_integrity,
)
from scripts.trace_evaluation import _reward_component_values, _signal_state, _outcome


class OffsetSetTests(unittest.TestCase):
    def test_sets_have_50_unique_ticks_each(self):
        for name, ticks in OFFSET_SETS.items():
            self.assertEqual(len(ticks), 50, name)
            self.assertEqual(len(set(ticks)), 50, name)
            # 0.5 s steps; both sets stay inside the 250-tick (125 s) cycle.
            self.assertTrue(all(0 <= tick < 250 for tick in ticks), name)

    def test_development_and_holdout_are_disjoint(self):
        self.assertFalse(set(OFFSET_SETS["development"]) & set(OFFSET_SETS["holdout"]))

    def test_sets_match_spec(self):
        self.assertEqual(OFFSET_SETS["development"], tuple(range(2, 250, 5)))
        self.assertEqual(OFFSET_SETS["holdout"], tuple(range(4, 250, 5)))

    def test_departure_seed_depends_on_offset_order_not_tick_value(self):
        # Same route and offset order -> same seed across the two sets.
        dev = _departure_seed(0, OFFSET_SETS["development"][7], "development")
        hold = _departure_seed(0, OFFSET_SETS["holdout"][7], "holdout")
        self.assertEqual(dev.entropy, hold.entropy)
        self.assertEqual(dev.spawn_key, hold.spawn_key)
        # Different route index or offset order -> different seed.
        self.assertNotEqual(dev.entropy,
                            _departure_seed(1, OFFSET_SETS["development"][7], "development").entropy)
        self.assertNotEqual(dev.entropy,
                            _departure_seed(0, OFFSET_SETS["development"][8], "development").entropy)

    def test_departure_seed_is_independent_of_training_seed(self):
        self.assertEqual(EVAL_SEED, 0)


class IntegrityTests(unittest.TestCase):
    def test_valid_rows_pass(self):
        routes = ["ego_d_A1_A1", "ego_d_A1_A2"]
        ticks = tuple(range(2, 6, 5))  # two offsets
        rows = [{"route": r, "signal_offset_s": t * STEP}
                for r in routes for t in ticks]
        _check_integrity(rows, routes, ticks)  # must not raise

    def test_missing_row_raises(self):
        routes = ["ego_d_A1_A1", "ego_d_A1_A2"]
        ticks = tuple(range(2, 6, 5))
        rows = [{"route": "ego_d_A1_A1", "signal_offset_s": 1.0}]
        with self.assertRaises(RuntimeError):
            _check_integrity(rows, routes, ticks)

    def test_duplicate_row_raises(self):
        routes = ["ego_d_A1_A1"]
        ticks = tuple(range(2, 6, 5))
        rows = [{"route": "ego_d_A1_A1", "signal_offset_s": 1.0}] * 2
        with self.assertRaises(RuntimeError):
            _check_integrity(rows, routes, ticks)


class TraceTests(unittest.TestCase):
    def test_reward_components_exclude_total(self):
        info = {"reward": 123.0, "green_pass_reward": 20.0, "red_stop_penalty": 0.05,
                "not_a_reward": 1.0}
        values = _reward_component_values(info)
        self.assertNotIn("reward", values)
        self.assertNotIn("not_a_reward", values)
        self.assertEqual(values["green_pass_reward"], 20.0)
        self.assertEqual(values["red_stop_penalty"], 0.05)

    def test_signal_state_returns_state_and_distance(self):
        info = {"observed_signals": [{"state": "r", "distance": 5.0}]}
        self.assertEqual(_signal_state(info), ("r", 5.0))
        self.assertEqual(_signal_state({"observed_signals": []}), ("", None))

    def test_outcome_priority_and_timeout(self):
        self.assertEqual(_outcome({"arrived": True}, False, False), "arrived")
        self.assertEqual(_outcome({"collided": True}, True, False), "collided")
        self.assertEqual(_outcome({"red_light_violation": True}, True, False),
                         "red_light_violation")
        self.assertEqual(_outcome({"wrong_lane": True}, True, False), "wrong_lane")
        self.assertEqual(_outcome({}, False, True), "timeout")
        self.assertEqual(_outcome({}, False, False), "")


if __name__ == "__main__":
    unittest.main()
