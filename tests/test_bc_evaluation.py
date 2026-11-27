"""Evaluation context, summaries, and BC training plots.

Run: python -m unittest tests/test_bc_evaluation.py
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.evaluate_fixed_routes import plot_bc_training, plt
from utils.bc_policy import evaluation_context
from utils.evaluation_summary import summarize


def make_run(root, expert_timing=None):
    run = root / "run"; run.mkdir()
    data = root / "data"; data.mkdir()
    (run / "model.pt").write_bytes(b"fixture")
    (run / "road_config.py").write_text(
        "ROUTE_ONLY_EXPERIMENT=True\nSIGNAL_EXPERIMENT=False\nTRAFFIC={}\nEGO={'depart_lane':'random'}\n"
        "NOHYEONG={'ring_segments':[{'vph':0}], 'loop_routes':[]}\n_RING_BACKGROUND_VPH=(80,)\n")
    (run / "mdp_config.py").write_text(
        "OBSERVATION={}\nSIMULATION={}\nACTION={'lane_change_safety':{'enabled':True,'spawn_protect':60}}\n")
    (run / "train.py").write_text("HPARAMS={}\n")
    (run / "config.json").write_text(json.dumps(dict(
        algorithm="supervised_bc_on_ppo_hybrid_actor", data=str(data / "dataset.npz"),
        hidden_sizes=[256, 256], activation="relu")))
    collection = dict(observation=dict(layout="bc49"), simulation=dict(step_length=.5),
                      traffic=dict(vehs_per_hour=800, background_enabled=True, lc_speed_gain=1))
    if expert_timing is not None:
        collection["expert_control_timing"] = expert_timing
    (data / "config.json").write_text(json.dumps(collection))
    return run


def run_config(path):
    namespace = {}
    exec(path.read_text(), namespace)
    return namespace


class EvaluationContextTests(unittest.TestCase):
    def test_restores_collection_traffic_without_editing_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = make_run(root)
            saved, curriculum, _ = evaluation_context(run, root / "context")
            road = run_config(root / "context/road_config.py")
            self.assertFalse(road["ROUTE_ONLY_EXPERIMENT"])
            self.assertTrue(road["SIGNAL_EXPERIMENT"])
            self.assertTrue(road["TRAFFIC"]["background_enabled"])
            self.assertFalse(road["NOHYEONG"]["all_signals_green"])
            self.assertEqual(road["NOHYEONG"]["ring_segments"][0]["vph"], 80)
            self.assertEqual(road["EGO"]["depart_lane"], "center")
            self.assertIn("ROUTE_ONLY_EXPERIMENT=True", (run / "road_config.py").read_text())
            self.assertEqual(saved, json.loads((root / "context/evaluation_config.json").read_text()))
            self.assertTrue(saved["evaluation_only"])
            self.assertEqual(saved["fixed_departure"], "center")
            self.assertTrue(curriculum.total_vph)
            self.assertEqual(curriculum.ring_reference_vph, 800)

    def test_ego_has_no_rule_based_lane_change_restriction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = make_run(root, expert_timing={"spawn_protect": 30})
            saved, _, _ = evaluation_context(run, root / "context")
            safety = run_config(root / "context/mdp_config.py")["ACTION"]["lane_change_safety"]
            self.assertFalse(safety["enabled"])
            self.assertFalse(safety["spawn_protect_only"])
            self.assertEqual(safety["spawn_protect"], 0)
            self.assertIn("'spawn_protect':60", (run / "mdp_config.py").read_text())
            # The expert's entry restriction is recorded but never applied to the ego.
            self.assertEqual(saved["bc_spawn_protect_m"], 0)
            self.assertEqual(saved["expert_spawn_protect_m"], 30)


class PortableContextTests(unittest.TestCase):
    def test_uses_the_collection_config_beside_the_model_when_the_dataset_is_absent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = make_run(root)
            (run / "collection_config.json").write_text((root / "data" / "config.json").read_text())
            (root / "data" / "config.json").unlink()
            saved, curriculum, _ = evaluation_context(run, root / "context")
            road = run_config(root / "context/road_config.py")
            self.assertTrue(road["TRAFFIC"]["background_enabled"])
            self.assertEqual(curriculum.ring_reference_vph, 800)


class SummaryTests(unittest.TestCase):
    def test_clearance_stats_keep_references_apart_and_keep_zero(self):
        rows = [dict(red_wait_clearance_m=value, red_wait_reference=ref, first_signal_stop_clearance_m=None,
                     green_restart_observed=False, green_restart_delay_s=None)
                for value, ref in [(0, "stop_line"), (2, "stop_line"), (10, "queued_vehicle"),
                                   (20, "queued_vehicle"), (None, "queued_vehicle")]]
        with patch("utils.evaluation_summary.episode_rates", return_value={}):
            result = summarize(rows)
            empty = summarize(rows[:2])["red_wait_clearance_by_reference"]["queued_vehicle"]
        split = result["red_wait_clearance_by_reference"]
        self.assertEqual(result["red_wait_clearance"]["n"], 4)
        self.assertEqual(split["stop_line"]["n"], 2)
        self.assertEqual(split["stop_line"]["median_m"], 1)
        self.assertEqual(split["queued_vehicle"]["n"], 2)
        self.assertEqual(split["queued_vehicle"]["median_m"], 15)
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["median_m"])


class TrainingPlotTests(unittest.TestCase):
    def test_panels_use_epoch_and_bc_losses(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "training_log.csv").write_text(
                "epoch,train_loss,val_loss,val_accel_mse,val_lane_ce,val_lane_macro_f1,val_left_precision,"
                "val_right_precision,val_left_recall,val_right_recall\n"
                "1,.8,.9,.1,.8,.2,.3,.4,.5,.6\n2,.4,.5,.1,.4,.4,.5,.6,.7,.8\n")
            (run / "result.json").write_text('{"best_epoch":2}')
            fig = plt.figure()
            try:
                plot_bc_training(fig, fig.add_gridspec(4, 2), run)
                self.assertEqual(len(fig.axes), 6)
                self.assertTrue(all(ax.get_xlabel() == "Epoch" for ax in fig.axes))
                self.assertEqual(list(fig.axes[0].lines[0].get_xdata()), [1, 2])
                self.assertEqual(list(fig.axes[0].lines[0].get_ydata()), [.8, .4])
            finally:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
