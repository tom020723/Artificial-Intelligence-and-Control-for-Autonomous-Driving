"""Helpers used by scripts/reproduce_pipeline.sh.

Run: python -m unittest tests/test_reproduce_pipeline.py
"""
import json
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET

from scripts.evaluate_fixed_routes import IDM_ONLY_SPEED_GAIN, set_background_speed_gain, write_benchmark_context
from scripts.ppo_checkpoints import BC_FILES, export, select

ROUTES = """<routes>
    <vTypeDistribution id="background">
        <vType id="car_krauss" lcSpeedGain="1.0"/>
        <vType id="car_idm" lcSpeedGain="1.0"/>
        <vType id="car_eidm" lcSpeedGain="1.0"/>
        <vType id="car_acc" lcSpeedGain="1.0"/>
    </vTypeDistribution>
</routes>
"""


class BackgroundSpeedGainTests(unittest.TestCase):
    def test_only_idm_keeps_speed_gain(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "highway.rou.xml"
            path.write_text(ROUTES)
            set_background_speed_gain(path, IDM_ONLY_SPEED_GAIN)
            values = {n.get("id"): n.get("lcSpeedGain") for n in ET.parse(path).iter("vType")}
        self.assertEqual(values, {"car_krauss": "0.0", "car_idm": "1.0", "car_eidm": "0.0", "car_acc": "0.0"})

    def test_rejects_unexpected_types(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "highway.rou.xml"
            path.write_text(ROUTES)
            with self.assertRaises(ValueError):
                set_background_speed_gain(path, {"car_idm": 1.0})


class BenchmarkContextTests(unittest.TestCase):
    def test_settings_come_from_repository_and_benchmark(self):
        with TemporaryDirectory() as tmp:
            context = Path(tmp) / "reference"
            curriculum = write_benchmark_context(context)
            self.assertEqual(sorted(p.name for p in context.iterdir()), ["mdp_config.py", "road_config.py", "train.py"])
            road = runpy.run_path(str(context / "road_config.py"))
            mdp = runpy.run_path(str(context / "mdp_config.py"))
        self.assertEqual(road["EGO"]["depart_lane"], "center")
        self.assertEqual(set(road["TRAFFIC"]["controllers"]), {"krauss", "idm", "eidm", "acc"})
        self.assertEqual(mdp["OBSERVATION"]["layout"], "bc49")
        self.assertEqual(mdp["OBSERVATION"]["guidance_lateral_scale"], 50.0)
        safety = mdp["ACTION"]["lane_change_safety"]
        self.assertFalse(safety["enabled"])          # no rule-based ego restriction
        self.assertEqual(safety["spawn_protect"], 0.0)
        self.assertTrue(curriculum.total_vph)
        self.assertEqual(curriculum.ring_reference_vph, 800)


class PpoCheckpointTests(unittest.TestCase):
    def make_run(self, root):
        bc = root / "BC/training"
        bc.mkdir(parents=True)
        for name in BC_FILES:
            (bc / name).write_text(f"bc {name}")
        run = root / "PPO"
        run.mkdir()
        (run / "run.json").write_text(json.dumps({"initial_model": str(bc / "model.pt")}))
        for i in (10, 20):
            (run / f"model_iter{i:03d}.pt").write_text(f"weights {i}")
        return run

    def test_export_copies_bc_config_and_checkpoint(self):
        with TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            export(run, [10, 20])
            folder = run / "eval_runs/iter020"
            self.assertEqual((folder / "model.pt").read_text(), "weights 20")
            self.assertEqual({p.name for p in folder.iterdir()}, set(BC_FILES) | {"model.pt"})
            with self.assertRaises(FileNotFoundError):
                export(run, [30])

    def test_select_picks_highest_success_and_later_on_ties(self):
        with TemporaryDirectory() as tmp:
            run = self.make_run(Path(tmp))
            export(run, [10, 20])
            for i, rate in ((10, 0.95), (20, 0.95)):
                out = run / f"eval_runs/iter{i:03d}/evaluation_validation"
                out.mkdir()
                (out / "evaluation_summary.json").write_text(
                    json.dumps({"overall": {"success_rate": rate, "episodes": 4}}))
            self.assertEqual(select(run, "evaluation_validation")["iteration"], 20)
            saved = json.loads((run / "selection.json").read_text())
            self.assertEqual(saved["best"]["iteration"], 20)
            self.assertEqual(len(saved["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
