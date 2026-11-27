"""Isolated (fresh SUMO per case) evaluation gives order- and batch-independent outcomes.

Run: python -m unittest tests/test_isolated_evaluation.py
The end-to-end tests use the tracked final outputs in results/final.
"""
import csv
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from env import mdp_config, road_builder, road_config
from scripts.evaluate_fixed_routes import case_sumo_seed, evaluate_worker, read_json, run
from utils.bc_policy import evaluation_context

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "results/final/suite_diverse_test"
MODEL = ROOT / "results/final/PPO/eval_runs/iter080/model.pt"
# Cases whose outcome changed with case order in the previous shared-SUMO mode.
CASES = (632, 3253, 6925, 681)


class CaseSeedTests(unittest.TestCase):
    def test_seed_depends_only_on_base_and_case(self):
        case = {"case_id": 17, "route": "ignored"}
        self.assertEqual(case_sumo_seed(0, case), case_sumo_seed(0, {"case_id": 17}))
        seeds = {case_sumo_seed(base, {"case_id": i}) for base in (0, 1) for i in range(500)}
        self.assertEqual(len(seeds), 1000)
        self.assertTrue(all(0 <= s < 2 ** 31 for s in seeds))


@unittest.skipUnless(SUITE.is_dir() and MODEL.is_file(), "local diverse start-lane suite and final PPO required")
class IsolatedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # In-process evaluation loads the model's saved settings into these modules
        # (utils/saved_configs.py); restore them so later tests see the defaults.
        cls.settings = {module: {k: v for k, v in vars(module).items() if k.isupper()}
                        for module in (road_config, mdp_config, road_builder)}
        cls.tmp = TemporaryDirectory(prefix="isolated-eval-test-")
        cls.root = Path(cls.tmp.name)
        cls.context = cls.root / "context"
        evaluation_context(MODEL.parent, cls.context, max_vph=read_json(SUITE / "suite.json")["traffic_vph"])
        shutil.copy2(MODEL, cls.context / "model.pt")
        by_id = {c["case_id"]: c for c in read_json(SUITE / "cases.json")}
        cls.cases = [by_id[i] for i in CASES]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        for module, saved in cls.settings.items():
            for name in [k for k in vars(module) if k.isupper() and k not in saved]:
                delattr(module, name)
            for name, value in saved.items():
                setattr(module, name, value)

    def evaluate(self, name, cases, **kw):
        rows = evaluate_worker(SUITE, self.context, self.root / name, cases, None, 1, [], 0.0,
                               isolated=True, **kw)
        return {row["case_id"]: row for row in rows}

    def test_outcomes_do_not_depend_on_case_order_or_companions(self):
        forward = self.evaluate("forward", self.cases)
        backward = self.evaluate("backward", self.cases[::-1])
        alone = self.evaluate("alone", self.cases[2:3])
        self.assertEqual(forward, backward)
        case_id = self.cases[2]["case_id"]
        self.assertEqual(alone[case_id], forward[case_id])

    def test_replicas_argument_is_ignored(self):
        # Isolated mode always runs one case per SUMO, so a batched policy call cannot occur.
        rows = evaluate_worker(SUITE, self.context, self.root / "replicas", self.cases[:2], None, 64, [], 0.0,
                               isolated=True)
        self.assertEqual({r["case_id"]: r for r in rows}, self.evaluate("one", self.cases[:2]))

    def test_sumo_seed_changes_the_driver_randomness(self):
        base = self.evaluate("seed0", self.cases)
        other = self.evaluate("seed1", self.cases, sumo_seed=1)
        self.assertNotEqual([base[i]["ep_length"] for i in CASES], [other[i]["ep_length"] for i in CASES])

    def test_run_rows_do_not_depend_on_worker_count(self):
        results = []
        for workers in (1, 3):
            out = self.root / f"run_{workers}"
            run(SUITE, "diverse_start_lanes", None, MODEL, 64, 1, output=out, isolated=True, workers=workers)
            with (out / "route_evaluation.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            summary = json.loads((out / "evaluation_summary.json").read_text())
            self.assertEqual(summary["conditions"]["case_isolation"], "fresh_sumo_per_case")
            self.assertEqual(summary["conditions"]["replicas_per_worker"], 1)
            self.assertEqual(summary["conditions"]["evaluation_workers"], workers)
            self.assertEqual(len(rows), 4)  # smoke=1 keeps one case from each of the suite's 4 shards
            results.append([{k: v for k, v in row.items() if k != "worker"} for row in rows])
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
