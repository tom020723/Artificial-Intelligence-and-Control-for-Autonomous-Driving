"""Regression coverage for PR #7 integration with the final BC/PPO artifacts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from collect_pomdp_data import detect_algorithm, ppo_run_directory, write_metadata
from env.sumo_env import SumoHighwayEnv


class PolicyMetadataTests(unittest.TestCase):
    def test_archived_bc_needs_no_training_state(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "config.json").write_text(json.dumps(
                {"algorithm": "supervised_bc_on_ppo_hybrid_actor"}))
            self.assertEqual(detect_algorithm(str(folder / "model.pt")), "bc")

    def test_ppo_export_takes_precedence_over_copied_bc_metadata(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "run.json").write_text('{"algorithm": "AnchoredPPO"}')
            export = run / "eval_runs" / "iter080"
            export.mkdir(parents=True)
            (export / "config.json").write_text(json.dumps(
                {"algorithm": "supervised_bc_on_ppo_hybrid_actor"}))
            model = str(export / "model.pt")
            self.assertEqual(detect_algorithm(model), "ppo")
            self.assertEqual(ppo_run_directory(model), run.resolve())

    def test_metadata_decodes_korean_git_diff_as_utf8(self):
        with TemporaryDirectory() as tmp, patch("subprocess.run") as git:
            git.return_value = SimpleNamespace(stdout="한글 변경\n")
            output = Path(tmp) / "metadata.json"
            write_metadata(SimpleNamespace(seed=0), {}, output)
            meta = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(meta["code_snapshot"]["git_diff"], "한글 변경")
            self.assertEqual(git.call_args.kwargs["encoding"], "utf-8")


class CollisionColumnsTests(unittest.TestCase):
    def test_partner_record_when_ego_is_victim(self):
        env = SumoHighwayEnv.__new__(SumoHighwayEnv)
        env.ego_id = "ego"
        api = Mock()
        api.simulation.getCollisions.return_value = [SimpleNamespace(
            collider="background", victim="ego", colliderType="car_idm",
            victimType="ego_type", colliderSpeed=7.5, victimSpeed=2.0)]
        with patch.object(SumoHighwayEnv, "_api", new=property(lambda _: api)):
            self.assertEqual(env._collision_partner(True), dict(
                collision_partner_id="background", collision_partner_type=1,
                collision_partner_speed=7.5, ego_is_collider=False))
            self.assertTrue(all(v is None for v in env._collision_partner(False).values()))


if __name__ == "__main__":
    unittest.main()
