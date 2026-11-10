"""Check that a replica steps and respawns like a standalone Phase 1 ego."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from env.sumo_env import SumoHighwayEnv
from train import ACTION, EGO, OBSERVATION, REWARD, ROAD, SIMULATION, TRAFFIC, road_builder

try:
    from env.parallel_sumo import ParallelSumo, libsumo
except ImportError:
    ParallelSumo = None


@unittest.skipUnless(ParallelSumo is not None, "libsumo is required")
class ParallelSumoTests(unittest.TestCase):
    def test_replica_matches_single_ego_and_respawns_without_an_extra_tick(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = road_builder.build(ROAD, EGO, TRAFFIC, str(root / "base"))
            kwargs = dict(cfg_path=config, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                          mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC)
            reference = SumoHighwayEnv(**kwargs)
            replicas = ParallelSumo(2, kwargs, root / "parallel", seed=0)
            try:
                replicas.reset()
                first, second = replicas.egos
                cases = [("ego_d_A1_A2", SIMULATION["max_steps"]),
                         ("ego_r_RING_A1_A3_A1_A3", 16),
                         ("ego_d_A2_A2", SIMULATION["max_steps"])]
                for route, max_steps in cases:
                    reference.max_steps = first.max_steps = max_steps
                    reference._choose_ego_route = lambda: route
                    expected, _ = reference.reset(seed=0)
                    actual = replicas._respawn(first, route)
                    np.testing.assert_allclose(actual, expected, atol=2e-6)
                    for step in range(160):
                        action = [0.2, 1 if step == 22 else -1 if step == 27 else 0]
                        expected, reward, term, trunc, _ = reference.step(action)
                        tick = libsumo.simulation.getTime()
                        second_steps = second.step_count
                        actual, actual_reward, actual_term, actual_trunc, timeout = (
                            replicas.step([action, [0.9, -1]])[0])
                        self.assertAlmostEqual(libsumo.simulation.getTime() - tick, first.step_length)
                        self.assertIn(second.step_count, (0, second_steps + 1))
                        self.assertAlmostEqual(actual_reward, reward, places=6)
                        self.assertEqual((actual_term, actual_trunc), (term, trunc))
                        if term or trunc:
                            self.assertEqual(first.step_count, 0)
                            self.assertTrue(first.present)
                            if trunc and not term:
                                np.testing.assert_allclose(timeout, expected, atol=2e-6)
                            break
                        np.testing.assert_allclose(actual, expected, atol=2e-6)
                    else:
                        self.fail(f"{route} did not terminate")
            finally:
                try:
                    replicas.close()
                finally:
                    reference.close()


if __name__ == "__main__":
    unittest.main()
