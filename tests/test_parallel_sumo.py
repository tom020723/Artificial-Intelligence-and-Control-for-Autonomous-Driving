"""Check that a replica steps and respawns like a standalone Phase 1 ego."""

from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from env.sumo_env import SumoHighwayEnv
from env.parallel_traffic import TrafficCurriculum
from train import ACTION, EGO, OBSERVATION, REWARD, ROAD, SIMULATION, TRAFFIC, road_builder

try:
    from env.parallel_sumo import ParallelSumo, ShardedParallelSumo, libsumo
except ImportError:
    ParallelSumo = None


@unittest.skipUnless(ParallelSumo is not None, "libsumo is required")
class ParallelSumoTests(unittest.TestCase):
    def test_worker_shards_match_single_simulator_in_global_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(road_builder.NOHYEONG, all_signals_green=False):
                config = road_builder.build(ROAD, EGO, TRAFFIC, str(root / "base"))
            kwargs = dict(cfg_path=config, road=ROAD, ego=EGO,
                          mdp_sim={**SIMULATION, "max_steps": 2}, mdp_obs=OBSERVATION,
                          action=ACTION, reward=REWARD, traffic=TRAFFIC)
            reference = ParallelSumo(4, kwargs, root / "reference", 0,
                                     randomize_signals=True, randomize_departures=True)
            template = ParallelSumo(2, kwargs, root / "shard", 0,
                                    randomize_signals=True, randomize_departures=True)
            sharded = ShardedParallelSumo(template, 2)
            routes = ["ego_d_A1_A2", "ego_d_A2_A3", "ego_d_A3_A5", "ego_d_A4_A5"]
            actions = np.array([[-0.2, 0], [0.2, 0], [0, 0], [0.4, 0]], dtype=np.float32)
            try:
                np.testing.assert_allclose(sharded.reset(routes), reference.reset(routes), atol=2e-6)
                for tick in range(4):
                    for expected, actual in zip(reference.step(actions), sharded.step(actions), strict=True):
                        np.testing.assert_allclose(actual[0], expected[0], atol=2e-6)
                        self.assertAlmostEqual(actual[1], expected[1], places=6)
                        self.assertEqual(actual[2:4], expected[2:4])
                        if tick % 2:
                            self.assertTrue(actual[3])
                            np.testing.assert_allclose(actual[4], expected[4], atol=2e-6)
                        else:
                            self.assertIsNone(actual[4])
            finally:
                sharded.close()
                reference.close()

    def test_phase2_samples_signal_offset_and_only_resets_its_own_road(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(road_builder.NOHYEONG, all_signals_green=False):
                config = road_builder.build(ROAD, EGO, TRAFFIC, str(root / "base"))
            kwargs = dict(cfg_path=config, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                          mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC)
            env = ParallelSumo(2, kwargs, root / "parallel", seed=0, randomize_signals=True)
            route = "ego_d_A1_A2"
            rngs = [np.random.default_rng(seed) for seed in np.random.SeedSequence(0).spawn(2)]
            ends = np.cumsum([40, 6, 4] * 5)

            def signal_state(index):
                light = f"r{index}_nC"
                return (libsumo.trafficlight.getPhase(light),
                        libsumo.trafficlight.getNextSwitch(light) - libsumo.simulation.getTime())

            def expected(index):
                offset = rngs[index].integers(250)
                phase = np.searchsorted(ends, offset, side="right")
                return phase, (ends[phase] - offset) * .5

            try:
                env.reset(routes=[route, route])
                self.assertEqual(signal_state(0), expected(0))
                self.assertEqual(signal_state(1), expected(1))
                self.assertNotEqual(signal_state(0), signal_state(1))
                other = signal_state(1)
                env._respawn(env.egos[0], route)
                self.assertEqual(signal_state(0), expected(0))
                self.assertEqual(signal_state(1), other)
            finally:
                env.close()

    def test_new_episode_samples_lane_and_signal_independently_of_route(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(road_builder.NOHYEONG, all_signals_green=False):
                config = road_builder.build(ROAD, EGO, TRAFFIC, str(root / "base"))
            kwargs = dict(cfg_path=config, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                          mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC)
            env = ParallelSumo(8, kwargs, root / "parallel", seed=0,
                               randomize_signals=True, randomize_departures=True)
            fixed = ParallelSumo(8, kwargs, root / "fixed", seed=0,
                                 randomize_signals=True, independent_episode_rng=True)
            route = "ego_d_A2_A1"
            try:
                env.reset()
                sampled = [ego.signal_offset_s for ego in env.egos]
                sampled_routes = [ego.ego_route_id for ego in env.egos]
                env.reset(routes=[route] * env.num_envs)
                self.assertEqual([ego.signal_offset_s for ego in env.egos], sampled)
                lanes = [libsumo.vehicle.getLaneIndex(ego.ego_id) for ego in env.egos]
                self.assertEqual(lanes, [ego.start_lane for ego in env.egos])
                self.assertGreater(len(set(lanes)), 1)
                for ego in env.egos:
                    self.assertEqual(libsumo.vehicle.getRoadID(ego.ego_id),
                                     ego.prefix(env.route_entries[route]))
                env.close()
                fixed.reset()
                self.assertEqual([ego.ego_route_id for ego in fixed.egos], sampled_routes)
                self.assertEqual([ego.signal_offset_s for ego in fixed.egos], sampled)
                for ego in fixed.egos:
                    edge = libsumo.vehicle.getRoadID(ego.ego_id)
                    self.assertEqual(libsumo.vehicle.getLaneIndex(ego.ego_id),
                                     libsumo.edge.getLaneNumber(edge) // 2)
            finally:
                fixed.close()
                env.close()

    def test_curriculum_changes_only_ended_road_and_clears_its_traffic(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            road = runpy.run_path(str(Path(road_builder.__file__).with_name("road_config.py")))
            road["configure_parallel_training"](2)
            with patch.object(road_builder, "NOHYEONG", road["NOHYEONG"]):
                config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"], str(root / "base"))
            kwargs = dict(cfg_path=config, road=road["ROAD"], ego=road["EGO"],
                          mdp_sim=SIMULATION, mdp_obs=OBSERVATION, action=ACTION,
                          reward=REWARD, traffic=road["TRAFFIC"])
            env = ParallelSumo(2, kwargs, root / "parallel", seed=0, randomize_signals=True,
                               randomize_departures=True, traffic_curriculum=TrafficCurriculum())
            try:
                env.reset(routes=["ego_d_A2_A1", "ego_d_A2_A1"])
                first, second = env.egos
                self.assertEqual(env.traffic_levels(), [0, 0])
                first.background.vph, second.background.vph = 1480, 600
                flow = next(flow for flow in env.traffic_flows if flow.route == "r_A1_A2")
                for ego in env.egos:
                    for i in range(3):
                        vid = ego.prefix(f"test_bg_{i}")
                        libsumo.vehicle.add(vid, ego.prefix(flow.route), typeID=flow.vehicle_type,
                                           departLane="0", departSpeed="0")
                        ego.background.ids.add(vid)
                old_first, old_second = set(first.background.ids), set(second.background.ids)
                finish_step = first._finish_step

                def complete(*args, **kwargs):
                    obs, reward, _, _, info = finish_step(*args, **kwargs)
                    pending = set(libsumo.simulation.getPendingVehicles())
                    self.assertTrue(old_first & pending)
                    self.assertTrue(old_first & set(libsumo.vehicle.getIDList()))
                    return obs, reward, True, False, {**info, "arrived": True}

                with patch.object(first, "_finish_step", side_effect=complete):
                    env.step([[0, 0], [0, 0]])
                self.assertEqual(env.traffic_levels(), [1500, 600])
                self.assertEqual(second.episode, 1)
                present = set(libsumo.vehicle.getIDList()) | set(libsumo.simulation.getPendingVehicles())
                self.assertFalse(old_first & present)
                self.assertEqual(old_second & present, old_second)

                first.max_steps = 1
                env.step([[0, 0], [0, 0]])
                self.assertEqual(env.traffic_levels(), [1350, 600])

                # A real background car on the ego's lane must enter its observation.
                flow = next(flow for flow in env.traffic_flows if flow.route == "r_A2_A1")
                leader = second.prefix("test_leader")
                libsumo.vehicle.add(leader, second.prefix(flow.route), typeID=flow.vehicle_type)
                lane = libsumo.vehicle.getLaneID(second.ego_id)
                libsumo.vehicle.moveTo(leader, lane, 35)
                gap, _ = second._neighbor(0, leader=True)
                self.assertGreater(gap, 0)
                self.assertLess(gap, second.visibility)
                self.assertIsNone(first._neighbor(0, leader=True))
            finally:
                env.close()

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
