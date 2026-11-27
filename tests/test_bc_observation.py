"""BC43 layout and live SUMO distances; run from repository root."""
import math
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np
from env.sumo_env import SumoHighwayEnv
from env.route_guidance import choice_geometry, navigation_observation
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD
from tests.road_fixture import road_files


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.geometry = choice_geometry(ET.parse(road_files()[1]).getroot())
        self.vehicle = Mock()
        self.vehicle.getRoute.return_value = ('A1_in2', 'A1_in1', 'A1_in', 'A5_out')
        self.vehicle.getRouteIndex.return_value = 0
        self.vehicle.getRoadID.return_value = 'A1_in2'
        self.vehicle.getDrivingDistance.side_effect = lambda vid, edge, pos: {'A1_in2': 100, 'A1_in': 180}[edge]

    def test_two_choices_and_normalization(self):
        obs = navigation_observation(self.vehicle, 'ego', self.geometry)
        np.testing.assert_allclose(obs[:4], [1, 100/650, 0, 1], atol=.001)
        np.testing.assert_allclose(obs[4:], [1, 180/650, math.sin(math.radians(-24)), math.cos(math.radians(-24))], atol=.001)

    def test_internal_junction_removes_passed_choice(self):
        self.vehicle.getRoadID.return_value = ':nM_A1_1_0'
        obs = navigation_observation(self.vehicle, 'ego', self.geometry)
        self.assertAlmostEqual(obs[1], 180/650)
        self.assertEqual(obs[4:], [0]*4)

    def test_distances_over_scale_are_not_clipped(self):
        self.vehicle.getDrivingDistance.side_effect = None
        self.vehicle.getDrivingDistance.return_value = 1300
        obs = navigation_observation(self.vehicle, 'ego', self.geometry)
        self.assertEqual(obs[1], 2)

    def test_terminal_route_and_unreachable_choices(self):
        self.vehicle.getRouteIndex.return_value = 3
        self.assertEqual(navigation_observation(self.vehicle, 'ego', self.geometry), [0]*8)
        self.vehicle.getRouteIndex.return_value = 0
        self.vehicle.getDrivingDistance.side_effect = None
        self.vehicle.getDrivingDistance.return_value = -1
        self.assertEqual(navigation_observation(self.vehicle, 'ego', self.geometry), [0]*8)


class BCObservationTests(unittest.TestCase):
    def setUp(self):
        sim = dict(SIMULATION, warmup_steps=0, warmup_seconds=0)
        self.env = SumoHighwayEnv(str(road_files()[0]), ROAD, EGO,
                                 sim, OBSERVATION, ACTION, REWARD, traffic=TRAFFIC)

    def tearDown(self):
        self.env.close()

    def test_layout_signal_and_unclipped_distance(self):
        env = self.env
        self.assertEqual(env.obs_dim, 43)
        self.assertEqual(env.connectivity_offsets, (2, 1, 0, -1, -2))
        self.assertEqual(env._signal_observation(None), [0]*4)
        self.assertEqual(env._signal_observation(('tls', 0, 50, 'r')), [1,0,0,.25])
        self.assertEqual(env._signal_observation(('tls', 0, 50, 'y')), [0,1,0,.25])
        self.assertEqual(env._signal_observation(('tls', 0, 50, 'G')), [0,0,1,.25])
        with patch.object(env, '_ego_is_present', return_value=False):
            np.testing.assert_array_equal(env._get_obs(), np.zeros(43))

    def test_legacy_snapshot_layout_remains_available(self):
        config = dict(OBSERVATION, layout="legacy", connectivity_half_width=4,
                      lookahead_connectivity_enabled=True, destination_enabled=True,
                      signal_count=2)
        env = SumoHighwayEnv(str(road_files()[0]), ROAD, EGO,
                            SIMULATION, config, ACTION, REWARD, traffic=TRAFFIC)
        self.assertEqual(env.obs_dim, 59)
        self.assertEqual(env._signal_observation(None), [1,0,0,0,1])

    def test_stop_offset_is_subtracted_before_encoding(self):
        # 45m to controlled lane end, stop line is 8m upstream => 37/200.
        env = self.env
        env._signal_stop_offsets = {('tls', 0): 8}
        env._signal_stop_lines = {('tls', 0): {}}
        vehicle = Mock()
        vehicle.getNextTLS.return_value = [('tls',0,45,'r')]
        with patch('env.sumo_env.traci') as api, patch.object(env, '_ego_is_present', return_value=True):
            api.vehicle = vehicle
            self.assertEqual(env._signal_observation(env._upcoming_signals()[0]), [1,0,0,37/200])

    def test_live_observation_and_physical_connectivity(self):
        env = self.env
        env._choose_ego_route = lambda: 'ego_d_A1_A5'
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape, (43,))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs[35], 1)
        self.assertEqual(obs[39], 1)
        np.testing.assert_allclose(obs[41:43], [math.sin(math.radians(-24)),math.cos(math.radians(-24))], atol=.001)
        # Basic connectivity must not query the route-matching helper.
        with patch.object(env, '_lane_follows_route', side_effect=AssertionError('route leak')):
            env._lane_connectivity(0)
        net = ET.parse(road_files()[1]).getroot()
        connection = next(c for c in net.findall('connection')
                          if c.get('from') == 'A1_in' and c.get('to') == 'A5_out'
                          and c.get('tl') is not None)
        key = (connection.get('tl'), int(connection.get('linkIndex')))
        stop = env._signal_stop_lines[key]
        env._api.vehicle.moveTo(env.ego_id, stop['lane_id'], stop['lane_position'] - 20)
        signal = next(s for s in env._upcoming_signals() if s[:2] == key)
        self.assertAlmostEqual(signal[2], 20, places=3)
        self.assertAlmostEqual(env._signal_observation(signal)[3], .1, places=5)
        for _ in range(3):
            obs, _, term, trunc, _ = env.step(np.array([0, 0], dtype=np.float32))
            self.assertTrue(env.observation_space.contains(obs))
            if term or trunc:
                break
