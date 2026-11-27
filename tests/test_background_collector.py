"""Expert-label timing, demand, and shared observation safety."""
import xml.etree.ElementTree as ET
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from collect_background_data import (action_from_speeds, lane_change_events,
                                     build_collection_road, BackgroundObserver, targets_reached)
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD
from env import road_builder


class BackgroundCollectorTests(unittest.TestCase):
    def test_lane_change_target_does_not_stop_at_total_sample_target(self):
        reports=[{'counts':dict(samples=30000,left_changes=100,right_changes=100)} for _ in range(5)]
        self.assertFalse(targets_reached(reports,100000,10000))
        self.assertTrue(targets_reached(reports,100000,1000))
        self.assertTrue(targets_reached(reports,0,1000))
        self.assertFalse(targets_reached(reports,0,0))
        self.assertFalse(targets_reached(reports[:4],0,1))

    def test_acceleration_uses_next_speed_and_asymmetric_ego_scales(self):
        self.assertEqual(action_from_speeds(10,11,.5,2,4), (1,2,False))
        self.assertEqual(action_from_speeds(10,9,.5,2,4), (-.5,-2,False))
        self.assertEqual(action_from_speeds(10,6,.5,2,4), (-1,-8,True))

    def test_sumos_native_event_labels_ignore_road_lane_index_resets(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'changes.xml'
            path.write_text('<lanechanges><change id="car" time="10.0" '
                            'dir="-1" from="A_in_2" to="A_in_1"/></lanechanges>')
            self.assertEqual(lane_change_events(path,.5), {('car',20):-1})

    def test_private_road_total_flow_and_readonly_observation_context(self):
        original = road_builder.NOHYEONG
        with tempfile.TemporaryDirectory() as directory:
            cfg,traffic,vph,config = build_collection_road(Path(directory)/'sumo',3000)
            self.assertIs(road_builder.NOHYEONG,original)
            self.assertFalse(config['all_signals_green'])
            network = ET.parse(Path(directory)/'sumo/highway.net.xml')
            self.assertGreater(len(network.findall("tlLogic[@id='nC']/phase")), 1)
            self.assertAlmostEqual(vph,3000,places=5)
            self.assertTrue(traffic['background_enabled'])
            for vtype in ET.parse(Path(directory)/'sumo/highway.rou.xml').getroot().iter('vType'):
                if not vtype.get('id', '').startswith('car_'):
                    continue
                self.assertEqual(float(vtype.get('lcSpeedGain')),1.0)
                self.assertEqual(float(vtype.get('lcKeepRight')),0.0)
                self.assertIsNone(vtype.get('lcStrategic'))
                self.assertIsNone(vtype.get('lcCooperative'))
            env=BackgroundObserver(cfg,ROAD,EGO,SIMULATION,OBSERVATION,ACTION,REWARD,traffic=traffic)
            with patch.object(env,'_get_obs',side_effect=lambda: env.ego_id):
                self.assertEqual(env.observe_vehicle('background'), 'background')
            self.assertEqual(env.ego_id,'ego')
            with patch.object(env,'_get_obs',side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):env.observe_vehicle('background')
            self.assertEqual(env.ego_id,'ego')

class ProportionalDemandTests(unittest.TestCase):
    def test_requested_origin_flow_matches_entry_lanes_including_ring(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg,traffic,total,_=build_collection_road(Path(directory)/'sumo',3000,True)
            expected={'A1_in2':3000*5/19,'A2_in1':3000*4/19,'A3_in1':3000*4/19,
                      'A4_in2':3000*3/19,'A5_in1':3000*3/19}
            self.assertAlmostEqual(total,3000,places=5)
            for edge,value in expected.items():
                self.assertAlmostEqual(traffic['origin_requested_vph'][edge],value,places=5)
