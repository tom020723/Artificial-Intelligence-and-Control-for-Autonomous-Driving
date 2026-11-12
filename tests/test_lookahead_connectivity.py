"""Two-route-edge guidance tested against actual SUMO lane connections."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import traci
from env.road_builder import build
from env.road_config import ROAD,EGO,TRAFFIC
from env.mdp_config import SIMULATION,OBSERVATION,ACTION
from env.sumo_env import SumoHighwayEnv
from train import REWARD


class LookaheadConnectivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='sumo-lookahead-test-')
        cls.cfg=build(ROAD,EGO,TRAFFIC,cls.tmp.name)
        cls.net=ET.parse(Path(cls.tmp.name)/'highway.net.xml').getroot()
        cls.connections=cls.net.findall('connection')
        cls.routes={r.get('id'):r.get('edges').split() for r in
                    ET.parse(Path(cls.tmp.name)/'highway.ego.rou.xml').getroot().findall('route')
                    if r.get('id').startswith(('ego_d_','ego_r_'))}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        # 이 테스트 클래스는 "full 59차원" 레이아웃(destination one-hot + 신호 2개)을
        # 검증한다. OBSERVATION 기본값은 Phase 2에서 49차원으로 바뀌었으므로
        # 여기서 명시적으로 59차원 구성으로 되돌려 지정한다.
        self.obs_config = {**OBSERVATION, 'lookahead_connectivity_enabled': True,
                           'destination_enabled': True, 'signal_count': 2}
        self.env = SumoHighwayEnv(self.cfg, ROAD, EGO, SIMULATION, self.obs_config,
                                  ACTION, REWARD, traffic=TRAFFIC)

    def tearDown(self):
        self.env.close()

    def reset_route(self,route):
        self.env._choose_ego_route=lambda:route
        return self.env.reset(seed=0)[0]

    def values_by_lane(self):
        index=traci.vehicle.getLaneIndex('ego')
        return {index+offset:value for offset,value in
                zip(self.env.connectivity_offsets,self.env._lookahead_connectivity_observation())
                if value!=-1}

    def test_A2_departure_guidance_distinguishes_destinations_before_50m(self):
        expected={'A1':{0:1,1:1,2:1,3:0},'A3':{0:0,1:0,2:0,3:1},'A4':{0:1,1:0,2:0,3:0}}
        for arm,values in expected.items():
            with self.subTest(destination=arm):
                obs=self.reset_route(f'ego_d_A2_{arm}')
                lane=traci.vehicle.getLaneID('ego')
                self.assertGreater(traci.lane.getLength(lane)-traci.vehicle.getLanePosition('ego'),50)
                self.assertEqual(self.values_by_lane(),values)
                # Original one-edge, distance-limited observations remain unchanged.
                for offset in self.env.connectivity_offsets:
                    index=traci.vehicle.getLaneIndex('ego')+offset
                    if index in values: self.assertEqual(self.env._lane_connectivity(offset),1.)
                self.assertEqual(obs.shape,(59,))
                self.assertTrue(self.env.observation_space.contains(obs))

    def test_branching_lane_uses_any_valid_chain_and_preserves_lane_identity(self):
        self.reset_route('ego_d_A2_A4')
        # A2_in1_0 branches to A2_in_0 and _1. Only _0 reaches the Ring.
        self.assertEqual(self.values_by_lane()[0],1.)
        self.reset_route('ego_d_A2_A1')
        # Only A2_in_1 reaches A1, but the same upstream lane remains valid.
        self.assertEqual(self.values_by_lane()[0],1.)
        self.assertEqual(self.values_by_lane()[3],0.)

    def test_guidance_is_not_a_wrong_lane_termination_rule(self):
        self.reset_route('ego_d_A2_A3')
        lane='A2_in1_2'
        traci.vehicle.moveTo('ego',lane,traci.lane.getLength(lane)-2)
        traci.vehicle.setSpeed('ego',0)
        self.assertEqual(self.values_by_lane()[2],0.)
        self.assertTrue(self.env._lane_follows_route(lane))
        self.assertFalse(self.env._stuck_in_wrong_lane())
        _,reward,terminated,truncated,info=self.env.step(np.array([0,0],dtype=np.float32))
        self.assertFalse(terminated or truncated)
        self.assertFalse(info['wrong_lane'])
        self.assertAlmostEqual(reward,REWARD['speed_weight']*info['speed']/self.env.vmax)

    def test_internal_junction_lanes_have_actual_guidance_instead_of_all_ones(self):
        self.reset_route('ego_d_A2_A3')
        connection=next(c for c in self.connections if c.get('from')=='A2_in1' and c.get('fromLane')=='3' and c.get('to')=='A2_in')
        traci.vehicle.moveTo('ego',connection.get('via'),1.)
        self.assertEqual(traci.vehicle.getRouteIndex('ego'),0)
        self.assertTrue(traci.vehicle.getRoadID('ego').startswith(':'))
        values=self.values_by_lane()
        current=traci.vehicle.getLaneIndex('ego')
        self.assertEqual(values[current],1.)
        self.assertIn(0.,values.values())
        self.assertTrue(self.env.observation_space.contains(self.env._get_obs()))

    def test_split_central_internal_lanes_keep_the_route_index_aligned(self):
        self.reset_route('ego_d_A2_A3')
        connection=next(c for c in self.connections if c.get('from')=='A2_in' and c.get('fromLane')=='4' and c.get('to')=='A3_out')
        first=connection.get('via')
        internal_edge=first.rsplit('_',1)[0]
        next_connection=next(c for c in self.connections if c.get('from')==internal_edge)
        for lane in [first,next_connection.get('via')]:
            self.assertIsNotNone(lane)
            traci.vehicle.moveTo('ego',lane,0.)
            self.assertEqual(traci.vehicle.getRouteIndex('ego'),1)
            self.assertEqual(self.values_by_lane()[traci.vehicle.getLaneIndex('ego')],1.)

    def test_last_edge_and_single_remaining_edge_are_handled(self):
        self.reset_route('ego_d_A2_A2')
        self.assertEqual(self.values_by_lane(),{0:0,1:0,2:0,3:1})
        traci.vehicle.moveTo('ego','A2_out1_1',10.)
        self.assertEqual(self.values_by_lane(),{0:1,1:1,2:1})
        self.assertIn(-1.,self.env._lookahead_connectivity_observation())

    def test_signal_color_does_not_change_static_route_guidance(self):
        self.reset_route('ego_d_A2_A3')
        original=self.env._lookahead_connectivity_observation()
        for color in ['r','g']:
            for tid in traci.trafficlight.getIDList():
                traci.trafficlight.setRedYellowGreenState(tid,color*len(traci.trafficlight.getRedYellowGreenState(tid)))
            self.assertEqual(self.env._lookahead_connectivity_observation(),original)

    def test_new_layout_preserves_original_blocks_and_missing_key_is_legacy(self):
        obs=self.reset_route('ego_d_A2_A3')
        for use_key in [False,True]:
            config={**self.obs_config,'lookahead_connectivity_enabled':False}
            if not use_key: config.pop('lookahead_connectivity_enabled')
            old_env=SumoHighwayEnv(self.cfg,ROAD,EGO,SIMULATION,config,ACTION,REWARD,traffic=TRAFFIC)
            old=old_env._get_obs()
            self.assertEqual(old.shape,(50,))
            np.testing.assert_array_equal(obs[:45],old[:45])
            np.testing.assert_array_equal(obs[-5:],old[-5:])
        np.testing.assert_array_equal(obs[45:54],self.env._lookahead_connectivity_observation())
        np.testing.assert_array_equal(obs[-5:],[0,0,1,0,0])

    def test_destination_can_still_be_disabled_independently(self):
        self.reset_route('ego_d_A2_A3')
        config={**self.obs_config,'destination_enabled':False}
        env=SumoHighwayEnv(self.cfg,ROAD,EGO,SIMULATION,config,ACTION,REWARD,traffic=TRAFFIC)
        self.assertEqual(env.obs_dim,54)
        self.assertEqual(env._get_obs().shape,(54,))

    def test_terminal_observation_has_59_zeroes(self):
        with patch('traci.vehicle.getIDList',return_value=[]):
            np.testing.assert_array_equal(self.env._get_obs(),np.zeros(59,dtype=np.float32))

    def test_internal_link_search_handles_multiple_internal_edges_and_cycles(self):
        links={'start':(('i0',),),'i0':(('i1',),),'i1':(('i0',),('target',),('other',))}
        edges={'i0':':junction0','i1':':junction1','target':'next','other':'wrong'}
        with patch('traci.lane.getLinks',side_effect=lambda lane:links[lane]),patch('traci.lane.getEdgeID',side_effect=lambda lane:edges[lane]):
            self.assertEqual(self.env._route_successor_lanes('start','next'),{'target'})
            self.assertEqual(self.env._route_successor_lanes('start','unreachable'),set())

    def test_every_route_edge_matches_independent_network_lane_chains(self):
        checked=0
        for route,edges in sorted(self.routes.items()):
            self.reset_route(route)
            for index,edge in enumerate(edges):
                traci.vehicle.moveTo('ego',f'{edge}_0',0.)
                self.assertEqual(traci.vehicle.getRouteIndex('ego'),index)
                actual=self.values_by_lane()
                for lane,value in actual.items():
                    reachable={(edge,str(lane))}
                    for next_edge in edges[index+1:index+3]:
                        reachable={(c.get('to'),c.get('toLane')) for c in self.connections
                                   if (c.get('from'),c.get('fromLane')) in reachable and c.get('to')==next_edge}
                    self.assertEqual(value,float(bool(reachable)),(route,edge,lane))
                    checked+=1
                self.assertTrue(self.env.observation_space.contains(self.env._get_obs()))
        self.assertGreater(checked,200)

if __name__=='__main__': unittest.main()
