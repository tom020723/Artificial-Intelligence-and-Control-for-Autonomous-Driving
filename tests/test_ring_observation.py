import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from env.ring_observation import ring_observation, EMPTY, prefix_geometry
from tests.road_fixture import road_files


class RingObservationTests(unittest.TestCase):
    def test_empty_outside_ring(self):
        api=SimpleNamespace(vehicle=Mock())
        api.vehicle.getLaneID.return_value='A1_in_0'
        self.assertEqual(ring_observation(api,'ego',{},10),EMPTY)

    def test_other_road_approach_sorted_and_missing(self):
        v=Mock(); lane=Mock();api=SimpleNamespace(vehicle=v,lane=lane)
        v.getLaneID.return_value='RING_A3_A5_0'
        lane.getLastStepVehicleIDs.return_value=['far','near','passed']
        v.getPosition.side_effect=lambda x:(0,0) if x=='ego' else (24,0)
        v.getDrivingDistance.side_effect=lambda x,*_:dict(ego=15,far=30,near=10,passed=-1)[x]
        v.getSpeed.side_effect=lambda x:dict(far=8,near=6)[x]
        geom={'RING_A3_A5_0':dict(goal='A5_out1',lanes=['A5_out_0'])}
        self.assertEqual(ring_observation(api,'ego',geom,10),[1,.3,.6,.2,.8,.6])
        lane.getLastStepVehicleIDs.return_value=['near']
        self.assertEqual(ring_observation(api,'ego',geom,10),[1,.3,.6,.2,0,-1])
        lane.getLastStepVehicleIDs.return_value=[]
        self.assertEqual(ring_observation(api,'ego',geom,10),[1,.3,0,-1,0,-1])

    def test_visibility_is_physical_not_only_route_distance(self):
        v=Mock();lane=Mock();api=SimpleNamespace(vehicle=v,lane=lane)
        v.getLaneID.return_value='ring_0';lane.getLastStepVehicleIDs.return_value=['hidden']
        v.getPosition.side_effect=lambda x:(0,0) if x=='ego' else (51,0)
        v.getDrivingDistance.return_value=10
        self.assertEqual(ring_observation(api,'ego',{'ring_0':dict(goal='out',lanes=['in_0'])},10),[1,.2,0,-1,0,-1])

    def test_prefixes_geometry_without_changing_distances(self):
        g={'RING_A3_A5_0':dict(goal='A5_out1',lanes=('A5_out_0',':merge_0_0'))}
        p=lambda s: ':r0_'+s[1:] if s.startswith(':') else 'r0_'+s
        self.assertEqual(prefix_geometry(g,p)['r0_RING_A3_A5_0']['lanes'],('r0_A5_out_0',':r0_merge_0_0'))

    def test_current_network_contains_hidden_merge_approach(self):
        import xml.etree.ElementTree as ET
        from env.ring_observation import ring_geometry
        net=ET.parse(road_files()[1]).getroot()
        plans=ring_geometry(net)
        self.assertIn('A5_out_0',plans['RING_A3_A5_0']['lanes'])
        self.assertIn(':nM_A5_0_6_0',plans['RING_A3_A5_0']['lanes'])
        self.assertEqual(plans['RING_A3_A5_0']['goal'],'A5_out1')

    def test_direct_scope_uses_only_one_straight_mainline_lane(self):
        import xml.etree.ElementTree as ET
        from env.ring_observation import ring_geometry

        net = ET.fromstring("""
        <net>
          <edge id="RING_A5_A2"><lane id="RING_A5_A2_0" index="0"/></edge>
          <edge id="A2_out"><lane id="A2_out_0" index="0"/></edge>
          <edge id="A2_in1"><lane id="A2_in1_3" index="3"/></edge>
          <edge id="A2_out1"/>
          <connection from="RING_A5_A2" fromLane="0" to="A2_out1" toLane="0" dir="r"/>
          <connection from="A2_out" fromLane="0" to="A2_out1" toLane="0" dir="s" via=":main_0"/>
          <connection from="A2_in1" fromLane="3" to="A2_out1" toLane="0" dir="t" tl="nC"/>
        </net>
        """)
        plans = ring_geometry(net, scope="direct")
        self.assertEqual(plans["RING_A5_A2_0"], dict(goal="A2_out1", lanes=("A2_out_0",)))
        vehicle, lane = Mock(), Mock()
        vehicle.getLaneID.return_value = "RING_A5_A2_0"
        vehicle.getPosition.return_value = (0, 0)
        vehicle.getDrivingDistance.side_effect = lambda vid, *_: {"ego": 15, "main": 20}[vid]
        vehicle.getSpeed.return_value = 8
        lane.getLastStepVehicleIDs.return_value = ["main"]
        obs = ring_observation(SimpleNamespace(vehicle=vehicle, lane=lane), "ego", plans, 10)
        self.assertEqual(obs, [1, .3, .8, .4, 0, -1])
        lane.getLastStepVehicleIDs.assert_called_once_with("A2_out_0")
