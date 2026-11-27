"""Keep near vehicles visible and use one physical distance convention."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import xml.etree.ElementTree as ET

from env.sumo_env import SumoHighwayEnv
from env.connected_lane_gaps import ConnectedLaneGaps
from env.collision_detection import configure_collision_gaps


class PhysicalNeighborTests(unittest.TestCase):
    def setUp(self):
        self.env = SumoHighwayEnv.__new__(SumoHighwayEnv)
        self.env.ego_id = 'ego'
        self.env.visibility = 50.
        self.vehicle = Mock()
        self.vehicle.getMinGap.side_effect = lambda vid: {'ego': 2.5, 'front': 2., 'rear': 2.}[vid]
        self.vehicle.getSpeed.side_effect = lambda vid: {'front': 0., 'rear': 1.}[vid]

    def test_close_leader_remains_visible_and_old_encoding_is_explicit(self):
        with patch('env.sumo_env.traci', SimpleNamespace(vehicle=self.vehicle)):
            for physical_gap in [2.51, 2.49, 2.0, .21, .01, -.1]:
                self.vehicle.getLeader.return_value = ('front', physical_gap - 2.5)
                gap, speed = self.env._neighbor(0, True, physical=True)
                self.assertAlmostEqual(gap, max(physical_gap, 0))
                self.assertEqual(speed, 0)
            self.assertIsNone(self.env._neighbor(0, True))
            self.vehicle.getLeader.return_value = ('', -1)
            self.assertIsNone(self.env._neighbor(0, True, physical=True))

    def test_clamped_junction_leader_gap_uses_lane_geometry(self):
        # SUMO reports 0 for a leader on the next junction lane once it is closer than minGap.
        self.env.leader_gap_geometry = True
        self.vehicle.getLeader.return_value = ('front', 0.0)
        self.vehicle.getLaneID.side_effect = lambda vid: {'ego': 'A_1', 'front': ':n_0_0'}[vid]
        self.vehicle.getLanePosition.side_effect = lambda vid: {'ego': 45.5, 'front': 1.8}[vid]
        self.vehicle.getLength.return_value = 5.0
        lane = Mock()
        lane.getLinks.return_value = [('B_1', True, True, False, ':n_0_0', 'M', 's', 9.0)]
        lane.getLength.return_value = 46.5
        with patch('env.sumo_env.traci', SimpleNamespace(vehicle=self.vehicle, lane=lane)):
            gap, _ = self.env._neighbor(0, True, physical=True)
            self.assertEqual(gap, 0)                    # 1.0 m left on the lane + (1.8 - 5.0) = -2.2 m -> overlap
            self.vehicle.getLanePosition.side_effect = lambda vid: {'ego': 45.5, 'front': 6.2}[vid]
            gap, _ = self.env._neighbor(0, True, physical=True)
            self.assertAlmostEqual(gap, 2.2)            # 1.0 + (6.2 - 5.0)
            self.env.leader_gap_geometry = False
            gap, _ = self.env._neighbor(0, True, physical=True)
            self.assertAlmostEqual(gap, 2.5)            # previous behaviour: stuck at minGap

    def test_rear_gap_uses_followers_min_gap(self):
        self.vehicle.getFollower.return_value = ('rear', -1.7)
        with patch('env.sumo_env.traci', SimpleNamespace(vehicle=self.vehicle)):
            gap, speed = self.env._neighbor(0, False, physical=True)
        self.assertAlmostEqual(gap, .3)
        self.assertEqual(speed, 1)

    def test_adjacent_gap_and_visibility_use_physical_distance(self):
        with patch('env.sumo_env.traci', SimpleNamespace(vehicle=self.vehicle)):
            self.vehicle.getNeighbors.return_value = [('front', -2.3)]
            self.assertAlmostEqual(self.env._neighbor(1, True, physical=True)[0], .2)
            self.vehicle.getNeighbors.return_value = [('rear', -1.6)]
            self.assertAlmostEqual(self.env._neighbor(-1, False, physical=True)[0], .4)
            self.vehicle.getLeader.return_value = ('front', 49.)
            self.assertIsNone(self.env._neighbor(0, True, physical=True))

    def test_two_lane_scan_keeps_overlap_and_does_not_add_min_gap(self):
        self.env._scan_lane_neighbors = lambda offset: ((.3, 0.), (-.1, 1.))
        self.assertEqual(self.env._neighbor(2, True, physical=True), (.3, 0.))
        self.assertEqual(self.env._neighbor(-2, False, physical=True), (0., 1.))


    def test_adjacent_leader_on_internal_lane_stays_visible(self):
        lanes, edges = Mock(), Mock()
        edges.getLaneNumber.return_value = 2
        self.env._neighbor_internal_links = {}
        self.env._max_neighbor_length = 5.
        self.vehicle.getRoadID.return_value = "A5_in"
        self.vehicle.getLaneIndex.return_value = 0
        self.vehicle.getLanePosition.side_effect = {"ego": 34.6, "front": 1.5}.__getitem__
        self.vehicle.getLength.return_value = 5.
        self.vehicle.getLaneID.return_value = ":nC_5_1"
        lanes.getLength.return_value = 35.6
        lanes.getLinks.side_effect = lambda lane: (
            [(None, None, None, None, ":nC_5_1")] if lane == "A5_in_1" else [])
        lanes.getLastStepVehicleIDs.return_value = ["front"]
        api = SimpleNamespace(vehicle=self.vehicle, lane=lanes, edge=edges)
        with patch("env.sumo_env.traci", api):
            self.assertEqual(self.env._adjacent_junction_leader(1), (0., 0.))
            self.assertIsNone(self.env._adjacent_junction_leader(-1))
        lanes.getLastStepVehicleIDs.assert_called_once_with(":nC_5_1")


class ConnectedLaneGapTests(unittest.TestCase):
    def test_bumpers_across_internal_connections_on_current_and_adjacent_lanes(self):
        lengths = {"A_0": 10., "A_1": 10., ":via0": 2., ":via1": 4.,
                   "B_0": 30., "B_1": 30., "P_1": 30.}
        links = {"A_0": [("B_0", None, None, None, ":via0")],
                 ":via0": [("B_0", None, None, None, "")],
                 "A_1": [("B_1", None, None, None, ":via1")],
                 ":via1": [("B_1", None, None, None, "")],
                 "P_1": [("A_1", None, None, None, "")]}
        records = {"ego": ("A_0", 8., 5., 1.), "leader": (":via0", 1., 5., 2.),
                   "adjacent_leader": ("B_1", 4., 5., 3.), "follower": ("P_1", 29., 4., 4.)}
        lanes = SimpleNamespace(getIDList=lambda: list(lengths), getLength=lengths.__getitem__,
                                getLinks=lambda lane: links.get(lane, []),
                                getLastStepVehicleIDs=lambda lane: [vid for vid, row in records.items()
                                                                     if row[0] == lane])
        vehicle = SimpleNamespace(getLaneID=lambda vid: records[vid][0],
                                  getLanePosition=lambda vid: records[vid][1],
                                  getLength=lambda vid: records[vid][2],
                                  getSpeed=lambda vid: records[vid][3])
        gaps = ConnectedLaneGaps(50., 5.)
        api = SimpleNamespace(lane=lanes, vehicle=vehicle)
        self.assertEqual(gaps.around(api, "ego", (0, 1)), [((0., 2.), None), ((5., 3.), (4., 4.))])
        records["adjacent_leader"] = (":via1", 1., 5., 3.)
        records["follower"] = ("A_1", 6., 4., 4.)
        signed = ConnectedLaneGaps(50., 5., signed_adjacent=True)
        self.assertEqual(signed.around(api, "ego", (0, 1)), [((0., 2.), None), ((-2., 3.), (-3., 4.))])


class CollisionGapConfigurationTests(unittest.TestCase):
    def test_all_expert_types_share_gap_without_changing_following_parameters(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'road.sumocfg'
            config.write_text('<configuration><input><route-files value="types.xml"/></input></configuration>')
            path = root / 'types.xml'
            path.write_text('<routes><vType id="ego_type"/><vTypeDistribution id="cars">'
                            '<vType id="car_krauss" minGap="2.5"/>'
                            '<vType id="car_idm" minGap="2.0"/></vTypeDistribution></routes>')
            result = configure_collision_gaps(config, ego_gap_m=.2, background_gap_m=.2)
            self.assertEqual(set(result), {'ego_type', 'car_krauss', 'car_idm'})
            for node in ET.parse(path).getroot().iter('vType'):
                min_gap = float(node.get('minGap', '2.5'))
                self.assertAlmostEqual(min_gap * float(node.get('collisionMinGapFactor')), .2)
            self.assertEqual(result['car_idm']['min_gap_m'], 2.)


if __name__ == '__main__':
    unittest.main()
