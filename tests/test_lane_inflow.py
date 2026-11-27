"""Queue admission and discharge accounting invariants."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from env.lane_inflow import LaneInflowController, LaneMeter
from env.parallel_traffic import TrafficFlow


class LaneMeterTests(unittest.TestCase):
    def test_empty_and_partial_greens_do_not_reduce_service_estimate(self):
        meter = LaneMeter('lane', 125, bootstrap_vph=300)
        meter.observe(True, 0, 5, .5)  # incomplete first observed phase
        meter.observe(False, 3, 0, 0)
        self.assertEqual(list(meter.samples), [])
        meter.observe(True, 0, 0, 0)
        meter.observe(False, 0, 0, 0)  # empty full phase
        self.assertEqual(meter.baseline_vph, 300)
        meter.observe(True, 0, 5, .5)
        meter.observe(True, 2, 3, .3)
        meter.observe(False, 4, 0, 0)
        self.assertEqual(meter.mean_departures, 6)
        self.assertAlmostEqual(meter.baseline_vph, 6 * 3600 / 125)

    def test_two_states_hysteresis_and_spillback_hold(self):
        meter = LaneMeter('lane', 125, bootstrap_vph=100)
        meter.queue_ratio = .2
        self.assertAlmostEqual(meter.budget(), 110)
        meter.queue_ratio = .5
        self.assertAlmostEqual(meter.budget(), 110)
        meter.queue_ratio = .6
        self.assertAlmostEqual(meter.budget(), 90)
        meter.queue_ratio = .4
        self.assertAlmostEqual(meter.budget(), 90)
        meter.queue_ratio = .9
        self.assertEqual(meter.budget(), 0)

    def test_blocked_green_is_a_valid_zero_service_sample(self):
        meter = LaneMeter('lane', 125, bootstrap_vph=100)
        meter.observe(False, 0, 3, .6)
        meter.observe(True, 0, 3, .6)
        meter.observe(False, 0, 3, .6)
        self.assertEqual(meter.mean_departures, 0)
        self.assertLess(meter.budget(), 100)

    def test_backlog_reserves_service_and_reopens_after_green_clears(self):
        meter = LaneMeter('lane', 125)
        meter.samples.append(5)
        meter.observe(False, 0, 0, 0)
        meter.observe(True, 0, 10, .2)
        meter.observe(False, 5, 10, .2)
        self.assertTrue(meter.draining)
        self.assertEqual(meter.budget(), 0)
        meter.observe(True, 0, 10, .2)
        meter.observe(False, 6, 4, .2)
        self.assertTrue(meter.draining)
        self.assertGreater(meter.budget(), 0)
        self.assertLess(meter.budget(), meter.baseline_vph)
        meter.observe(True, 0, 4, .2)
        meter.observe(False, 4, 1, .1)
        self.assertFalse(meter.draining)

    def test_long_queue_that_clears_does_not_enter_drain_mode(self):
        meter = LaneMeter('lane', 125)
        meter.observe(False, 0, 20, .8)
        meter.observe(True, 0, 20, .8)
        meter.observe(False, 20, 0, 0)
        self.assertFalse(meter.draining)

    def test_moving_vehicles_still_before_stop_line_reserve_next_green(self):
        meter = LaneMeter('lane', 125)
        meter.samples.append(5)
        meter.observe(False, 0, 0, 0)
        meter.observe(True, 0, 5, .4)
        meter.observe(False, 5, 0, 0, committed=10)
        self.assertEqual(meter.residual_queued, 10)
        self.assertTrue(meter.draining)
        self.assertEqual(meter.budget(), 0)

    def test_small_stagnant_residual_triggers_after_two_greens(self):
        meter = LaneMeter('lane', 125)
        meter.samples.append(10)
        meter.observe(False, 0, 0, 0)
        for _ in range(2):
            meter.observe(True, 0, 5, .2)
            meter.observe(False, 10, 3, .2)
        self.assertTrue(meter.draining)
        self.assertLess(meter.budget(), meter.baseline_vph)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / 'net.xml'
        path.write_text('''<net>
          <edge id="A1_in"><lane id="A1_in_0" length="100"/><lane id="A1_in_1" length="100"/></edge>
          <edge id="A2_out"><lane id="A2_out_0" length="100"/></edge>
          <tlLogic id="nC"><phase duration="20"/><phase duration="105"/></tlLogic>
          <connection from="A1_in" fromLane="0" to="A2_out" toLane="0" via=":j_0_0" tl="nC" linkIndex="0"/>
          <connection from="A1_in" fromLane="1" to="A2_out" toLane="0" via=":j_1_0" tl="nC" linkIndex="1"/>
        </net>''', encoding='utf-8')
        flow = TrafficFlow('route', 'car', 'random', '8', 1.)
        self.controller = LaneInflowController(path, [flow], {'route': ['A1_in', 'A2_out']},
                                                6000, np.random.default_rng(0))

    def test_route_budget_uses_both_lanes_and_respects_total_limit(self):
        c = self.controller
        c.meters['A1_in_0'].samples.append(2)
        c.meters['A1_in_1'].samples.append(8)
        c.update_rates(10)
        self.assertAlmostEqual(sum(c.rates), 10 * 3600 / 125 * 1.1)
        self.assertGreater(c.gate_weights[0][1], c.gate_weights[0][0])
        c.meters['A1_in_0'].samples.clear()
        c.meters['A1_in_1'].samples.clear()
        c.update_rates(20)
        self.assertAlmostEqual(sum(c.rates), 6000)

    def test_lane_change_and_teleport_are_not_stop_line_discharge(self):
        c = self.controller
        c.observe({}, {'nC': 'rr'})
        c.observe({'v': ('A1_in_0', 99, 0, 5)}, {'nC': 'GG'})
        c.observe({'v': ('A1_in_1', 99, 0, 5)}, {'nC': 'GG'})
        self.assertEqual(c.meters['A1_in_0'].departures, 0)
        c.observe({'v': (':j_1_0', 2, 3, 5)}, {'nC': 'GG'}, teleported=['v'])
        self.assertEqual(c.meters['A1_in_1'].departures, 0)

    def test_actual_crossing_and_separated_upstream_queue_are_both_counted(self):
        c = self.controller
        c.observe({}, {'nC': 'rr'})
        c.observe({'v': ('A1_in_0', 99, 0, 5), 'far': ('A1_in_0', 30, 0, 5)}, {'nC': 'GG'})
        self.assertAlmostEqual(c.meters['A1_in_0'].queue_ratio, .75)
        c.observe({'v': (':j_0_0', 2, 3, 5)}, {'nC': 'GG'})
        self.assertEqual(c.meters['A1_in_0'].departures, 1)

    def test_pending_route_blocks_new_requests_without_accumulating_due_events(self):
        c = self.controller
        c.route_by_id['old'] = 'route'
        c.next_due = [0]
        api = SimpleNamespace(simulation=SimpleNamespace(getPendingVehicles=lambda: ['old']),
                              vehicle=SimpleNamespace(add=lambda *a, **k: self.fail('unexpected admission')))
        c.spawn_due(api, 10, {})
        self.assertEqual(c.serial, 0)
        self.assertGreater(c.next_due[0], 10)

    def test_other_movements_queue_does_not_train_zero_service_for_this_route(self):
        c = self.controller
        c.route_by_id.update(v='other_route', w='other_route')
        states = {'v': ('A1_in_0', 99, 0, 5), 'w': ('A1_in_0', 91, 0, 5)}
        c.observe({}, {'nC': 'rr'})
        c.observe(states, {'nC': 'GG'})
        c.observe(states, {'nC': 'rr'})
        self.assertGreater(c.meters['A1_in_0'].queue_ratio, 0)
        self.assertEqual(list(c.meters['A1_in_0'].samples), [])

    def test_split_approach_includes_internal_lanes_and_all_upstream_branches(self):
        path = Path(self.directory.name) / 'split.xml'
        path.write_text('''<net>
          <edge id="A2_in"><lane id="A2_in_4" length="20"/></edge>
          <edge id="A2_in1"><lane id="A2_in1_3" length="100"/><lane id="A2_in1_2" length="100"/></edge>
          <edge id=":minor"><lane id=":minor_0" length="10"/></edge>
          <edge id=":branch"><lane id=":branch_0" length="15"/></edge>
          <tlLogic id="nC"><phase duration="20" state="GG"/><phase duration="105" state="rr"/></tlLogic>
          <connection from="A2_in1" fromLane="3" to="A2_in" toLane="4" via=":minor_0"/>
          <connection from=":minor" fromLane="0" to="A2_in" toLane="4"/>
          <connection from="A2_in1" fromLane="2" to="A2_in" toLane="4" via=":branch_0"/>
          <connection from=":branch" fromLane="0" to="A2_in" toLane="4"/>
          <connection from="A2_in" fromLane="4" to="A3_out" toLane="0" tl="nC" linkIndex="0"/>
          <connection from="A2_in1" fromLane="3" to="A2_out1" toLane="0" tl="nC" linkIndex="1"/>
        </net>''', encoding='utf-8')
        c = LaneInflowController(path, [], {}, 6000, np.random.default_rng(0))
        segments = {part[0] for part in c.corridors['A2_in_4']}
        self.assertEqual(segments, {'A2_in_4', 'A2_in1_3', 'A2_in1_2', ':minor_0', ':branch_0'})
        self.assertEqual(c.corridors['A2_in1_3'], c.corridors['A2_in_4'])
        c.observe({'v': (':minor_0', 5, 0, 5)}, {'nC': 'rr'})
        self.assertEqual(c.meters['A2_in_4'].approach_length_m, 135)
        self.assertEqual(c.meters['A2_in_4'].queue_length_m, 30)
        c.observe({'v': ('A2_in1_3', 50, 0, 5)}, {'nC': 'rr'})
        self.assertEqual(c.meters['A2_in_4'].queue_length_m, 85)
        c.update_rates(10)
        self.assertEqual(c.meters['A2_in_4'].control_queue_ratio,
                         c.meters['A2_in1_3'].control_queue_ratio)
        flows = [TrafficFlow('left', 'car', 'random', '8', .85),
                 TrafficFlow('turn', 'car', 'random', '8', .15)]
        c = LaneInflowController(path, flows,
                                {'left': ['A2_in1', 'A2_in', 'A3_out'],
                                 'turn': ['A2_in1', 'A2_out1']},
                                6000, np.random.default_rng(0))
        self.assertAlmostEqual(sum(m.bootstrap_cap_vph for m in c.meters.values()), 230.4)
        self.assertAlmostEqual(sum(c.rates), 230.4 * 1.1)
        left = c.meters['A2_in_4']
        left.samples.append(4)
        left.draining = True
        left.residual_queued = 10
        c.update_rates(10)
        self.assertEqual(sum(c.rates), 0)
        self.assertEqual(c.meters['A2_in1_3'].shared_drain_factor, 0)

    def test_green_opening_does_not_erase_last_cycle_queue_peak(self):
        meter = LaneMeter('lane', 125)
        meter.observe(False, 0, 5, .8)
        meter.observe(True, 0, 2, .2)
        meter.observe(False, 2, 4, .7)
        meter.observe(True, 0, 0, 0)
        self.assertEqual(meter.last_cycle_queue_peak, .7)


if __name__ == '__main__':
    unittest.main()
