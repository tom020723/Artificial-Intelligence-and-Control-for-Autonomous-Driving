"""중앙 교차로 신호(차선별 동시신호 / 예전 순차 신호)를 실제 생성 도로로 검사한다."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from env import signal_plan
from env.road_builder import build
from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG
from scripts import compare_signal_plans


def build_net(change=None, all_green=False):
    """Build the road in a temporary folder and return the parsed network."""
    config = copy.deepcopy(NOHYEONG)
    config["all_signals_green"] = all_green
    if change:
        change(config["signal_plan"])
    with tempfile.TemporaryDirectory(prefix="sumo-signal-plan-") as folder:
        with patch("env.road_builder.NOHYEONG", config):
            build(ROAD, EGO, TRAFFIC, folder)
        return ET.parse(Path(folder) / "highway.net.xml").getroot()


def phases(net):
    return [(p.get("name"), p.get("duration"), p.get("state")) for p in net.find("tlLogic[@id='nC']")]


def green_stages(net):
    """{(incoming edge, lane): names of the stages whose green phase turns it green}."""
    result = {}
    for c in net.findall("connection"):
        if c.get("tl") == "nC":
            result.setdefault((c.get("from"), int(c.get("fromLane"))), set()).update(
                name.removesuffix("_green") for name, _, state in phases(net)
                if name.endswith("_green") and state[int(c.get("linkIndex"))] in "Gg")
    return result


class GroupedSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = build_net()
        cls.links = signal_plan.junction_links(cls.net, "nC")

    def state(self, phase, edge, lane):
        states = dict((name, state) for name, _, state in phases(self.net))
        return {states[phase][i] for i, c in self.links.items()
                if c["from"] == edge and int(c["fromLane"]) == lane}

    def test_default_road_gets_the_five_grouped_stages(self):
        names = ["A1A2_through", "A1A2_left", "A3", "A5", "A4"]
        self.assertEqual([name for name, _, _ in phases(self.net)],
                         [f"{name}_{part}" for name in names for part in ("green", "yellow", "clearance")])
        self.assertEqual([float(seconds) for _, seconds, _ in phases(self.net)],
                         [20, 3, 2, 17, 3, 2, 18, 3, 2, 20, 3, 2, 20, 3, 2])
        self.assertEqual(len(self.links), 28)

    def test_each_lane_is_green_in_its_stages(self):
        through, left = {"A1A2_through"}, {"A1A2_left"}
        expected = {("A1_in", 0): through | left | {"A5"}, ("A1_in", 4): left,
                    ("A2_in", 4): left, ("A2_in1", 3): left,
                    ("A3_in", 0): through | left | {"A3"}, ("A5_in1", 2): {"A5"}}
        expected.update({("A1_in", lane): through for lane in (1, 2, 3)})
        expected.update({("A2_in", lane): through for lane in (1, 2, 3)})
        expected.update({("A3_in", lane): {"A3"} for lane in (1, 2, 3, 4)})
        expected.update({("A4_in", lane): {"A4"} for lane in (0, 1, 2)})
        expected.update({("A5_in", lane): {"A5"} for lane in (0, 1, 2)})
        self.assertEqual(green_stages(self.net), expected)

    def test_lane_green_in_consecutive_stages_stays_green_between_them(self):
        for phase in ("A1A2_through_yellow", "A1A2_through_clearance",
                      "A1A2_left_yellow", "A1A2_left_clearance"):
            self.assertEqual(self.state(phase, "A3_in", 0), {"G"}, phase)
        for phase in ("A1A2_through_yellow", "A1A2_through_clearance"):
            self.assertEqual(self.state(phase, "A1_in", 0), {"G"}, phase)
        # ... and ends with yellow then red once the next stage does not include it.
        self.assertEqual(self.state("A5_yellow", "A1_in", 0), {"y"})
        self.assertEqual(self.state("A5_clearance", "A1_in", 0), {"r"})
        self.assertEqual(self.state("A1A2_left_yellow", "A1_in", 0), {"y"})
        self.assertEqual(self.state("A1A2_left_clearance", "A1_in", 0), {"r"})
        self.assertEqual(self.state("A3_yellow", "A3_in", 0), {"y"})
        self.assertEqual(self.state("A1A2_through_yellow", "A1_in", 2), {"y"})
        self.assertEqual(self.state("A1A2_through_clearance", "A1_in", 2), {"r"})

    def test_uturns_only_get_a_yielding_green(self):
        self.assertEqual(self.state("A1A2_left_green", "A2_in1", 3), {"g"})
        self.assertEqual(self.state("A5_green", "A5_in1", 2), {"g"})
        turns = [i for i, c in self.links.items() if c["dir"] == "t"]
        self.assertTrue(turns)
        self.assertFalse(any(state[i] == "G" for _, _, state in phases(self.net) for i in turns))

    def test_program_equals_the_grouped_plan_of_the_comparison_script(self):
        script = compare_signal_plans.plan_program(self.links, "grouped")
        self.assertEqual([(p.get("name"), p.get("duration"), p.get("state")) for p in script],
                         phases(self.net))

    def test_all_signals_green_still_replaces_the_program(self):
        self.assertEqual([(seconds, state) for _, seconds, state in phases(build_net(all_green=True))],
                         [("100000", "G" * 28)])


class SequentialSignalTests(unittest.TestCase):
    def check_one_approach_at_a_time(self, net):
        arms = ["A1", "A2", "A3", "A4", "A5"]
        self.assertEqual([name for name, _, _ in phases(net)],
                         [f"{arm}_{part}" for arm in arms for part in ("green", "yellow", "clearance")])
        self.assertEqual(sum(float(seconds) for _, seconds, _ in phases(net)), 125)
        for (edge, _), stages in green_stages(net).items():
            self.assertEqual(stages, {edge[:2]}, edge)
        for name, _, state in phases(net):
            if not name.endswith("_green"):
                self.assertNotIn("G", state)
                self.assertNotIn("g", state)

    def test_sequential_plan_can_still_be_selected(self):
        net = build_net(lambda plan: plan.update(type="sequential"))
        self.check_one_approach_at_a_time(net)
        links = signal_plan.junction_links(net, "nC")
        script = compare_signal_plans.plan_program(links, "current")
        self.assertEqual([p.get("state") for p in script], [state for _, _, state in phases(net)])

    def test_plan_saved_before_the_type_key_is_sequential(self):
        def old_format(plan):
            del plan["type"], plan["stages"]
        self.check_one_approach_at_a_time(build_net(old_format))


class SignalConflictTests(unittest.TestCase):
    def test_crossing_protected_greens_fail_the_build(self):
        def cross(plan):   # A3 through traffic crosses the Nohyeong-ro through traffic
            plan["stages"][0]["lanes"]["A3"] = "all"
        with self.assertRaisesRegex(ValueError, "Signal stage A1A2_through: .* conflicts with "):
            build_net(cross)

    def test_second_stream_into_the_same_lane_yields_instead_of_conflicting(self):
        def link(edge, y):
            return {"from": edge, "fromLane": "0", "to": "out", "toLane": "0", "dir": "s",
                    "path": [(0., y), (10., 0.)]}
        links = {0: link("first", 1.), 1: link("second", -1.)}
        self.assertEqual(signal_plan.green_state(links, {0, 1}, {}), ["G", "g"])
        self.assertEqual(signal_plan.conflicts(links, {0, 1}, {}), [])

    def test_upstream_uturn_with_traffic_into_its_exit_fails_the_build(self):
        def feed(plan):    # A1 through traffic enters A2_out, where the A2 U-turn merges
            plan["stages"][0]["uturns"] = ["A2"]
        with self.assertRaisesRegex(ValueError, "Signal stage A1A2_through: A2_in1_3>A2_out1 conflicts with "):
            build_net(feed)

    def test_lane_that_never_gets_green_fails_the_build(self):
        def drop(plan):
            plan["stages"][1]["lanes"]["A2"] = []
        with self.assertRaisesRegex(ValueError, "No signal link for A2_in"):
            build_net(drop)

        def forget(plan):
            plan["stages"] = [stage for stage in plan["stages"] if stage["name"] != "A4"]
        with self.assertRaisesRegex(ValueError, "A4_in_0>.* is never green"):
            build_net(forget)

    def test_unknown_plan_type_fails_the_build(self):
        with self.assertRaisesRegex(ValueError, "Unknown signal_plan.type"):
            build_net(lambda plan: plan.update(type="actuated"))


class SignalOffsetTests(unittest.TestCase):
    def test_offsets_follow_generated_cycles(self):
        from scripts.evaluate_quick_phase1 import signal_offsets
        for kind, through_seconds, left_seconds, a3_seconds, cycle, count in (
                ("grouped", 20, 17, 18, 240, 48),
                ("grouped", 15, 20, 20, 240, 48), ("grouped", 20, 15, 20, 240, 48),
                ("grouped", 20, 10, 20, 230, 46), ("sequential", 20, 20, 20, 250, 50)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def change(plan):
                    plan["type"] = kind
                    next(stage for stage in plan["stages"] if stage["name"] == "A1A2_through")["seconds"] = through_seconds
                    next(stage for stage in plan["stages"] if stage["name"] == "A1A2_left")["seconds"] = left_seconds
                    next(stage for stage in plan["stages"] if stage["name"] == "A3")["seconds"] = a3_seconds
                net = build_net(change)
                ET.ElementTree(net).write(root / "road.net.xml")
                config = root / "road.sumocfg"
                config.write_text('<configuration><input><net-file value="road.net.xml"/></input></configuration>')
                actual_cycle, ticks = signal_offsets(config)
                self.assertEqual(actual_cycle, cycle)
                self.assertEqual(len(ticks), count)
                self.assertEqual(ticks[0], 2)
                self.assertLess(ticks[-1], cycle)

    def test_dagger_samples_stay_inside_each_cycle_and_exclude_eval_offsets(self):
        from scripts.run_bc51_dagger import COMBOS, plan_round
        combos = [(f"route_{i}", 0) for i in range(COMBOS)]
        for cycle in (230, 240, 250):
            cases, _ = plan_round(combos, [], 0, 1, cycle_ticks=cycle)
            ticks = {case["tick"] for case in cases}
            self.assertTrue(all(0 <= tick < cycle for tick in ticks))
            self.assertFalse(ticks.intersection(range(2, cycle, 5)))

    def test_a3_uturn_has_an_intervening_stage_before_a4(self):
        net = build_net()
        times, clock = {}, 0
        for name, seconds, _ in phases(net):
            times[name] = clock
            clock += float(seconds)
        # A3 last permitted entry (end of yellow) to A4 green: 2 + 20 + 3 + 2 s.
        self.assertEqual(times["A4_green"] - times["A3_clearance"], 27)
        self.assertEqual(clock, 120)


if __name__ == "__main__":
    unittest.main()
