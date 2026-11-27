"""Compare signal plans for the central junction (nC) with background traffic only.

The road file is not changed: each plan is written as a separate SUMO additional
file holding a new program for traffic light nC, and SUMO uses the program loaded
last. The same calibrated demand is then simulated under every plan.

    python scripts/compare_signal_plans.py analysis/yj_bc_suite_4740/sumo

A plan is a list of stages. A stage names the lanes that get green, as
(incoming edge, lane indices); None means every lane of that edge. Lanes that are
green in two consecutive stages stay green through the yellow and all-red between
them. U-turns only get a permissive green (they yield). The programs come from
env/signal_plan.py, the same code road_builder uses for the road itself.
"""
import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env import signal_plan

TLS = "nC"
YELLOW, ALL_RED = 3, 2
# U-turns made at the node before the junction, controlled by the same program,
# and the junction exit that feeds the road each one turns into.
UTURN_EXITS = {"A2_in1": "A2_out", "A5_in1": "A5_out"}

# Clock positions: A1 = 8, A2 = 2, A3 = 5, A4 = 10, A5 = 3 o'clock.
SEQUENTIAL = [
    ("A1", [("A1_in", None)]),
    ("A2", [("A2_in", None), ("A2_in1", None)]),
    ("A3", [("A3_in", None)]),
    ("A4", [("A4_in", None)]),
    ("A5", [("A5_in", None), ("A5_in1", None)]),
]
# Nohyeong-ro (A1/A2) runs both ways at once, then its left turns together.
# The right-turn lanes of A3 and A1 keep green wherever nothing crosses them.
A3_RIGHT, A1_RIGHT = ("A3_in", [0]), ("A1_in", [0])
GROUPED_ORIGINAL = [
    ("A1A2_through", [("A1_in", [0, 1, 2, 3]), ("A2_in", [1, 2, 3]), A3_RIGHT]),
    ("A1A2_left", [("A1_in", [4]), ("A2_in", [4]), ("A2_in1", None), A3_RIGHT, A1_RIGHT]),
    ("A3", [("A3_in", None)]),
    ("A4", [("A4_in", None)]),
    ("A5", [("A5_in", None), ("A5_in1", None), A1_RIGHT]),
]
# A5 separates the slow A3 U-turn from conflicting A4 traffic.
GROUPED = [GROUPED_ORIGINAL[i] for i in (0, 1, 2, 4, 3)]

PLANS = {
    "current": (SEQUENTIAL, [20, 20, 20, 20, 20]),
    "current_weighted": (SEQUENTIAL, [25, 17, 23, 19, 16]),
    "grouped_original": (GROUPED_ORIGINAL, [20, 10, 20, 20, 20]),
    "grouped_10s": (GROUPED, [20, 10, 20, 20, 20]),
    "grouped_15s": (GROUPED, [20, 15, 20, 20, 20]),
    "grouped": (GROUPED, [20, 17, 18, 20, 20]),
    "grouped_weighted": (GROUPED, [23, 12, 21, 16, 18]),
    "current_short": (SEQUENTIAL, [15, 15, 15, 15, 15]),
    "grouped_short": (GROUPED, [15, 8, 15, 15, 15]),
}


def plan_program(links, name):
    """Checked SUMO tlLogic of the named plan."""
    stages, greens = PLANS[name]
    timed = [(stage, seconds, lanes) for (stage, lanes), seconds in zip(stages, greens, strict=True)]
    logic, sets = signal_plan.program(links, timed, YELLOW, ALL_RED, UTURN_EXITS, TLS, "compare")
    signal_plan.check(links, timed, sets, UTURN_EXITS)
    return logic


def simulate(sumo_dir, plan_file, seed, seconds, out):
    cfg = sumo_dir / "highway.sumocfg"
    root = ET.parse(cfg).getroot()
    node = root.find("input/additional-files")
    additional = [str(sumo_dir / name) for name in (node.get("value").split(",") if node is not None else [])]
    stats = out / f"{plan_file.stem}_{seed}.xml"
    subprocess.run(
        ["sumo", "-c", str(cfg), "--additional-files", ",".join(additional + [str(plan_file)]),
         "--step-length", "0.5", "--seed", str(seed), "--end", str(seconds),
         "--max-depart-delay", "125", "--collision.action", "warn", "--collision.check-junctions", "true",
         "--no-step-log", "true", "--no-warnings", "true", "--duration-log.statistics", "true",
         "--collision-output", str(out / f"{plan_file.stem}_{seed}.collisions.xml"),
         "--statistic-output", str(stats)], check=True, capture_output=True)
    s = ET.parse(stats).getroot()
    trips = s.find("vehicleTripStatistics")
    return dict(inserted=int(s.find("vehicles").get("inserted")), arrived=int(trips.get("count")),
                wait=float(trips.get("waitingTime")), loss=float(trips.get("timeLoss")),
                collisions=int(s.find("safety").get("collisions")),
                teleports=int(s.find("teleports").get("total")))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sumo_dir", type=Path, help="Folder with highway.sumocfg, network and background flows")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--keep", type=Path, help="Folder to keep the plan files in (default: temporary)")
    parser.add_argument("--plans", nargs="+", choices=PLANS, default=list(PLANS))
    args = parser.parse_args()
    sumo_dir = args.sumo_dir.resolve(strict=True)
    links = signal_plan.junction_links(ET.parse(sumo_dir / "highway.net.xml").getroot(), TLS)

    with tempfile.TemporaryDirectory(prefix="signal-plans-") as work:
        out = args.keep.resolve() if args.keep else Path(work)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in args.plans:
            stages, greens = PLANS[name]
            logic = plan_program(links, name)
            ET.indent(logic, space="    ")
            wrapper = ET.Element("additional")
            wrapper.append(logic)
            files[name] = out / f"{name}.add.xml"
            ET.ElementTree(wrapper).write(files[name], encoding="utf-8")
            cycle = sum(greens) + len(greens) * (YELLOW + ALL_RED)
            print(f"{name}: green {greens}, cycle {cycle} s")

        jobs = [(name, seed) for name in args.plans for seed in args.seeds]
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda job: simulate(sumo_dir, files[job[0]], job[1], args.seconds, out), jobs))
        (out / "results.json").write_text(json.dumps([dict(plan=name, seed=seed, **result)
            for (name, seed), result in zip(jobs, results)], indent=2) + "\n")
        by_plan = defaultdict(list)
        for (name, _), result in zip(jobs, results):
            by_plan[name].append(result)
        print(f"\nmean of {len(args.seeds)} seeds, {args.seconds} s each")
        print(f"{'plan':18s} {'inserted':>9s} {'arrived':>8s} {'wait s':>7s} {'loss s':>7s} {'collisions':>10s} {'teleports':>9s}")
        for name, rows in by_plan.items():
            mean = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
            print(f"{name:18s} {mean['inserted']:9.0f} {mean['arrived']:8.0f} {mean['wait']:7.1f} "
                  f"{mean['loss']:7.1f} {mean['collisions']:10.1f} {mean['teleports']:9.1f}")


if __name__ == "__main__":
    main()
