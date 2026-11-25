"""Build a balanced route/departure benchmark with independently generated SUMO traffic."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.parallel_sumo import _signal_schedule, libsumo
from scripts.evaluate_fixed_routes import (
    FixedTrafficSchedule, MAX_REPLICAS_PER_WORKER, STEP, default_workers, prepare_road, read_json, run, write_json,
)

DEFAULT_SEEDS = tuple(range(52101, 52126))
CAPTURE_WINDOWS = ((120, 600), (900, 1800), (2400, 3500))
EXPECTED_COMBINATIONS = 94
WORKERS = 4


def capture_traffic(cfg, flows, lanes, controllers, vph, seed):
    traffic_seed, signal_seed, capture_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(traffic_seed)
    signal_rng = np.random.default_rng(signal_seed)
    capture_rng = np.random.default_rng(capture_seed)
    net = ET.parse(cfg.with_name("highway.net.xml")).getroot()
    signals, phase_ends = _signal_schedule(net, STEP, True)
    cycle_ticks = int(phase_ends[-1])
    offset = int(signal_rng.integers(cycle_ticks))
    phase = int(np.searchsorted(phase_ends, offset, side="right"))
    capture_ticks = [int(capture_rng.integers(round(lo / STEP), round(hi / STEP) + 1))
                     for lo, hi in CAPTURE_WINDOWS]
    schedule = FixedTrafficSchedule(flows, controllers, lanes)
    captures = []
    libsumo.start(["sumo", "-c", str(cfg), "--seed", str(seed), "--step-length", str(STEP),
                   "--collision.action", "remove", "--no-warnings", "true", "--no-step-log", "true"])
    try:
        for signal in signals:
            libsumo.trafficlight.setPhase(signal, phase)
            libsumo.trafficlight.setPhaseDuration(signal, float((phase_ends[phase] - offset) * STEP))
        schedule.reset(libsumo, 0., vph, rng, set())
        vehicle = libsumo.vehicle
        capture_set = set(capture_ticks)
        for tick in range(1, capture_ticks[-1] + 1):
            schedule.spawn_due(libsumo, libsumo.simulation.getTime(), str, rng)
            libsumo.simulationStep()
            schedule.ids.clear()
            if tick not in capture_set:
                continue
            now = libsumo.simulation.getTime()
            phase = libsumo.trafficlight.getPhase(signals[0])
            remaining = round((libsumo.trafficlight.getNextSwitch(signals[0]) - now) / STEP)
            signal_tick = int(phase_ends[phase] - remaining) % cycle_ticks
            vehicles = []
            for vid in sorted(vehicle.getIDList()):
                lane = vehicle.getLaneID(vid)
                if lane:
                    vehicles.append((vehicle.getRouteID(vid), vehicle.getTypeID(vid), lane,
                                     vehicle.getLanePosition(vid), vehicle.getSpeed(vid),
                                     vehicle.getSpeedFactor(vid), vehicle.getLength(vid)))
            captures.append((now, signal_tick, vehicles))
    finally:
        libsumo.close()
    if len(captures) != len(CAPTURE_WINDOWS):
        raise RuntimeError(f"Seed {seed} did not produce all scheduled traffic snapshots")
    return seed, captures


def departure_combinations(source, routes):
    net = ET.parse(source / "sumo/highway.net.xml").getroot()
    route_nodes = ET.parse(source / "sumo/highway.ego.rou.xml").getroot()
    entries = {node.get("id"): node.get("edges").split()[0] for node in route_nodes.findall("route")}
    lanes_by_edge = {
        edge.get("id"): [int(lane.get("index")) for lane in edge.findall("lane")
                         if (not lane.get("allow") or "passenger" in lane.get("allow").split())
                         and not {"passenger", "all"}.intersection(lane.get("disallow", "").split())]
        for edge in net.findall("edge") if edge.get("function") != "internal"
    }
    combinations = [(route, lane) for route in routes for lane in lanes_by_edge[entries[route]]]
    if len(set(combinations)) != EXPECTED_COMBINATIONS or len(combinations) != EXPECTED_COMBINATIONS:
        raise ValueError(f"Expected {EXPECTED_COMBINATIONS} distinct route/lane combinations; got {len(combinations)}")
    return combinations


def prepare(source, suite, seeds):
    spec = read_json(source / "suite.json")
    if spec.get("adaptive_traffic") or spec["traffic_vph"] != 4740:
        raise ValueError("Use the calibrated 4,740 veh/h fixed-demand suite as the road template")
    combinations = departure_combinations(source, spec["routes"])
    center_lanes = {row["route"]: row["start_lane"] for row in read_json(source / "cases.json")}
    suite.mkdir(parents=True)
    shutil.copytree(source / "sumo", suite / "sumo")
    shutil.copytree(source / "reference", suite / "reference")
    cases = []
    states = []
    metadata = []
    with TemporaryDirectory(prefix="diverse-lane-capture-") as scratch:
        cfg, flows, lanes = prepare_road(source, Path(scratch), spec)
        with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(capture_traffic, cfg, flows, lanes, spec["vehicle_types"], spec["traffic_vph"], seed)
                    for seed in seeds]
            for job in jobs:
                seed, captures = job.result()
                for time, signal_tick, vehicles in captures:
                    snapshot_id = len(states) + 1
                    states.append((spec["traffic_vph"], snapshot_id, 0, json.dumps(vehicles, separators=(",", ":"))))
                    metadata.append(dict(id=snapshot_id, source_seed=seed, source_time_s=time,
                                         signal_tick=signal_tick, vehicles=len(vehicles)))
                    for route, lane in combinations:
                        cases.append(dict(case_id=len(cases), worker=(snapshot_id - 1) % WORKERS,
                                          snapshot_tick=snapshot_id, snapshot_time_s=time, tick=signal_tick,
                                          seed=seed, traffic_seed=[seed, 88301, snapshot_id],
                                          route=route, start_lane=lane))
                print(f"Captured {len(states)} states", flush=True)
    coverage = Counter((case["route"], case["start_lane"]) for case in cases)
    if coverage != {combo: len(states) for combo in combinations}:
        raise RuntimeError("Route/lane coverage is not balanced")
    with closing(sqlite3.connect(suite / "traffic_states.sqlite")) as db, db:
        db.execute("CREATE TABLE states (vph INTEGER, tick INTEGER, slot INTEGER, vehicles TEXT, "
                   "PRIMARY KEY (vph, tick, slot)) WITHOUT ROWID")
        db.executemany("INSERT INTO states VALUES (?, ?, ?, ?)", states)
    spec = dict(spec, all_start_lanes=True, center_lanes=center_lanes, snapshot_seeds=list(seeds),
                snapshot_capture_windows_s=CAPTURE_WINDOWS, snapshot_count=len(states),
                route_lane_combinations=len(combinations), snapshot_source=str(source), spawn_protect_m=0)
    for obsolete in ("snapshot_times_seconds", "warmup_seconds"):
        spec.pop(obsolete, None)
    write_json(suite / "suite.json", spec)
    write_json(suite / "cases.json", cases)
    write_json(suite / "snapshots.json", metadata)
    print(f"Prepared {len(cases):,} cases for {len(combinations)} combinations in {suite}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("prepare")
    builder.add_argument("source", type=Path)
    builder.add_argument("suite", type=Path)
    builder.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("suite", type=Path)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--workers", type=int, default=default_workers(),
                          help="Processes; each case runs in its own fresh SUMO (default: up to 24)")
    evaluate.add_argument("--sumo-seed", type=int, default=0,
                          help="Base SUMO seed; each case derives its own from it")
    evaluate.add_argument("--shared-sumo", action="store_true",
                          help="Previous mode: 4 workers run --replicas cases at once in one SUMO. "
                               "Outcomes then depend on case order and batching.")
    evaluate.add_argument("--replicas", type=int, default=MAX_REPLICAS_PER_WORKER,
                          help="Cases per SUMO with --shared-sumo")
    evaluate.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or min(args.seeds) < 0:
            parser.error("Supply at least two distinct nonnegative snapshot seeds")
        prepare(args.source.resolve(strict=True), args.suite.resolve(), args.seeds)
    else:
        if not 1 <= args.replicas <= MAX_REPLICAS_PER_WORKER or args.smoke < 0:
            parser.error("Invalid replica or smoke case count")
        if args.workers < 1 or args.sumo_seed < 0:
            parser.error("--workers must be positive and --sumo-seed nonnegative")
        if args.shared_sumo and args.sumo_seed:
            parser.error("--sumo-seed applies only to the default isolated mode")
        suite = args.suite.resolve(strict=True)
        if not read_json(suite / "suite.json").get("all_start_lanes"):
            parser.error("Prepare a diverse start-lane suite before evaluation")
        model = args.model.resolve(strict=True)
        output = args.output.resolve() if args.output else None
        run(suite, "diverse_start_lanes", None, model, args.replicas, args.smoke, output=output,
            isolated=not args.shared_sumo, workers=args.workers, sumo_seed=args.sumo_seed)


if __name__ == "__main__":
    main()
