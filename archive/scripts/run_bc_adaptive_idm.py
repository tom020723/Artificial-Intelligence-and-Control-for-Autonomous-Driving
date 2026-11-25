"""Collect uniform-route IDM demonstrations, fit a fresh BC actor, and evaluate it."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_background_data import BackgroundObserver, collect_episode, save_json
from env.adaptive_traffic_schedule import AdaptiveTrafficSchedule
from env.parallel_traffic import TrafficCurriculum, parse_traffic_flows
from env.uniform_idm import separate_idm_type
from scripts.evaluate_fixed_routes import run
from test import load_saved_configs


def prepare_collection(dataset, suite, curriculum, reference_vph):
    shutil.copytree(suite / "sumo", dataset / "sumo")
    config = dataset / "sumo/highway.sumocfg"
    route_file = config.with_name("highway.rou.xml")
    tree = ET.parse(route_file)
    expert = separate_idm_type(tree)
    expert.set("lcSpeedGain", "1.0")
    expert.set("lcKeepRight", "0")
    for node in tree.findall("./vTypeDistribution[@id='carDist']/vType"):
        node.set("lcSpeedGain", "0.0")
        node.set("lcKeepRight", "0")
    root = tree.getroot()
    routes = {r.get("id"): r.get("edges").split() for r in root.findall("route")}
    existing = set(routes)
    nodes = root.findall("flow")
    net_path = config.with_name("highway.net.xml")
    flows = parse_traffic_flows(nodes, routes, net_path, reference_vph, curriculum)
    for node in nodes:
        root.remove(node)
    for name in routes.keys() - existing:
        ET.SubElement(root, "route", id=name, edges=" ".join(routes[name]))
    tree.write(route_file, encoding="utf-8", xml_declaration=True)
    lights = [node.get("id") for node in ET.parse(net_path).findall("tlLogic")]
    return config, flows, routes, lights


def initialize_worker(config, flows, routes, lights, reference_model, observation, args, dataset):
    global worker
    road, mdp, train = load_saved_configs(str(reference_model))
    env = BackgroundObserver(config, road["ROAD"], road["EGO"], mdp["SIMULATION"],
                             observation, mdp["ACTION"], train["REWARD"],
                             traffic=road["TRAFFIC"])
    schedule = AdaptiveTrafficSchedule(flows, config.with_name("highway.net.xml"), routes,
                                      lights, args.uniform_idm_vph)
    worker = env, schedule, args, dataset


def collect_run(index):
    env, schedule, args, dataset = worker
    return collect_episode(env, index, args.seed + index, args, dataset, traffic_schedule=schedule)


def collect(args, dataset, spec, settings):
    road, mdp, _ = settings
    dataset.mkdir()
    curriculum = TrafficCurriculum(**spec["settings"])
    config, flows, routes, lights = prepare_collection(
        dataset, args.eval_suite, curriculum, road["TRAFFIC"]["vehs_per_hour"])
    metadata = dict(observation=mdp["OBSERVATION"], simulation=mdp["SIMULATION"],
                    traffic=deepcopy(road["TRAFFIC"]), action=mdp["ACTION"],
                    collision_detection=spec["collision_detection"], warmup_seconds=0,
                    seconds_per_run=args.seconds, workers=args.workers, replicas_per_worker=1,
                    traffic_vph=args.traffic_vph, uniform_idm_vph=args.uniform_idm_vph,
                    adaptive_vph=args.traffic_vph - args.uniform_idm_vph, target_samples=args.target_samples,
                    seed=args.seed, expert="native SUMO IDM, lcSpeedGain=1.0",
                    label="pre-step policy observation paired with realized SUMO action",
                    train_validation_split="whole simulation runs")
    save_json(dataset / "config.json", metadata)
    index = samples = 0
    pairs, entries, requests = Counter(), Counter(), Counter()
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(config, flows, routes, lights, args.eval_suite / "reference/model.pt",
                                       metadata["observation"], args, dataset)) as pool:
        while index < args.workers or samples < args.target_samples:
            if shutil.disk_usage(dataset).free < args.min_free_gb * 1024**3:
                raise RuntimeError("Insufficient free disk space to start another collection wave")
            for report in pool.map(collect_run, range(index, index + args.workers)):
                samples += report["counts"]["samples"]
                pairs.update(report["routes"])
                entries.update(report["idm_entries_by_route"])
                requests.update(report["idm_requests_by_route"])
            index += args.workers
            summary = dict(samples=samples, runs=index, pairs_by_route=dict(pairs),
                           idm_entries_by_route=dict(entries), idm_requests_by_route=dict(requests),
                           collection_wall_seconds=time.monotonic() - started)
            save_json(dataset / "summary.json", summary)
            print(f"Collected {samples:,}/{args.target_samples:,} pairs in {index} runs", flush=True)
    with np.load(dataset / "episode_000.npz", allow_pickle=False) as first:
        obs_dim = first["observation"].shape[1]
    save_json(dataset / "manifest.json", summary | dict(
        format="episode_npz_shards", obs_dim=obs_dim, targets_reached=True,
        shards=[f"episode_{ep:03d}.npz" for ep in range(index)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic-vph", type=int, required=True)
    parser.add_argument("--uniform-idm-vph", type=int, required=True)
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results") / datetime.now().strftime("BC_%Y%m%d_%H%M%S"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--target-samples", type=int, default=100_000_000)
    parser.add_argument("--min-free-gb", type=float, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=99000)
    parser.add_argument("--ring-merge-scope", choices=("legacy", "direct"),
                        help="Select which approach lanes populate the ring merge observation")
    parser.add_argument("--eval-smoke", type=int, default=0, help="Evaluate only N cases per worker")
    args = parser.parse_args()
    if (not 0 < args.uniform_idm_vph < args.traffic_vph or args.workers < 2
            or min(args.seconds, args.target_samples, args.epochs) < 1 or args.eval_smoke < 0):
        parser.error("Expected a valid traffic split, at least two workers, and positive collection and training limits")
    args.eval_suite = args.eval_suite.resolve(strict=True)
    args.out_dir = args.out_dir.resolve()
    spec = json.loads((args.eval_suite / "suite.json").read_text())
    if not spec.get("adaptive_traffic") or spec.get("uniform_idm_vph"):
        parser.error("Evaluation requires an all-adaptive suite")
    road, mdp, train = load_saved_configs(str(args.eval_suite / "reference/model.pt"))
    if args.ring_merge_scope is not None:
        mdp["OBSERVATION"] = dict(mdp["OBSERVATION"], ring_merge_scope=args.ring_merge_scope)
    args.out_dir.mkdir(parents=True)
    dataset = args.out_dir / "dataset"
    # Collection uses native IDM decisions and begins with a completely empty road.
    args.include_vehicle_type = "car_idm"
    args.spawn_protect_m = None
    args.warmup_seconds = 0
    collect(args, dataset, spec, (road, mdp, train))
    model = args.out_dir / "training"
    subprocess.run([sys.executable, "-u", "scripts/train_bc_shards.py", str(dataset), "--out-dir",
                    str(model), "--epochs", str(args.epochs), "--device", args.device], check=True)
    run(args.eval_suite, args.out_dir.name, None, model / "model.pt", 64, args.eval_smoke,
        output=args.out_dir / "evaluation")


if __name__ == "__main__":
    main()
