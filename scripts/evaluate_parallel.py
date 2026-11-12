"""Evaluate a train_parallel.py model on many parallel roads (deterministic policy).

Usage (project root):
    python scripts/evaluate_parallel.py results/parallel_2_.../  [--model model_iter0015.pt]

Reads the run's parallel_config.json so the phase and every --set override match training.
Besides the usual rates it reports how the ego behaves at red lights:
    stop_distance : distance to the stop line while waiting (speed < 0.5 m/s) at a red light
    restart_delay : seconds from the light turning green to the ego moving again (> 2 m/s)
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.parallel_sumo import ParallelSumo  # noqa: E402
from env.road_config import NOHYEONG  # noqa: E402
from train import (ACTION, EGO, HPARAMS, OBSERVATION, REWARD, ROAD, SIMULATION,  # noqa: E402
                   TRAFFIC, road_builder)
from utils.networks import HybridActorCritic  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--signal-green", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="도로마다 초록불 길이(초)를 이 범위에서 무작위로 정해 평가 (학습 때 못 본 신호 길이 시험용)")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "parallel_config.json").read_text())
    NOHYEONG["all_signals_green"] = config["phase"] == 1
    groups = {"reward": REWARD, "obs": OBSERVATION, "hparams": HPARAMS}
    for target, value in config.get("overrides", {}).items():
        group, key = target.split(".", 1)
        groups[group][key] = value

    tag = f"_green{args.signal_green[0]:g}-{args.signal_green[1]:g}" if args.signal_green else ""
    work = args.run_dir / f"eval_{args.seed}{tag}"
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, str(work / "sumo"))
    env = ParallelSumo(args.num_envs, dict(
        cfg_path=sumocfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD, traffic=TRAFFIC), work / "parallel", args.seed,
        green_range=args.signal_green)
    policy = HybridActorCritic(env.prototype.obs_dim, HPARAMS["hidden_sizes"], HPARAMS["activation"])
    policy.load_state_dict(torch.load(args.run_dir / args.model, map_location="cpu"))
    policy.eval()

    step = env.prototype.step_length
    outcomes, returns, lengths = Counter(), np.zeros(args.num_envs), np.zeros(args.num_envs, dtype=int)
    stop_distances, restart_delays = [], []
    by_route = {}
    waiting_since_green = {}          # env index -> steps since the light it waited for turned green
    was_waiting = set()               # env indices currently stopped at a red light
    active = list(range(args.num_envs))
    obs = env.reset()
    try:
        with torch.inference_mode():
            while active:
                actions = policy.det_action(torch.as_tensor(obs, dtype=torch.float32)).numpy()
                pending, next_obs = [], []
                for index, (o, reward, terminated, truncated, info) in zip(
                        active, env.step_evaluation(active, actions), strict=True):
                    returns[index] += reward
                    lengths[index] += 1
                    speed = info.get("speed")
                    if speed is not None:
                        if info["signal_red_visible"] and speed < 0.5:
                            stop_distances.append(info["red_signal_distance"])
                            was_waiting.add(index)
                            waiting_since_green.pop(index, None)
                        elif index in was_waiting and not info["signal_red_visible"]:
                            waiting_since_green[index] = waiting_since_green.get(index, 0) + 1
                            if speed > 2.0:
                                restart_delays.append(waiting_since_green.pop(index) * step)
                                was_waiting.discard(index)
                    if terminated or truncated:
                        outcome = next((name for name in ("red_light_violation", "collided", "arrived", "wrong_lane")
                                        if info.get(name)), "timeout")
                        outcomes[outcome] += 1
                        by_route.setdefault(info["route"].split("_", 1)[1], Counter())[outcome] += 1
                    else:
                        pending.append(index)
                        next_obs.append(o)
                active = pending
                if active:
                    obs = np.stack(next_obs)
    finally:
        env.close()

    result = {name: outcomes[name] / args.num_envs
              for name in ("arrived", "red_light_violation", "wrong_lane", "collided", "timeout")}
    result.update(
        episodes=args.num_envs, mean_return=float(returns.mean()),
        mean_seconds=float(lengths.mean() * step),
        never_restarted=len(waiting_since_green) / args.num_envs,
        stop_distance_median=float(np.median(stop_distances)) if stop_distances else None,
        stop_distance_p90=float(np.percentile(stop_distances, 90)) if stop_distances else None,
        restart_delay_median=float(np.median(restart_delays)) if restart_delays else None,
    )
    print(json.dumps(result, indent=2))
    for route in sorted(by_route):
        print(f"  {route:>6}: " + ", ".join(f"{name} {count}" for name, count in by_route[route].most_common()))
    (args.run_dir / f"eval_{Path(args.model).stem}{tag}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
