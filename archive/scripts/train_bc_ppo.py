"""Fine-tune a BC checkpoint using the calibrated traffic and event reward."""

import argparse
from datetime import datetime
import json
from math import isfinite
from pathlib import Path
import shutil
from time import perf_counter

import torch

from algorithms.parallel_ppo import ParallelPPO
from env.bc_ppo_reward import STEP_REWARD, STOP_AMPLITUDE, STOP_SIGMA_M, TERMINAL_REWARDS
from env.parallel_sumo import ParallelSumo, ShardedParallelSumo
from env.parallel_traffic import TrafficCurriculum
from test import load_saved_configs
from utils.logger import RunLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("traffic_suite", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--early-lane-change-bonus", type=float, default=0.,
                        help="Reward per newly saved lane change toward a branch")
    parser.add_argument("--early-lane-change-distance-m", type=float, default=120.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rollout = args.envs * args.steps_per_env
    if min(args.iterations, args.workers, args.envs, args.steps_per_env) < 1:
        parser.error("Iteration, worker, environment and step counts must be positive")
    if args.envs % args.workers or rollout % 1024:
        parser.error("Split environments evenly across workers and use a rollout divisible by 1024")
    if (not isfinite(args.early_lane_change_bonus) or args.early_lane_change_bonus < 0
            or not isfinite(args.early_lane_change_distance_m) or args.early_lane_change_distance_m <= 0):
        parser.error("Use a finite nonnegative lane change bonus and a positive finite distance")

    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve(strict=True)
    suite = args.traffic_suite.resolve(strict=True)
    spec = json.loads((suite / "suite.json").read_text())
    road, _, _ = load_saved_configs(str(suite / "reference/model.pt"))
    _, mdp, training = load_saved_configs(str(checkpoint))
    action = dict(mdp["ACTION"])
    action["lane_change_safety"] = dict(action["lane_change_safety"], enabled=False, spawn_protect=0)
    reward = dict(training["REWARD"], bc_ppo_reward=True,
                  early_lane_change_bonus=args.early_lane_change_bonus,
                  early_lane_change_distance_m=args.early_lane_change_distance_m)
    hparams = dict(training["HPARAMS"], n_steps=rollout, n_epochs=10, minibatch_size=1024,
                   lr=1e-4, seed=0, target_kl=None)
    output = (args.output or Path("results") / datetime.now().strftime("PPO_%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True)
    for name in ("road_config.py", "mdp_config.py", "train.py"):
        source = suite / "reference" if name == "road_config.py" else checkpoint.parent
        shutil.copy2(source / name, output / name)
    with (output / "mdp_config.py").open("a") as file:
        file.write(f"\nACTION = {action!r}\n")
    with (output / "train.py").open("a") as file:
        file.write(f"\nREWARD = {reward!r}\nHPARAMS = {hparams!r}\n")

    template = ParallelSumo(
        args.envs // args.workers,
        dict(cfg_path=str(suite / "sumo/highway.sumocfg"), road=road["ROAD"], ego=road["EGO"],
             traffic=road["TRAFFIC"], mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
             action=action, reward=reward),
        output / "parallel", seed=0, randomize_signals=True, randomize_departures=True,
        uniform_route_departures=True, traffic_curriculum=TrafficCurriculum(**spec["settings"]),
        fixed_traffic_vph=spec["traffic_vph"],
    )
    env = template if args.workers == 1 else ShardedParallelSumo(template, args.workers)
    logger = RunLogger(str(output))
    try:
        template._find_departures()
        departures = len({(route, lane) for group in template.route_groups for route in group
                          for _, _, lane in template.departures[route]})
        agent = ParallelPPO(obs_dim=template.prototype.obs_dim, act_dim=template.prototype.act_dim, **hparams)
        agent.load(str(checkpoint))
        metadata = dict(initial_model=str(checkpoint), traffic_suite=str(suite),
                        traffic_vph=spec["traffic_vph"], departures=departures, rollout=rollout,
                        num_envs=args.envs, num_workers=args.workers, iterations=args.iterations,
                        device=str(agent.device), step_reward=STEP_REWARD, terminal_rewards=TERMINAL_REWARDS,
                        stop_amplitude=STOP_AMPLITUDE, stop_sigma_m=STOP_SIGMA_M,
                        early_lane_change_bonus=args.early_lane_change_bonus,
                        early_lane_change_distance_m=args.early_lane_change_distance_m)
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{output} | {departures} departures | {agent.device} | {rollout} steps/iteration", flush=True)
        start = perf_counter()

        def checkpoint_iteration(iteration):
            if iteration % 10 == 0:
                agent.save(str(output / f"model_iter{iteration:03d}.pt"))
                logger.save_csv()
            print(f"Iteration {iteration}/{args.iterations}, elapsed {perf_counter() - start:.1f}s", flush=True)

        agent.learn_parallel(env, rollout * args.iterations, logger=logger, on_iteration=checkpoint_iteration)
        agent.save(str(output / "model.pt"))
        print(f"Training completed in {perf_counter() - start:.1f}s", flush=True)
    finally:
        env.close()
        logger.close()


if __name__ == "__main__":
    main()
