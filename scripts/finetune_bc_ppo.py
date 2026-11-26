"""Fine-tune the BC checkpoint with anchored PPO on the event reward.

Run as a module: ``python -m scripts.finetune_bc_ppo CHECKPOINT SUITE``.

Compared with the first PPO (scripts/train_bc_ppo.py, in git history) this trains without the ego
speed cap (the cap is a rule outside the policy, and the previous run learned to
lean on it), and uses ``algorithms.anchored_ppo.AnchoredPPO``: a separate
critic with warm-up, a KL anchor to the BC policy, a small acceleration
exploration std and sharper sampled lane choices. The reward is the event
reward, optionally with a per-step penalty for speeding above the lane limit
(``--overspeed-penalty``, off by default).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch

from algorithms.anchored_ppo import AnchoredPPO
from env.collision_detection import junction_collisions
from env.bc_ppo_reward import OVERSPEED_TOLERANCE, STEP_REWARD, STOP_AMPLITUDE, STOP_SIGMA_M, TERMINAL_REWARDS
from env.parallel_sumo import ParallelSumo, ShardedParallelSumo
from env.parallel_traffic import TrafficCurriculum
from utils.saved_configs import load_saved_configs
from utils.logger import RunLogger


def build_env(checkpoint, suite, output, args):
    spec = json.loads((suite / "suite.json").read_text())
    road, _, _ = load_saved_configs(str(suite / "reference/model.pt"))
    _, mdp, training = load_saved_configs(str(checkpoint))
    action = dict(mdp["ACTION"])
    action["lane_change_safety"] = dict(action["lane_change_safety"], enabled=False, spawn_protect=0)
    reward = dict(training["REWARD"], bc_ppo_reward=True, early_lane_change_bonus=0.,
                  early_lane_change_distance_m=120., overspeed_penalty=args.overspeed_penalty,
                  overspeed_tolerance=args.overspeed_tolerance)
    if args.lane_change_penalty:
        reward["bc_lane_change_penalty"] = args.lane_change_penalty
    template = ParallelSumo(
        args.envs // args.workers,
        dict(cfg_path=str(suite / "sumo/highway.sumocfg"), road=road["ROAD"], ego=road["EGO"],
             traffic=road["TRAFFIC"], mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
             action=action, reward=reward),
        output / "parallel", seed=args.seed, randomize_signals=True, randomize_departures=True,
        uniform_route_departures=True, traffic_curriculum=TrafficCurriculum(**spec["settings"]),
        fixed_traffic_vph=spec["traffic_vph"],
    )
    # Egos are shallow copies of the prototype, so this reaches every replica.
    template.prototype.disable_ego_rule_barriers()
    template.prototype.ego_speed_cap_enabled = args.ego_speed_cap
    return spec, mdp, training, action, reward, template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("traffic_suite", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--critic-warmup", type=int, default=5,
                        help="Leading iterations that train only the critic")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--anchor-coef", type=float, default=0.1)
    parser.add_argument("--init-std", type=float, default=0.1,
                        help="Acceleration std in action units (1 = 5.4 m/s^2)")
    parser.add_argument("--lane-sharpness", type=float, default=2.0,
                        help="Scale on the BC lane logits; keeps the argmax")
    parser.add_argument("--overspeed-penalty", type=float, default=0.,
                        help="Per-step penalty per unit of relative speed above the current lane's limit")
    parser.add_argument("--overspeed-tolerance", type=float, default=OVERSPEED_TOLERANCE,
                        help="Relative excess over the lane limit that is not penalized "
                             "(default 0.1 = 10%%, the IDM expert's speed spread)")
    parser.add_argument("--ego-speed-cap", action="store_true",
                        help="Clamp the ego speed to the lane limit (a rule outside the policy; off by default)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="PPO without BC: random actor, no anchor. The checkpoint folder still supplies "
                             "the settings and the input normalization (mean/std only)")
    parser.add_argument("--lane-keep-prob", type=float, default=0.9,
                        help="With --from-scratch: initial probability of the keep-lane command")
    parser.add_argument("--lane-change-penalty", type=float, default=0.,
                        help="Reward subtracted for each executed lane change (default 0, off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rollout = args.envs * args.steps_per_env
    if args.envs % args.workers or rollout % 1024 or args.iterations < 1:
        parser.error("Split environments evenly across workers and use a rollout divisible by 1024")
    if not (np.isfinite(args.overspeed_penalty) and args.overspeed_penalty >= 0
            and np.isfinite(args.overspeed_tolerance) and args.overspeed_tolerance >= 0):
        parser.error("Use a finite nonnegative overspeed penalty and tolerance")
    if not (np.isfinite(args.lane_change_penalty) and args.lane_change_penalty >= 0):
        parser.error("Use a finite nonnegative lane-change penalty")
    if args.from_scratch:
        if not 0 < args.lane_keep_prob < 1:
            parser.error("--lane-keep-prob must be in (0, 1)")
        args.anchor_coef = 0.0                # there is no BC policy to anchor to

    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve(strict=True)
    suite = args.traffic_suite.resolve(strict=True)
    output = (args.output or Path("results") / datetime.now().strftime("PPO_anchored_%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True)
    spec, mdp, training, action, reward, template = build_env(checkpoint, suite, output, args)
    hparams = dict(training["HPARAMS"], n_steps=rollout, n_epochs=args.epochs, minibatch_size=1024,
                   lr=args.lr, seed=args.seed, target_kl=args.target_kl, lane_entropy_coef=0.,
                   entropy_coef=0.)
    for name in ("road_config.py", "mdp_config.py", "train.py"):
        source = suite / "reference" if name == "road_config.py" else checkpoint.parent
        shutil.copy2(source / name, output / name)
    with (output / "mdp_config.py").open("a") as file:
        file.write(f"\nACTION = {action!r}\n")
    with (output / "train.py").open("a") as file:
        file.write(f"\nREWARD = {reward!r}\nHPARAMS = {hparams!r}\n")

    env = template if args.workers == 1 else ShardedParallelSumo(template, args.workers)
    logger = RunLogger(str(output))
    try:
        normalization = np.load(checkpoint.parent / "normalization.npz")
        agent = AnchoredPPO(obs_dim=template.prototype.obs_dim, act_dim=template.prototype.act_dim,
                            normalization=normalization, critic_lr=args.critic_lr,
                            anchor_coef=args.anchor_coef, critic_warmup_iterations=args.critic_warmup,
                            **hparams)
        if args.from_scratch:
            agent.init_from_scratch(normalization, args.init_std, args.lane_keep_prob)
        else:
            agent.init_from_bc(str(checkpoint), args.init_std, args.lane_sharpness)
        agent.save(str(output / "model_iter000.pt"))
        metadata = dict(initial_model=str(checkpoint), traffic_suite=str(suite),
                        traffic_vph=spec["traffic_vph"], rollout=rollout, num_envs=args.envs,
                        num_workers=args.workers, device=str(agent.device), step_reward=STEP_REWARD,
                        terminal_rewards=TERMINAL_REWARDS, stop_amplitude=STOP_AMPLITUDE,
                        stop_sigma_m=STOP_SIGMA_M, 
                        algorithm="PPO from scratch" if args.from_scratch else "AnchoredPPO",
                        **{k: v for k, v in vars(args).items()
                                                    if k not in ("checkpoint", "traffic_suite", "output")})
        if junction_collisions():
            metadata["junction_collisions"] = True
        (output / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        print(f"{output} | {agent.device} | {rollout} steps/iteration", flush=True)
        start = perf_counter()

        def checkpoint_iteration(iteration):
            if iteration % 10 == 0:
                agent.save(str(output / f"model_iter{iteration:03d}.pt"))
                agent.save_critic(str(output / "critic.pt"))
                logger.save_csv()
            print(f"Iteration {iteration}/{args.iterations}, elapsed {perf_counter() - start:.1f}s", flush=True)

        agent.learn_parallel(env, args.iterations, logger=logger, on_iteration=checkpoint_iteration)
        agent.save(str(output / "model.pt"))
        agent.save_critic(str(output / "critic.pt"))
        print(f"Training completed in {perf_counter() - start:.1f}s", flush=True)
    finally:
        env.close()
        logger.close()


if __name__ == "__main__":
    main()
