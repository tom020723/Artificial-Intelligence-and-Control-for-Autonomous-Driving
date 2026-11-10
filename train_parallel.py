"""Train Phase 1 PPO with disconnected road replicas in one libsumo simulation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

from algorithms.parallel_ppo import ParallelPPO
from env.parallel_sumo import ParallelSumo
from env.road_config import ROUTE_ONLY_EXPERIMENT
from train import (
    ACTION, BASE, EGO, HPARAMS, LOG_INTERVAL,
    OBSERVATION, RESULTS_DIR, REWARD, ROAD, SIMULATION, TRAFFIC, road_builder, snapshot_configs,
)
from utils.evaluator import evaluate_parallel_policy
from utils.logger import RunLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1,), default=1,
                        help="학습 단계 (현재 지원: Phase 1, 기본값: 1)")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="정기 평가 간격 (iteration 수, 0이면 비활성화)")
    parser.add_argument("--results-dir", type=Path, default=Path(BASE) / RESULTS_DIR)
    args = parser.parse_args()
    if not ROUTE_ONLY_EXPERIMENT:
        parser.error("--phase 1은 env/road_config.py의 ROUTE_ONLY_EXPERIMENT=True 설정이 필요합니다.")
    rollout_steps = args.num_envs * args.steps_per_env
    if min(args.num_envs, args.steps_per_env, args.iterations, args.minibatch_size) < 1:
        parser.error("환경 수, rollout 길이, iteration 수, 미니배치는 양수여야 합니다.")
    if rollout_steps < 2 or rollout_steps % args.minibatch_size:
        parser.error("rollout은 2 step 이상이며 미니배치 크기의 배수여야 합니다.")
    if args.eval_interval < 0:
        parser.error("--eval-interval은 0 이상이어야 합니다.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.results_dir.resolve() / f"parallel_{args.phase}_{timestamp}"
    run_dir.mkdir(parents=True)
    snapshot_configs(run_dir)
    for directory in (run_dir, run_dir / "source"):
        shutil.copy2(__file__, directory)
    total_steps = rollout_steps * args.iterations
    with (run_dir / "parallel_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "phase": args.phase,
            "num_envs": args.num_envs,
            "steps_per_env": args.steps_per_env,
            "rollout_steps": rollout_steps,
            "iterations": args.iterations,
            "total_timesteps": total_steps,
            "minibatch_size": args.minibatch_size,
            "epochs": HPARAMS["n_epochs"],
            "seed": HPARAMS["seed"],
            "eval_interval": args.eval_interval,
            "eval_episodes": args.num_envs if args.eval_interval else 0,
            "eval_seed": HPARAMS["seed"] + 1 if args.eval_interval else None,
            "initialization": "random",
        }, file, indent=2)

    config = road_builder.build(ROAD, EGO, TRAFFIC, str(run_dir / "sumo"))
    env_kwargs = dict(cfg_path=config, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                      mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC)
    hparams = dict(HPARAMS, n_steps=rollout_steps, minibatch_size=args.minibatch_size)
    print(f"결과 폴더: {run_dir}", flush=True)
    env = ParallelSumo(args.num_envs, env_kwargs, run_dir / "parallel", hparams["seed"])
    agent = ParallelPPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim, **hparams)
    print(f"Phase {args.phase} | rollout: {args.num_envs} × {args.steps_per_env} = {rollout_steps} step, "
          f"{args.iterations} iterations, {total_steps} total steps, device={agent.device}", flush=True)
    eval_env = None
    if args.eval_interval:
        # Share the generated network files. The worker owns its own simulator.
        eval_env = deepcopy(env)
        eval_env.seed += 1
    logger = RunLogger(run_dir)
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            def evaluate():
                policy = deepcopy(agent.policy).cpu()
                return executor.submit(evaluate_parallel_policy, policy, eval_env, str(agent.device)).result()

            agent.learn_parallel(env, total_steps, logger=logger, log_interval=LOG_INTERVAL,
                                 evaluate=evaluate, eval_interval=args.eval_interval)
    except KeyboardInterrupt:
        print("학습을 중단하고 현재 모델을 저장합니다.")
    finally:
        try:
            env.close()
        finally:
            logger.close()
    agent.save(run_dir / "model.pt")
    print(f"저장 완료: {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
