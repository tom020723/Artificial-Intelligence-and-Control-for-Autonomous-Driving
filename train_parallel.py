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
from env.road_config import NOHYEONG
from train import (
    ACTION, BASE, EGO, HPARAMS, LOG_INTERVAL,
    OBSERVATION, RESULTS_DIR, REWARD, ROAD, SIMULATION, TRAFFIC, road_builder, snapshot_configs,
)
from utils.evaluator import evaluate_parallel_policy
from utils.logger import RunLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1,
                        help="학습 단계. 1: 항상 초록불 / 2: 신호등 작동 (둘 다 배경 차량 없음)")
    parser.add_argument("--set", action="append", default=[], metavar="GROUP.KEY=VALUE",
                        help="설정 덮어쓰기. GROUP은 reward / obs / hparams. "
                             "예: --set reward.red_overspeed_coef=0.3 --set hparams.gamma=0.995")
    parser.add_argument("--signal-green", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Phase 2 학습용: 도로마다 초록불 길이(초)를 이 범위에서 무작위로 정함. "
                             "생략하면 road_config의 고정 길이 사용")
    parser.add_argument("--init-model", type=Path, default=None,
                        help="이어서 학습할 model.pt (예: Phase 1 모델). 관측이 뒤쪽으로 늘어난 경우 자동으로 맞춤")
    parser.add_argument("--name", default=None, help="결과 폴더 이름 뒤에 붙일 실험 이름")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="정기 평가 간격 (iteration 수, 0이면 비활성화)")
    parser.add_argument("--checkpoint-interval", type=int, default=0,
                        help="이 iteration 수마다 중간 모델 저장 (0이면 저장 안 함)")
    parser.add_argument("--results-dir", type=Path, default=Path(BASE) / RESULTS_DIR)
    args = parser.parse_args()
    NOHYEONG["all_signals_green"] = args.phase == 1
    groups = {"reward": REWARD, "obs": OBSERVATION, "hparams": HPARAMS}
    overrides = {}
    for item in args.set:
        target, _, value = item.partition("=")
        group, _, key = target.partition(".")
        if group not in groups or not key or not value:
            parser.error(f"--set 형식 오류: {item}")
        groups[group][key] = overrides[target] = json.loads(value)
    rollout_steps = args.num_envs * args.steps_per_env
    if min(args.num_envs, args.steps_per_env, args.iterations, args.minibatch_size) < 1:
        parser.error("환경 수, rollout 길이, iteration 수, 미니배치는 양수여야 합니다.")
    if rollout_steps < 2 or rollout_steps % args.minibatch_size:
        parser.error("rollout은 2 step 이상이며 미니배치 크기의 배수여야 합니다.")
    if args.eval_interval < 0:
        parser.error("--eval-interval은 0 이상이어야 합니다.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_{args.name}" if args.name else ""
    run_dir = args.results_dir.resolve() / f"parallel_{args.phase}_{timestamp}{suffix}"
    run_dir.mkdir(parents=True)
    snapshot_configs(run_dir)
    # test.py --saved-config rebuilds the road from this copy, so record the phase in it.
    saved_road = run_dir / "road_config.py"
    saved_road.write_text(saved_road.read_text(encoding="utf-8").replace(
        "ROUTE_ONLY_EXPERIMENT = True", f"ROUTE_ONLY_EXPERIMENT = {args.phase == 1}", 1), encoding="utf-8")
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
            "initialization": str(args.init_model) if args.init_model else "random",
            "overrides": overrides,
            "signal_green": args.signal_green,
        }, file, indent=2)

    config = road_builder.build(ROAD, EGO, TRAFFIC, str(run_dir / "sumo"))
    env_kwargs = dict(cfg_path=config, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                      mdp_obs=OBSERVATION, action=ACTION, reward=REWARD, traffic=TRAFFIC)
    hparams = dict(HPARAMS, n_steps=rollout_steps, minibatch_size=args.minibatch_size)
    print(f"결과 폴더: {run_dir}", flush=True)
    env = ParallelSumo(args.num_envs, env_kwargs, run_dir / "parallel", hparams["seed"],
                       green_range=args.signal_green)
    agent = ParallelPPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim, **hparams)
    if args.init_model:
        import torch
        state = torch.load(args.init_model, map_location=agent.device)
        first = state["trunk.0.weight"]
        extra = env.prototype.obs_dim - first.shape[1]
        if extra < 0:
            parser.error("--init-model의 관측 차원이 현재 환경보다 큽니다.")
        # New observation dims are appended last; zero weights keep the old behaviour at start.
        state["trunk.0.weight"] = torch.cat([first, first.new_zeros(first.shape[0], extra)], dim=1)
        if "hparams.init_log_std" in overrides:
            state["log_std"] = state["log_std"].new_full(state["log_std"].shape, HPARAMS["init_log_std"])
        agent.policy.load_state_dict(state)
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
                                 evaluate=evaluate, eval_interval=args.eval_interval,
                                 checkpoint=lambda i: agent.save(run_dir / f"model_iter{i:04d}.pt"),
                                 checkpoint_interval=args.checkpoint_interval)
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
