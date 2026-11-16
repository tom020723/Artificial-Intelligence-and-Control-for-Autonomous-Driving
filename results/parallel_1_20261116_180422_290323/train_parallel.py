"""Train one PPO policy with disconnected roads across libsumo workers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

from algorithms.parallel_ppo import ParallelPPO
from env.parallel_sumo import ParallelSumo, ShardedParallelSumo
from env.parallel_traffic import TrafficCurriculum
from env import road_config
from train import (
    ACTION, BASE, EGO, HPARAMS, LOG_INTERVAL,
    OBSERVATION, RESULTS_DIR, REWARD, ROAD, SIMULATION, TRAFFIC, road_builder, snapshot_configs,
)
from utils.evaluator import evaluate_parallel_policy
from utils.logger import RunLogger


SIGNAL_STOP_REWARDS = dict(
    wrong_lane_penalty=5.0,
    green_pass_bonus=20.0,
    green_stop_penalty=0.02,       # 스텝당 감점
    green_stop_grace_seconds=1.0,
    green_stop_speed=0.1,          # m/s 이하를 정지로 간주
    green_stop_clear_gap=10.0,     # 앞차가 없거나 범퍼 간격이 이 이상이면 진행 가능
    red_stop_penalty_coef=0.0,      # 멀리 있다고 감점하지 않음
    red_stop_close_penalty_coef=0.05,  # 적신호 정지선 1m 안쪽에서 거리 비례 감점
    red_stop_close_distance=1.0,
    red_stop_clearance_reward=0.0, # 정지 중에도 들어오던 거리 보상 비활성화
    red_stop_progress_per_meter=0.0,  # 적신호 전진량 보상 비활성화
    red_speed_tracking_cap=0.20,      # 0.02, 0.05 실험을 거쳐 57번 설정 채택.
    red_speed_tracking_approach_only=True,  # 같은 적신호 정지선에 전진 중, 정차 구간 밖에서만 가점
    red_speed_tracking_linear_underspeed=True,  # 목표보다 느릴 때도 출발부터 보상
    red_speed_tracking_target_decel=1.5,  # 목표 속도 곡선의 감속도(m/s²)
    red_speed_tracking_safe_decel=2.5,    # 정지 여유 판정의 감속도(m/s²)
    red_stopped_brake_coef=0.01,    # 적신호 정차 중 음수 가속 명령의 제곱 감점
    red_stopped_brake_speed=0.1,    # 행동 전 속도(m/s) 기준
    red_stopped_brake_min_room=2.0, # 목표 정차 위치까지 남아 있어야 할 추가 여유(m)
    red_safe_zone_first_reward=5.0,   # 적신호 안전 구역 최초 정차 보상
    red_safe_zone_hold_reward=0.005,  # 이후 안전 구역 유지 스텝당 보상
    red_safe_zone_hold_cap=0.2,       # 신호별 유지 보상 누적 상한
    red_safe_zone_min_gap=2.5,
    red_safe_zone_max_gap=4.0,
    red_safe_zone_max_speed=0.1,
    red_safe_zone_dwell_seconds=1.0,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1,
                        help="1: 신호등, 배경 차량 없음; 2: 신호등과 배경 교통량 커리큘럼")
    parser.add_argument("--init-model", type=Path,
                        help="초기 정책 가중치로 사용할 기존 model.pt")
    parser.add_argument("--num-envs", type=int, default=4096, help="전체 환경 수")
    parser.add_argument("--num-workers", type=int, default=4, help="학습용 SUMO 프로세스 수")
    parser.add_argument("--randomize-departures", choices=("True", "False"), default="False",
                        help="ego 출발 차선 랜덤화 (기본 False: 도로 설정의 차선으로 고정)")
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=333)
    parser.add_argument("--seed", type=int, default=HPARAMS["seed"], help="학습 난수 시드")
    parser.add_argument("--lookahead-edges", type=int, default=2, help="차선 연결성 확인 구간 수")
    parser.add_argument("--lane-decision-distance-scale", type=float, default=0.0,
                        help="경로에 맞는 차선이 제한되는 지점까지 거리 관측 범위(m), 0이면 비활성화")
    parser.add_argument("--lane-change-margin-scale", type=float, default=0.0,
                        help="차선을 유지한다고 가정한 차선 변경 여유 관측 범위(m), 0이면 비활성화")
    parser.add_argument("--junction-lane-reference", action="store_true",
                        help="교차로 내부 차선을 다음 일반 차선 기준으로 관측")
    parser.add_argument("--off-route-lane-penalty", type=float, default=0.0,
                        help="경로에 연결되지 않는 차선에서 스텝당 감점")
    parser.add_argument("--early-lane-change-bonus", type=float, default=0.0,
                        help="갈림길 전 필요한 차선 변경 횟수를 한 칸 줄일 때의 가점 (0: 끔)")
    parser.add_argument("--lane-keep-bias", type=float, default=HPARAMS["lane_keep_bias"],
                        help="차선 유지의 초기 정책 편향")
    parser.add_argument("--checkpoint-iterations", type=int, nargs="+", default=[],
                        help="지정한 iteration의 PPO 업데이트 후 가중치 저장")
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-epochs", type=int, default=5,
                        help="한 rollout을 PPO로 반복 학습하는 횟수")
    parser.add_argument("--target-kl", type=float, default=0.0,
                        help="미니배치 추정 KL이 이 값의 1.5배를 넘으면 rollout 업데이트를 종료 (0: 끔)")
    parser.add_argument("--eval-interval", type=int, default=0,
                        help="정기 평가 간격 (iteration 수, 기본값 0: 비활성화)")
    parser.add_argument("--results-dir", type=Path, default=Path(BASE) / RESULTS_DIR)
    parser.add_argument("--no-red-stop-hold-reward", action="store_true",
                        help="적신호 정차 유지 가점 비활성화")
    parser.add_argument("--no-green-stop-penalty", action="store_true",
                        help="청신호 정차 감점 비활성화")
    parser.add_argument("--red-park-penalty-coef", type=float, default=0.0,
                        help="적신호 저속 거리 감점의 스텝당 상한, 0이면 비활성화")
    parser.add_argument("--red-speed-tracking-cap", type=float,
                        default=SIGNAL_STOP_REWARDS["red_speed_tracking_cap"],
                        help="적신호 접근 속도 추종 보상의 스텝당 상한")
    parser.add_argument("--legacy-red-speed-tracking", action="store_true",
                        help="93번처럼 접근 스텝과 정차 구간에 대한 추종 가점 제한을 끔")
    parser.add_argument("--yellow-tracking-max-decel", type=float, default=0.0,
                        help="황신호 정차 가능 판정 및 추종 감속도 상한(m/s²), 0이면 비활성화")
    parser.add_argument("--signal-fine-distance-scale", type=float, default=0.0,
                        help="신호별 정밀 거리 관측 범위(m), 0이면 비활성화")
    args = parser.parse_args()
    if not 0 <= args.target_kl < float("inf"):
        parser.error("--target-kl은 유한한 0 이상의 값이어야 합니다.")
    if args.lookahead_edges < 1:
        parser.error("--lookahead-edges는 1 이상이어야 합니다.")
    if not 0 <= args.lane_decision_distance_scale < float("inf"):
        parser.error("--lane-decision-distance-scale은 유한한 0 이상의 값이어야 합니다.")
    if not 0 <= args.lane_change_margin_scale < float("inf"):
        parser.error("--lane-change-margin-scale은 유한한 0 이상의 값이어야 합니다.")
    if not 0 <= args.off_route_lane_penalty < float("inf"):
        parser.error("--off-route-lane-penalty는 유한한 0 이상의 값이어야 합니다.")
    if not 0 <= args.early_lane_change_bonus < float("inf"):
        parser.error("--early-lane-change-bonus는 유한한 0 이상의 값이어야 합니다.")
    if not abs(args.lane_keep_bias) < float("inf"):
        parser.error("--lane-keep-bias는 유한한 값이어야 합니다.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed는 0 이상 2**32 미만이어야 합니다.")
    checkpoint_iterations = set(args.checkpoint_iterations)
    if any(not 1 <= i <= args.iterations for i in checkpoint_iterations):
        parser.error("--checkpoint-iterations는 1~--iterations 사이여야 합니다.")
    if not 0 <= args.red_park_penalty_coef < float("inf"):
        parser.error("--red-park-penalty-coef는 유한한 0 이상의 값이어야 합니다.")
    if not 0 <= args.red_speed_tracking_cap < float("inf"):
        parser.error("--red-speed-tracking-cap은 유한한 0 이상의 값이어야 합니다.")
    if not 0 <= args.yellow_tracking_max_decel < float("inf"):
        parser.error("--yellow-tracking-max-decel은 유한한 0 이상의 값이어야 합니다.")
    if not 0 <= args.signal_fine_distance_scale < float("inf"):
        parser.error("--signal-fine-distance-scale은 유한한 0 이상의 값이어야 합니다.")
    randomize_departures = args.randomize_departures == "True"
    if args.init_model is not None:
        args.init_model = args.init_model.resolve()
        if not args.init_model.is_file():
            parser.error("--init-model에는 기존 model.pt를 지정해야 합니다.")
    rollout_steps = args.num_envs * args.steps_per_env
    if min(args.num_envs, args.num_workers, args.steps_per_env, args.iterations,
           args.minibatch_size, args.learning_epochs) < 1:
        parser.error("환경·워커 수, rollout 길이, iteration·epoch 수, 미니배치는 양수여야 합니다.")
    if args.num_envs % args.num_workers:
        parser.error("전체 환경 수는 워커 수의 배수여야 합니다.")
    envs_per_worker = args.num_envs // args.num_workers
    if rollout_steps < 2 or rollout_steps % args.minibatch_size:
        parser.error("rollout은 2 step 이상이며 미니배치 크기의 배수여야 합니다.")
    if args.eval_interval < 0:
        parser.error("--eval-interval은 0 이상이어야 합니다.")
    minibatches_per_epoch = rollout_steps // args.minibatch_size

    road_config.configure_parallel_training(args.phase)
    curriculum = TrafficCurriculum() if args.phase == 2 else None
    signal_rewards = dict(SIGNAL_STOP_REWARDS)
    if args.phase == 1:
        signal_rewards.update(red_safe_zone_min_gap=1.5, red_safe_zone_max_gap=3.0,
                              red_stop_margin=2.25)
    signal_rewards["red_park_penalty_coef"] = args.red_park_penalty_coef
    signal_rewards["red_speed_tracking_cap"] = args.red_speed_tracking_cap
    if args.legacy_red_speed_tracking:
        signal_rewards["red_speed_tracking_approach_only"] = False
    signal_rewards["yellow_tracking_max_decel"] = args.yellow_tracking_max_decel
    signal_rewards["off_route_lane_penalty"] = args.off_route_lane_penalty
    signal_rewards["early_lane_change_bonus"] = args.early_lane_change_bonus
    if args.no_red_stop_hold_reward:
        signal_rewards["red_safe_zone_hold_reward"] = 0.0
    if args.no_green_stop_penalty:
        signal_rewards["green_stop_penalty"] = 0.0
    reward = {**REWARD, **signal_rewards}
    observation = dict(OBSERVATION, destination_enabled=False, signal_count=1, signal_enabled=True)
    observation["signal_fine_distance_scale"] = args.signal_fine_distance_scale
    observation.update(lookahead_edges=args.lookahead_edges,
                       lane_decision_distance_scale=args.lane_decision_distance_scale,
                       lane_change_margin_scale=args.lane_change_margin_scale,
                       junction_lane_reference=args.junction_lane_reference)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = args.results_dir.resolve() / f"parallel_{args.phase}_{timestamp}"
    run_dir.mkdir(parents=True)
    snapshot_configs(run_dir)
    for path in (run_dir / "mdp_config.py", run_dir / "source/env/mdp_config.py"):
        with path.open("a", encoding="utf-8") as file:
            file.write(f"\nOBSERVATION = {observation!r}\n")
    for path in (run_dir / "road_config.py", run_dir / "source/env/road_config.py"):
        with path.open("a", encoding="utf-8") as file:
            file.write(f"\nconfigure_parallel_training({args.phase})\n")
    for path in (run_dir / "train.py", run_dir / "source/train.py"):
        with path.open("a", encoding="utf-8") as file:
            file.write(f"\nREWARD.update({signal_rewards!r})\n")
            file.write(f"HPARAMS['seed'] = {args.seed}\n")
            file.write(f"HPARAMS['lane_keep_bias'] = {args.lane_keep_bias}\n")
            file.write(f"HPARAMS['target_kl'] = {args.target_kl or None!r}\n")
    for directory in (run_dir, run_dir / "source"):
        shutil.copy2(__file__, directory)
    total_steps = rollout_steps * args.iterations
    with (run_dir / "parallel_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "phase": args.phase,
            "num_envs": args.num_envs,
            "num_workers": args.num_workers,
            "envs_per_worker": envs_per_worker,
            "steps_per_env": args.steps_per_env,
            "rollout_steps": rollout_steps,
            "iterations": args.iterations,
            "total_timesteps": total_steps,
            "minibatch_size": args.minibatch_size,
            "minibatches_per_epoch": minibatches_per_epoch,
            "epochs": args.learning_epochs,
            "target_kl": args.target_kl or None,
            "seed": args.seed,
            "eval_interval": args.eval_interval,
            "checkpoint_iterations": sorted(checkpoint_iterations),
            "eval_episodes": envs_per_worker if args.eval_interval else 0,
            "eval_seed": args.seed + 1 if args.eval_interval else None,
            "randomize_signals": True,
            "randomize_departures": randomize_departures,
            "lookahead_edges": args.lookahead_edges,
            "lane_decision_distance_scale": args.lane_decision_distance_scale,
            "lane_change_margin_scale": args.lane_change_margin_scale,
            "junction_lane_reference": args.junction_lane_reference,
            "off_route_lane_penalty": args.off_route_lane_penalty,
            "early_lane_change_bonus": args.early_lane_change_bonus,
            "lane_keep_bias": args.lane_keep_bias,
            "signal_fine_distance_scale": args.signal_fine_distance_scale,
            "red_stop_hold_reward": signal_rewards["red_safe_zone_hold_reward"],
            "green_stop_penalty": signal_rewards["green_stop_penalty"],
            "red_park_penalty_coef": args.red_park_penalty_coef,
            "red_speed_tracking_cap": args.red_speed_tracking_cap,
            "yellow_tracking_max_decel": args.yellow_tracking_max_decel,
            "red_speed_tracking_approach_only": signal_rewards["red_speed_tracking_approach_only"],
            "independent_episode_rng": True,
            "traffic_curriculum": asdict(curriculum) if curriculum else None,
            "periodic_eval_traffic_vph": 800 if curriculum and args.eval_interval else 0,
            "ring_route_probability": road_config.NOHYEONG["ego_routes"]["ring_prob"],
            "initialization": "random" if args.init_model is None else str(args.init_model),
        }, file, indent=2)

    config = road_builder.build(ROAD, EGO, TRAFFIC, str(run_dir / "sumo"))
    env_kwargs = dict(cfg_path=config, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                      mdp_obs=observation, action=ACTION, reward=reward, traffic=TRAFFIC)
    hparams = dict(HPARAMS, n_steps=rollout_steps, minibatch_size=args.minibatch_size,
                   n_epochs=args.learning_epochs, seed=args.seed, lane_keep_bias=args.lane_keep_bias,
                   target_kl=args.target_kl or None)
    print(f"결과 폴더: {run_dir}", flush=True)
    template = ParallelSumo(envs_per_worker, env_kwargs, run_dir / "parallel", hparams["seed"],
                            randomize_signals=True, randomize_departures=randomize_departures,
                            independent_episode_rng=True,
                            traffic_curriculum=curriculum)
    env = template if args.num_workers == 1 else ShardedParallelSumo(template, args.num_workers)
    agent = ParallelPPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim, **hparams)
    if args.init_model is not None:
        agent.load(str(args.init_model))
        shutil.copy2(args.init_model, run_dir / "initial_model.pt")
    print(f"Phase {args.phase} | {args.num_workers} workers × {envs_per_worker} envs | "
          f"rollout: {args.num_envs} × {args.steps_per_env} = {rollout_steps} step, "
          f"{args.iterations} iterations, {total_steps} total steps, "
          f"{args.learning_epochs} epochs × {minibatches_per_epoch} minibatches, "
          f"device={agent.device}", flush=True)
    eval_env = None
    if args.eval_interval:
        # Share the generated network files. The worker owns its own simulator.
        eval_env = deepcopy(template)
        eval_env.seed += 1
        if curriculum is not None:
            eval_env.fixed_traffic_vph = 800
    logger = RunLogger(run_dir)

    def save_checkpoint(iteration):
        if iteration in checkpoint_iterations:
            path = run_dir / "checkpoints" / f"iteration_{iteration:04d}" / "model.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(path)
            print(f"체크포인트 저장: {path}", flush=True)

    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
            def evaluate():
                policy = deepcopy(agent.policy).cpu()
                return executor.submit(evaluate_parallel_policy, policy, eval_env, str(agent.device)).result()

            agent.learn_parallel(env, total_steps, logger=logger, log_interval=LOG_INTERVAL,
                                 evaluate=evaluate, eval_interval=args.eval_interval,
                                 eval_episodes=envs_per_worker,
                                 on_iteration=save_checkpoint if checkpoint_iterations else None)
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
