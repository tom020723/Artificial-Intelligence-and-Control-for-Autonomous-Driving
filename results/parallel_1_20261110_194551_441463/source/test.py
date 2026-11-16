"""학습된 모델을 sumo-gui로 재생하고 주행 지표를 출력한다.

사용법:
    python test.py results/run_20260824_153012/model.pt   ← 모델 경로 지정
    python test.py                                        ← 최근 run 자동 선택
    python test.py --episodes 5 --nogui                   ← 화면 없이 지표만 측정
    python test.py results/bc_example/model.pt --algorithm bc --nogui

빨간 차가 ego. 앞차가 느리면 감속하고 앞이 비면 가속하는지 관찰해보자.
"""
import argparse
import glob
import os
import runpy
import xml.etree.ElementTree as ET
import torch

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env import road_builder
from env.sumo_env import SumoHighwayEnv

from algorithms.ppo import PPO
from algorithms.bc import BCPolicy
from utils.evaluator import evaluate_policy

# train.py와 동일한 네트워크 구조로 만들어야 가중치를 불러올 수 있으므로
# 하이퍼파라미터를 그대로 가져온다 (구조 관련 항목만 실제로 쓰임).
from train import HPARAMS, RESULTS_DIR, REWARD  # ← 보상도 train과 동일 조건

BASE = os.path.dirname(os.path.abspath(__file__))


def load_saved_configs(model_path):
    """평가 프로세스 안에서만 run의 설정 snapshot을 복원한다."""
    from env import road_config, mdp_config
    folder = os.path.dirname(model_path)
    required = [os.path.join(folder, name) for name in ("road_config.py", "mdp_config.py", "train.py")]
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"저장된 설정이 없습니다: {path}")
    road = runpy.run_path(required[0])
    # 실험 1처럼 신호 실험 플래그가 없던 snapshot도 지원한다.
    road_config.SIGNAL_EXPERIMENT = road.get("SIGNAL_EXPERIMENT", False)
    for name, value in road.items():
        if name.isupper():
            setattr(road_config, name, value)
    mdp = runpy.run_path(required[1])
    for name, value in mdp.items():
        if name.isupper():
            setattr(mdp_config, name, value)
    training = runpy.run_path(required[2])
    road_builder.NOHYEONG = road_config.NOHYEONG
    road_builder.SCENERY = road_config.SCENERY
    road_builder.SIMULATION = mdp["SIMULATION"]
    return road, mdp, training


def find_latest_model() -> str:
    """results/*/model.pt 중 가장 최근에 저장된 것을 찾는다."""
    candidates = glob.glob(os.path.join(BASE, RESULTS_DIR, "*", "model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"'{RESULTS_DIR}/' 안에 model.pt가 없습니다. "
            f"먼저 train.py 또는 train_bc.py로 학습하세요.")
    return max(candidates, key=os.path.getmtime)


def load_agent(model_path, obs_dim, act_dim, algorithm="auto"):
    """BC 체크포인트와 기존 PPO state_dict를 구분해 평가 정책을 복원한다."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("지원하지 않는 모델 파일 형식입니다.")
    if checkpoint.get("algorithm") == "bc":
        detected = "bc"
    elif "trunk.0.weight" in checkpoint and "mu_head.weight" in checkpoint:
        detected = "ppo"
    else:
        raise ValueError("현재 BC 체크포인트 또는 PPO state_dict가 아닙니다.")
    if algorithm != "auto" and algorithm != detected:
        raise ValueError(f"--algorithm {algorithm}과 모델 유형 {detected}가 다릅니다.")

    if detected == "bc":
        if not {"state_dim", "action_dim", "state_dict"}.issubset(checkpoint):
            raise ValueError("현재 BCPolicy 형식이 아닙니다. 현재 train_bc.py로 학습한 모델을 사용하세요.")
        dims = (checkpoint["state_dim"], checkpoint["action_dim"])
        if dims != (obs_dim, act_dim):
            raise ValueError(f"BC 모델 입출력 차원 {dims} != 환경 {(obs_dim, act_dim)}")
        agent = BCPolicy.load(model_path, device="cpu")
    else:
        agent = PPO(obs_dim=obs_dim, act_dim=act_dim, **HPARAMS)
        try:
            agent.policy.load_state_dict(checkpoint)
        except RuntimeError as exc:
            raise ValueError("PPO 모델과 현재 환경 차원 또는 HPARAMS의 네트워크 구조가 다릅니다.") from exc
        agent.policy.eval()
    print(f"평가 알고리즘: {detected.upper()}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델 재생/평가")
    parser.add_argument("model", nargs="?", default=None,
                        help="model.pt 경로 (생략 시 최근 run 자동 선택)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="재생/평가할 에피소드 수")
    parser.add_argument("--route", default=None,
                        help="ego 경로 ID 고정 (예: ego_d_A4_A2). 생략하면 기존 랜덤 경로")
    parser.add_argument("--seed", type=int, default=None,
                        help="평가 시작 시드. 에피소드별 seed, seed+1, ... 적용")
    parser.add_argument("--saved-config", action="store_true",
                        help="모델 폴더의 저장된 환경/관측/보상/네트워크 설정으로 평가")
    parser.add_argument("--nogui", action="store_true",
                        help="화면 없이 주행 지표만 측정 (빠름)")
    parser.add_argument("--algorithm", choices=["auto", "ppo", "bc"], default="auto",
                        help="기본은 모델 파일에서 PPO/BC 자동 판별")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes는 1 이상이어야 합니다.")
    if args.seed is not None and not 0 <= args.seed <= 2**32 - args.episodes:
        parser.error("--seed는 0 이상이며 마지막 에피소드 시드는 2**32 미만이어야 합니다.")

    model_path = args.model or find_latest_model()
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    print(f"모델 로드: {model_path}")
    if args.saved_config:
        saved_road, saved_mdp, saved_train = load_saved_configs(model_path)
        ROAD, EGO, TRAFFIC = (saved_road[name] for name in ("ROAD", "EGO", "TRAFFIC"))
        SIMULATION, OBSERVATION, ACTION = (saved_mdp[name] for name in ("SIMULATION", "OBSERVATION", "ACTION"))
        HPARAMS, REWARD = saved_train["HPARAMS"], saved_train["REWARD"]
        print("저장된 실험 설정 복원 (현재 설정 파일 수정 없음)")

    # 주의: 현재 env/ 설정과 다른 설정으로 학습된 모델이면 성능이 이상할 수 있다.
    # 그 run의 설정 스냅샷은 model.pt와 같은 폴더의 *_config.py 에서 확인 가능.
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC,
                                 os.path.join(BASE, "env", "sumo"))
    if args.route:
        route_files = ET.parse(sumocfg).getroot().find("input/route-files").get("value").split(",")
        routes = {}
        for filename in route_files:
            path = os.path.join(os.path.dirname(sumocfg), filename.strip())
            routes.update({r.get("id"): r.get("edges")
                           for r in ET.parse(path).getroot().findall("route")})
        if args.route not in routes:
            parser.error(f"존재하지 않는 경로 ID: {args.route}")
        print(f"고정 경로: {args.route} ({routes[args.route]})")
    env = SumoHighwayEnv(
        cfg_path=sumocfg,
        road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD,
        traffic=TRAFFIC,
        gui=not args.nogui,
        gui_autostart=False,   # ▶(플레이) 버튼을 눌러야 에피소드 시작
    )
    if args.route:
        # 평가 인스턴스에만 적용한다. 학습 설정과 도로 원본은 변경하지 않는다.
        env._choose_ego_route = lambda: args.route

    if not args.nogui:
        print("※ 창이 뜨면 ▶(플레이) 버튼을 눌러야 에피소드가 시작됩니다.")
        print("  에피소드가 끝나면 다음 에피소드 창이 다시 뜨고, 또 ▶를 누르면 됩니다.")

    try:
        agent = load_agent(model_path, env.observation_space.shape[0],
                           env.action_space.shape[0], args.algorithm)
        if args.seed is not None:
            print(f"평가 시드: {args.seed} ~ {args.seed + args.episodes - 1}")
        metrics = evaluate_policy(agent, env, n_episodes=args.episodes,
                                  deterministic=True, seed=args.seed)
    finally:
        env.close()

    print(f"\n─── 주행 지표 ({args.episodes} 에피소드 평균) ───")
    print(f"  신호위반율   : {metrics.red_light_violation_rate:.0%}")
    print(f"  청신호 통과  : {metrics.green_passes:.2f} 회/에피소드")
    print(f"  적신호 정차 감점 합: {metrics.red_stop_cost:.3f} /에피소드")
    print(f"  차선실패율   : {metrics.wrong_lane_rate:.0%}")
    print(f"  충돌률       : {metrics.collision_rate:.0%}")
    print(f"  완주율       : {metrics.success_rate:.0%}")
    print(f"  시간초과율   : {metrics.timeout_rate:.0%}")
    print(f"  평균 속도    : {metrics.mean_speed:.2f} m/s "
          f"({metrics.mean_speed * 3.6:.1f} km/h)")
    print(f"  평균 차간거리: {metrics.mean_gap:.2f} m")
    print(f"  최소 차간거리: {metrics.min_gap:.2f} m")
    print(f"  차선변경     : {metrics.lane_changes:.1f} 회/에피소드")
    print(f"  평균 보상    : {metrics.ep_return:.2f}")
    print(f"  평균 길이    : {metrics.ep_length:.0f} 스텝")
