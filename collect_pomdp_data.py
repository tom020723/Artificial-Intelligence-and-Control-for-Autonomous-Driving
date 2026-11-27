"""POMDP trajectory data collector for the SUMO highway environment.

저장 단위는 POMDP transition 하나다:
    (o_t, a_t, r_t, o_{t+1}, terminated, truncated, s_t, s_{t+1})

- o_t / o_{t+1}: agent가 실제로 보는 부분관측 (최종 BC 모델 기준 52차원)
- a_t: [가감속 raw, 차선변경 raw]
- lane_change: 실제 환경에 적용된 {-1, 0, +1}
- s_t / s_{t+1}: SUMO 내부의 privileged simulator state.
  정책 입력에는 쓰지 않고 데이터 검증/representation/belief 학습용으로만 저장한다.

용량을 줄이기 위해:
- JSONL은 gzip으로 압축해서 쓰고(.jsonl.gz), observation/state/detected_vehicles의
  float는 --ndigits 자리로 반올림한다 (NPZ는 float32 그대로).
- next_observation / next_state는 다음 레코드의 observation / state와 같으므로
  에피소드 마지막 스텝(terminated or truncated)에만 기록한다.
  읽을 때는 iter_transitions()가 채워서 돌려준다.
- NPZ의 감지 차량 좌표는 object 배열(pickle) 대신 flat (M, 2) + offsets로 저장한다.

속도:
- headless 수집은 libsumo(프로세스 안 SUMO)로 통신해 TraCI보다 약 10배 빠르다 (GUI는 TraCI). --workers N으로 SUMO 인스턴스 N개를
  병렬로 돌려 시간을 줄인다.

예시:
    python collect_pomdp_data.py --episodes 100
    python collect_pomdp_data.py --episodes 10 --gui
    python collect_pomdp_data.py --policy keep-lane --episodes 20
    python collect_pomdp_data.py --model results/run_xxx/model.pt --episodes 100   # 학습한 PPO로 수집
    python collect_pomdp_data.py --model results/run_xxx/model.pt --stochastic    # 샘플링 행동
    python collect_pomdp_data.py --episodes 200 --workers 8   # SUMO 8개 병렬 수집

출력:
    data/pomdp_YYYYMMDD_HHMMSS.jsonl.gz
    data/pomdp_YYYYMMDD_HHMMSS.npz

읽기:
    from collect_pomdp_data import iter_transitions
    for rec in iter_transitions("data/pomdp_....jsonl.gz"):
        rec["observation"], rec["next_observation"], rec["state"], rec["next_state"], ...

    d = np.load("data/pomdp_....npz")          # allow_pickle 불필요
    off = d["detected_offsets"]
    xy_i = d["detected_xy"][off[i]:off[i + 1]]  # 스텝 i의 감지 차량 (N_i, 2)
"""
import argparse
import gzip
import json
import multiprocessing
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env import road_builder
from env.sumo_env import SumoHighwayEnv
from train import HPARAMS, REWARD  # 학습과 같은 보상 조건 + PPO 네트워크 구조

BASE = os.path.dirname(os.path.abspath(__file__))


class LibsumoEnv(SumoHighwayEnv):
    """headless 수집용: TraCI 소켓 대신 libsumo(프로세스 안 SUMO)로 통신한다.

    같은 입력이면 TraCI와 결과가 같고 약 10배 빠르다. libsumo는 프로세스당 SUMO
    하나만 띄울 수 있지만, 워커마다 별도 프로세스라 문제없다. GUI는 TraCI만 가능.
    """
    @property
    def _api(self):
        import libsumo
        return libsumo


def detect_algorithm(model_path: str) -> str:
    """model.pt가 어느 학습 스크립트 출력인지 파일 내용으로 판별한다.

    train.py(PPO)  : policy.state_dict()를 그대로 저장 → 텐서만 있는 dict
    train_bc.py(BC): {"algorithm": "bc", "state_dim", ..., "state_dict"} dict
    train_bc_shards.py(BC): PPO와 같은 actor state_dict로 내보내고, 같은 폴더에
                    bc_training_state.pt를 남긴다 → BC로 기록하되 PPO actor로 불러온다.
    """
    # PPO evaluation exports also carry the initial BC's config.json.
    # The enclosing PPO run takes precedence over that copied metadata.
    if ppo_run_directory(model_path) is not None:
        return "ppo"
    config = Path(model_path).parent / "config.json"
    if config.is_file():
        algorithm = json.loads(config.read_text(encoding="utf-8")).get("algorithm")
        if algorithm == "supervised_bc_on_ppo_hybrid_actor":
            return "bc"
    if is_bc_policy_checkpoint(model_path):
        return "bc"
    bc_export = os.path.join(os.path.dirname(model_path), "bc_training_state.pt")
    return "bc" if os.path.isfile(bc_export) else "ppo"


def ppo_run_directory(model_path):
    folder = Path(model_path).resolve().parent
    candidates = [folder]
    if folder.parent.name == "eval_runs":
        candidates.append(folder.parent.parent)
    for candidate in candidates:
        if (candidate / "run.json").is_file():
            return candidate
    return None


def is_bc_policy_checkpoint(model_path: str) -> bool:
    """train_bc.py의 BCPolicy 형식이면 True (PPO 형식 actor면 False)."""
    import torch
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    return isinstance(ck, dict) and ck.get("algorithm") == "bc"


def make_policy(args, env, worker_seed: int):
    """args.policy에 맞는 obs -> action_raw(np.float32, shape (2,)) 함수를 만든다.

    ppo / bc 출력은 [-1, 1]로 클리핑해 저장한다. 환경의 입력 범위이기도 하고,
    train_bc.py가 |action_raw| <= 1을 검증하므로 이 데이터로 바로 BC를 돌릴 수 있다.
    """
    # train_bc_shards.py의 BC는 PPO와 같은 actor 형식이라 PPO 경로로 불러온다 (결정적 행동).
    if args.policy == "ppo" or (args.policy == "bc" and not is_bc_policy_checkpoint(args.model)):
        # test.py와 동일: train.py의 HPARAMS로 같은 구조를 만든 뒤 가중치만 덮어쓴다.
        # seed는 워커마다 다르게 (--stochastic 샘플링이 워커 간에 겹치지 않도록).
        from algorithms.ppo import PPO
        agent = PPO(obs_dim=env.observation_space.shape[0],
                    act_dim=env.action_space.shape[0],
                    **{**HPARAMS, "device": args.device, "seed": worker_seed})
        agent.load(args.model)
        deterministic = not args.stochastic   # hybrid: (μ, argmax 차선) / 샘플링

        def act(obs):
            a = agent.predict(obs, deterministic=deterministic)
            return np.clip(a, -1.0, 1.0).astype(np.float32)
        return act

    if args.policy == "bc":
        raise ValueError("Legacy BCPolicy checkpoints are unsupported; use scripts/train_bc_shards.py model.pt")

    if args.policy == "keep-lane":
        # 완만한 가속 + 차선 유지. GUI/파이프라인 점검용.
        return lambda obs: np.asarray([0.25, 0.0], dtype=np.float32)

    # random: 가감속은 연속 uniform[-1,1], 차선변경은 {-1,0,+1} 직접 샘플링 (유지를 더 자주)
    rng = np.random.default_rng(worker_seed)

    def act(obs):
        accel = rng.uniform(-1.0, 1.0)
        lane = rng.choice(np.asarray([-1.0, 0.0, 1.0]), p=[0.15, 0.70, 0.15])
        return np.asarray([accel, lane], dtype=np.float32)
    return act


def compact(obj, ndigits: int):
    """중첩 dict/list/ndarray를 float가 ndigits 자리로 반올림된 순수 Python 객체로 바꾼다.

    float32 값을 그대로 json.dumps하면 0.10000000149011612처럼 17자리로 찍히는데,
    반올림하면 0.1이 되어 텍스트 길이가 크게 준다. numpy 스칼라/배열도 함께 변환한다.
    """
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return compact(obj.tolist(), ndigits)
    if isinstance(obj, dict):
        return {k: compact(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [compact(v, ndigits) for v in obj]
    return obj


def iter_transitions(jsonl_path: str):
    """저장된 JSONL(.gz 또는 평문)을 읽어, 생략된 next_observation / next_state를
    다음 레코드에서 채운 완전한 transition dict를 순서대로 yield한다.

    (수집이 중간에 끊긴 파일이면 마지막 레코드에만 next_*가 없을 수 있다.)
    """
    opener = gzip.open if jsonl_path.endswith(".gz") else open
    with opener(jsonl_path, "rt", encoding="utf-8") as f:
        prev = None
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if prev is not None:
                # done 레코드는 next_*를 이미 갖고 있으므로 setdefault가 건드리지 않는다.
                prev.setdefault("next_observation", rec["observation"])
                prev.setdefault("next_state", rec["state"])
                yield prev
            prev = rec
        if prev is not None:
            yield prev


def has_saved_config(model_path) -> bool:
    folder = os.path.dirname(model_path or "")
    return bool(model_path) and all(
        os.path.isfile(os.path.join(folder, n)) for n in ("road_config.py", "mdp_config.py", "train.py"))


def apply_saved_config(model_path):
    """모델 폴더의 설정 스냅샷(test.py --saved-config와 동일)으로 이 모듈의 설정을 바꾼다.

    spawn 워커는 모듈을 새로 import하므로 워커마다 다시 호출해야 한다.
    """
    from utils.saved_configs import load_saved_configs
    global ROAD, EGO, TRAFFIC, SIMULATION, OBSERVATION, ACTION, HPARAMS, REWARD
    # Exported PPO folders contain BC snapshots. Use the actual PPO run's
    # snapshots to retain its event reward and lane-change penalty.
    run = ppo_run_directory(model_path)
    config_path = str(run / "model.pt") if run else model_path
    road, mdp, training = load_saved_configs(config_path)
    ROAD, EGO, TRAFFIC = road["ROAD"], road["EGO"], road["TRAFFIC"]
    SIMULATION, OBSERVATION, ACTION = mdp["SIMULATION"], mdp["OBSERVATION"], mdp["ACTION"]
    HPARAMS, REWARD = training["HPARAMS"], training["REWARD"]
    ACTION = deepcopy(ACTION)
    ACTION["lane_change_safety"] = dict(ACTION.get("lane_change_safety", {}),
                                       enabled=False, spawn_protect_only=False, spawn_protect=0.0)


# 교수님 기본 컬럼 + 팀 추가 event 컬럼. 중첩 값(list/dict)은 JSON 문자열로 저장한다.
CSV_COLUMNS = [
    "episode", "t", "observation", "action_raw", "action.accel_raw", "action.lane_change_raw",
    "action.lane_change", "action.lane_change_applied", "reward", "terminated", "truncated",
    "state", "detected_vehicles", "event.collided", "event.arrived", "event.lane_before",
    "event.lane_after", "event.red_light_violation", "event.wrong_lane",
    "event.collision_partner_id", "event.collision_partner_type",
    "event.collision_partner_speed", "event.ego_is_collider",
    "next_observation", "next_state",
    "episode_info", "new_routes",   # t=0 / 경로를 처음 본 행에만 값이 있음
]


def export_csv(jsonl_path: str, csv_path: str) -> int:
    """JSONL transition을 next_* 를 복원한 평평한 CSV로 바꾼다. 반환: 행 수."""
    import csv
    rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in iter_transitions(jsonl_path):
            flat = {**r, **{f"{k}.{s}": v for k in ("action", "event") for s, v in r[k].items()}}
            # list/dict/bool은 JSON 표기(true/false)로, null은 빈 칸으로 쓴다.
            w.writerow([json.dumps(flat.get(c), ensure_ascii=False, separators=(",", ":"))
                        if isinstance(flat.get(c), (list, dict, bool)) else flat.get(c)
                        for c in CSV_COLUMNS])
            rows += 1
    return rows


def write_metadata(args, env_kwargs, path):
    """실행 설정·코드 버전을 한 번 저장한다 (metadata.json)."""
    import subprocess

    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=BASE, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    # ponytail: 커밋 해시 + 미커밋 diff 원문. 완전 재현은 이 둘과 아래 설정으로 충분.
    meta = {
        "schema_version": 1,
        "step_length_s": float(SIMULATION["step_length"]),
        "signal_count": int(OBSERVATION.get("signal_count", 1)),
        "signal_visibility": float(OBSERVATION.get("signal_visibility", 100.0)),
        "lookahead_edges": int(OBSERVATION.get("lookahead_edges", 2)),
        "args": {k: v for k, v in vars(args).items()},
        "effective_config": env_kwargs,
        "code_snapshot": {"git_commit": git("rev-parse", "HEAD"),
                          "git_diff": git("diff", "HEAD")},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(compact(meta, 6), f, ensure_ascii=False, indent=1, default=str)


def collect(args, sumocfg, episodes, worker_id, jsonl_path, npz_path, tag=""):
    """SUMO 인스턴스 하나로 episodes(전역 에피소드 번호 목록)를 수집해 두 파일로 저장한다.

    --workers > 1이면 이 함수가 프로세스마다 하나씩 돌고, main()이 결과를 합친다.
    반환값: 수집한 transition 수.
    """
    worker_seed = (args.seed * 1000 + worker_id) % 2**32
    if has_saved_config(args.model):
        apply_saved_config(args.model)
    if args.workers > 1:
        # 프로세스마다 torch가 코어 수만큼 스레드를 띄우면 서로 뺏는다. 관측 1개씩
        # 추론이라 1스레드면 충분.
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass

    env = (SumoHighwayEnv if args.gui else LibsumoEnv)(
        cfg_path=sumocfg,
        road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD,
        traffic=getattr(args, "traffic_config", None) or TRAFFIC,
        gui=args.gui,
        gui_autostart=True,
    )
    if args.model:
        env.disable_ego_rule_barriers()
    if args.warmup_seconds is not None:
        env.warmup_seconds = args.warmup_seconds
    if args.randomize_departures:
        env._ego_depart_lane_cfg = "random"   # SUMO departLane=random (sumo_seed로 재현)
    signal_ids, end_ticks = (), ()
    if args.randomize_signals:
        import xml.etree.ElementTree as ET
        from env.parallel_sumo import _signal_schedule
        cfg = ET.parse(sumocfg).getroot()
        net = os.path.join(os.path.dirname(sumocfg), cfg.find("input/net-file").get("value"))
        signal_ids, end_ticks = _signal_schedule(ET.parse(net).getroot(), env.step_length, True)
    act = make_policy(args, env, worker_seed)
    # sumo_env가 SUMO --seed를 np.random(전역)에서 뽑는다. PPO 생성자가 전역 시드를
    # HPARAMS["seed"]로 되돌려 놓으므로 정책을 만든 "뒤에" 워커별로 다시 시드해야
    # 워커들이 같은 교통 패턴을 반복하지 않는다.
    np.random.seed(worker_seed)

    # NPZ는 학습에서 바로 쓰기 좋은 고정 크기 값만 모은다.
    obs_buf, next_obs_buf, action_raw_buf = [], [], []
    lane_cmd_buf, reward_buf = [], []
    term_buf, trunc_buf, episode_buf, step_buf = [], [], [], []
    # 스텝별 감지 차량 (x,y)는 한 줄로 이어 붙이고, 스텝 경계는 offsets로 기록한다.
    detected_xy_buf, detected_offsets = [], [0]

    total = 0
    try:
        # compresslevel 6: 9와 크기 차이는 거의 없고 훨씬 빠르다.
        # 더 줄이고 싶으면 gzip.open 대신 lzma.open(..., "wt") (.jsonl.xz) —
        # 느린 대신 보통 20~30% 추가 절감.
        with gzip.open(jsonl_path, "wt", encoding="utf-8", compresslevel=6) as jf:
            for ep in episodes:
                # sumo_seed(np.random 전역에서 뽑음)를 (--seed, 에피소드 번호)로만 정한다.
                # 워커 수·배치와 상관없이 같은 에피소드는 같은 교통으로 재현된다.
                np.random.seed([args.seed, ep])
                obs, _ = env.reset(seed=args.seed + ep)
                signal_offset_s = None
                if signal_ids:
                    # 학습(parallel_sumo)과 같은 방식: 신호 주기 안의 임의 시점에서 시작
                    offset = int(np.random.default_rng(args.seed + ep).integers(end_ticks[-1]))
                    phase = int(np.searchsorted(end_ticks, offset, side="right"))
                    for tls_id in signal_ids:
                        env._api.trafficlight.setPhase(tls_id, phase)
                        env._api.trafficlight.setPhaseDuration(
                            tls_id, float((end_ticks[phase] - offset) * env.step_length))
                    signal_offset_s = offset * env.step_length
                    obs = env._get_obs()
                state_t = env.get_privileged_state()
                # 에피소드 단위 정보: 첫 transition(t=0)에만 기록한다.
                episode_info = {
                    "route_id": env.ego_route_id,
                    "seed": int(args.seed + ep),
                    "sumo_seed": env.sumo_seed,
                    "sim_time_start_s": float(env._api.simulation.getTime()),
                    "signal_offset_s": signal_offset_s,
                }
                # 경로 정의: 이 에피소드에서 route_id를 처음 본 transition에만
                # {route_id: edges}로 기록한다. 조회 실패면 다음 스텝에 다시 시도.
                seen_routes = set()

                def new_routes(state):
                    found = {}
                    for v in state.get("vehicles", []):
                        rid = v.get("route_id")
                        if rid is None or rid in seen_routes:
                            continue
                        edges = env.get_vehicle_route(v["id"])
                        if edges is not None:
                            seen_routes.add(rid)
                            found[rid] = edges
                    return found

                done = False
                t = 0
                ep_return = 0.0

                while not done:
                    # 이번 관측 시점에 시야(W) 안에 들어온 차량들의 좌표.
                    # 관측 벡터와 별도로 저장하는 부가 데이터 (분석/시각화용).
                    detected = env.get_detected_vehicles()
                    routes_t = new_routes(state_t)   # 차량이 아직 있는 지금 조회
                    action = act(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    state_tp1 = env.get_privileged_state()
                    done = terminated or truncated

                    record = {
                        "episode": int(ep),
                        "t": t,
                        "observation": compact(obs, args.ndigits),
                        # 스칼라 몇 개는 용량에 영향이 없으니 반올림하지 않는다.
                        "action_raw": np.asarray(action, dtype=float).tolist(),
                        "action": {
                            "accel_raw": float(info["accel_raw"]),
                            "lane_change_raw": float(info["lane_change_raw"]),
                            "lane_change": int(info["lane_change"]),
                            "lane_change_applied": bool(info["lane_change_applied"]),
                        },
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "state": compact(state_t, args.ndigits),
                        # 감지 차량 좌표 (관측과 같은 시점 t 기준)
                        "detected_vehicles": compact(detected, args.ndigits),
                        "event": {
                            "collided": bool(info["collided"]),
                            "arrived": bool(info["arrived"]),
                            "lane_before": info["lane_before"],
                            "lane_after": info["lane_after"],
                            # 환경 판정 원본. wrong_lane에는 원인 불명 소실도 포함될 수 있다.
                            "red_light_violation": bool(info["red_light_violation"]),
                            "wrong_lane": bool(info["wrong_lane"]),
                            # 충돌 상대. 충돌하지 않은 행은 모두 null.
                            "collision_partner_id": info["collision_partner_id"],
                            "collision_partner_type": info["collision_partner_type"],
                            "collision_partner_speed": info["collision_partner_speed"],
                            "ego_is_collider": info["ego_is_collider"],
                        },
                    }
                    if t == 0:
                        record["episode_info"] = episode_info
                    if done:
                        # 중간 스텝의 next_*는 다음 레코드의 observation/state와 같으므로
                        # 에피소드 마지막 스텝에만 기록한다 (iter_transitions가 복원).
                        record["next_observation"] = compact(next_obs, args.ndigits)
                        record["next_state"] = compact(state_tp1, args.ndigits)
                        routes_t.update(new_routes(state_tp1))
                    if routes_t:
                        record["new_routes"] = routes_t
                    jf.write(json.dumps(record, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")

                    obs_buf.append(np.asarray(obs, dtype=np.float32))
                    next_obs_buf.append(np.asarray(next_obs, dtype=np.float32))
                    action_raw_buf.append(np.asarray(action, dtype=np.float32))
                    lane_cmd_buf.append(int(info["lane_change"]))
                    reward_buf.append(float(reward))
                    term_buf.append(bool(terminated))
                    trunc_buf.append(bool(truncated))
                    episode_buf.append(int(ep))
                    step_buf.append(t)
                    detected_xy_buf.extend((d["x"], d["y"]) for d in detected)
                    detected_offsets.append(len(detected_xy_buf))

                    total += 1
                    ep_return += reward
                    t += 1
                    obs = next_obs
                    state_t = state_tp1  # 스텝 끝의 상태 = 다음 스텝 시작의 상태

                print(f"{tag}episode {ep + 1:4d}/{args.episodes} | "
                      f"steps={t:4d} return={ep_return:8.3f}", flush=True)
    finally:
        env.close()

    np.savez_compressed(
        npz_path,
        observation=np.asarray(obs_buf, dtype=np.float32).reshape(total, -1),
        action_raw=np.asarray(action_raw_buf, dtype=np.float32).reshape(total, -1),
        lane_change=np.asarray(lane_cmd_buf, dtype=np.int8),
        reward=np.asarray(reward_buf, dtype=np.float32),
        next_observation=np.asarray(next_obs_buf, dtype=np.float32).reshape(total, -1),
        terminated=np.asarray(term_buf, dtype=np.bool_),
        truncated=np.asarray(trunc_buf, dtype=np.bool_),
        episode=np.asarray(episode_buf, dtype=np.int32),
        t=np.asarray(step_buf, dtype=np.int32),
        # 감지 차량 (x,y)를 전 스텝에 걸쳐 이어 붙인 (M, 2) float32.
        # 스텝 i의 좌표: detected_xy[detected_offsets[i]:detected_offsets[i+1]]
        detected_xy=np.asarray(detected_xy_buf, dtype=np.float32).reshape(-1, 2),
        detected_offsets=np.asarray(detected_offsets, dtype=np.int64),
    )
    return total


def merge_parts(parts, jsonl_path, npz_path):
    """워커별 부분 파일들을 하나로 합치고 부분 파일은 지운다."""
    # gzip은 멤버를 이어 붙인 파일도 유효하므로 바이트 그대로 연결한다.
    with open(jsonl_path, "wb") as out:
        for part_jsonl, _ in parts:
            with open(part_jsonl, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(part_jsonl)

    arrays, xy, offsets, base = {}, [], [], 0
    for _, part_npz in parts:
        with np.load(part_npz) as d:
            for key in d.files:
                if key not in ("detected_xy", "detected_offsets"):
                    arrays.setdefault(key, []).append(d[key])
            xy.append(d["detected_xy"])
            off = d["detected_offsets"]
            offsets.append(off[:-1] + base)   # 부분 파일의 offset을 전체 기준으로 옮김
            base += int(off[-1])
        os.remove(part_npz)
    merged = {key: np.concatenate(v) for key, v in arrays.items()}
    merged["detected_xy"] = np.concatenate(xy).reshape(-1, 2)
    merged["detected_offsets"] = np.concatenate(
        offsets + [np.asarray([base], dtype=np.int64)]).astype(np.int64)
    np.savez_compressed(npz_path, **merged)


def main():
    parser = argparse.ArgumentParser(description="SUMO POMDP trajectory collector")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--policy", choices=["random", "keep-lane", "ppo", "bc"], default=None,
                        help="행동 정책. 생략 시 --model이 있으면 파일을 보고 ppo/bc 자동 판별, "
                             "없으면 random")
    parser.add_argument("--model", default=None,
                        help="최종 BC 또는 PPO actor 형식의 model.pt. "
                             "주면 그 학습된 정책으로 수집한다")
    parser.add_argument("--stochastic", action="store_true",
                        help="ppo: 결정적(μ, argmax 차선) 대신 분포에서 샘플링해서 행동. "
                             "상태 다양성이 필요할 때")
    parser.add_argument("--device", default="cpu",
                        help="정책 추론 장치 (cpu/cuda). 관측 1개씩 추론이라 cpu로 충분")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true",
                        help="SUMO GUI로 수집. 기본은 headless")
    parser.add_argument("--workers", type=int, default=1,
                        help="병렬 SUMO 인스턴스 수. 코어 수만큼 주면 거의 그 배수로 빨라진다. "
                             "(--gui면 1로 고정)")
    parser.add_argument("--out-dir", default=os.path.join(BASE, "data"))
    parser.add_argument("--name", default=None,
                        help="출력 basename. 생략 시 날짜/시각 자동 생성")
    parser.add_argument("--randomize-signals", action=argparse.BooleanOptionalAction, default=None,
                        help="에피소드마다 신호 주기의 임의 시점에서 시작 (학습과 동일, episode_info에 기록). "
                             "--traffic-vph 사용 시 기본 ON (웜업 250s가 주기의 배수라 끄면 매번 같은 현시로 시작)")
    parser.add_argument("--randomize-departures", action="store_true",
                        help="출발 차선을 SUMO departLane=random으로 정함")
    parser.add_argument("--traffic-vph", type=float, default=None,
                        help="배경 교통과 신호 주기를 켜고 총 유입량(veh/h)을 지정 (최종 환경: 4700). "
                             "BC 학습 데이터 수집(collect_background_data.py)과 같은 도로를 만든다. "
                             "생략하면 모델 snapshot 설정 그대로(Phase 1: 전체 녹색·배경 차량 없음)")
    parser.add_argument("--warmup-seconds", type=float, default=None,
                        help="에피소드마다 자차 투입 전 배경 차량만 돌리는 시간(s). "
                             "--traffic-vph 사용 시 기본 250 (BC 학습 데이터 수집과 동일)")
    parser.add_argument("--csv", action="store_true",
                        help="수집 후 <수집명>.csv도 만든다 (next_* 복원, 중첩 값은 JSON 문자열)")
    parser.add_argument("--junction-check", action=argparse.BooleanOptionalAction, default=True,
                        help="교차로 내부 충돌 검사 (기본 ON, 최종 평가와 동일)")
    parser.add_argument("--ndigits", type=int, default=4,
                        help="JSONL의 observation/state/detected float 소수 자릿수 "
                             "(NPZ에는 영향 없음)")
    args = parser.parse_args()
    os.environ["SUMO_CHECK_JUNCTIONS"] = "1" if args.junction_check else "0"
    if args.episodes < 1 or args.workers < 1 or args.seed < 0:
        parser.error("--episodes와 --workers는 양수, --seed는 0 이상이어야 합니다.")
    if args.traffic_vph is not None and not 0 < args.traffic_vph < float("inf"):
        parser.error("--traffic-vph는 양수인 유한한 값이어야 합니다.")
    if args.warmup_seconds is None and args.traffic_vph is not None:
        args.warmup_seconds = 250.0
    if args.randomize_signals is None:
        args.randomize_signals = args.traffic_vph is not None
    if args.warmup_seconds is not None and not 0 <= args.warmup_seconds < float("inf"):
        parser.error("--warmup-seconds는 0 이상의 유한한 값이어야 합니다.")
    if args.policy in ("ppo", "bc") and args.model is None:
        parser.error(f"--policy {args.policy}에는 --model이 필요합니다.")
    if args.model is not None and not os.path.exists(args.model):
        parser.error(f"모델 파일이 없습니다: {args.model}")
    if args.policy is None:
        args.policy = detect_algorithm(args.model) if args.model else "random"
    if args.model:
        print(f"정책: {args.policy} ({args.model})"
              + (" [stochastic]" if args.stochastic and args.policy == "ppo" else ""))
    workers = 1 if args.gui else max(1, min(args.workers, args.episodes))
    if has_saved_config(args.model):
        apply_saved_config(args.model)
        print("모델 폴더의 저장된 도로/관측/보상 설정으로 수집")

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or datetime.now().strftime("pomdp_%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(args.out_dir, name + ".jsonl.gz")
    npz_path = os.path.join(args.out_dir, name + ".npz")

    # 도로 파일은 부모에서 한 번만 생성한다 (워커들이 동시에 쓰면 파일이 깨진다).
    road_dir = os.path.join(args.out_dir, name + ".sumo")
    traffic = TRAFFIC
    if args.traffic_vph is not None:
        # apply_saved_config 뒤에 import해야 snapshot의 도로·교통 설정을 쓴다.
        from pathlib import Path
        from collect_background_data import build_collection_road
        from env.collision_detection import configure_collision_gaps
        if os.path.exists(road_dir):
            parser.error(f"이미 있는 도로 폴더입니다: {road_dir}")
        sumocfg, traffic, _ = build_collection_road(Path(road_dir), args.traffic_vph)
        configure_collision_gaps(sumocfg, ego_gap_m=EGO.get("collision_gap_m"),
                                 background_gap_m=traffic.get("collision_gap_m"))
        sumocfg = str(sumocfg)
        print(f"배경 교통 {args.traffic_vph:g} veh/h, 신호 주기 ON으로 수집")
    else:
        if os.path.exists(road_dir):
            parser.error(f"이미 있는 도로 폴더입니다: {road_dir}")
        sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, road_dir)
    args.traffic_config = traffic  # 워커에 전달 (spawn 워커는 snapshot TRAFFIC을 다시 읽는다)
    from env import road_config
    write_metadata(args, {
        "ROAD": ROAD, "EGO": EGO, "TRAFFIC": traffic, "NOHYEONG": road_config.NOHYEONG,
        "ROUTE_ONLY_EXPERIMENT": road_config.ROUTE_ONLY_EXPERIMENT,
        "SIGNAL_EXPERIMENT": road_config.SIGNAL_EXPERIMENT,
        "SIMULATION": SIMULATION, "OBSERVATION": OBSERVATION, "ACTION": ACTION, "REWARD": REWARD,
    }, os.path.join(args.out_dir, name + ".metadata.json"))

    t0 = time.time()
    if workers == 1:
        total = collect(args, sumocfg, list(range(args.episodes)), 0,
                        jsonl_path, npz_path)
    else:
        chunks = [c.tolist() for c in np.array_split(np.arange(args.episodes), workers)]
        parts = [(os.path.join(args.out_dir, f".{name}.w{w}.jsonl.gz"),
                  os.path.join(args.out_dir, f".{name}.w{w}.npz"))
                 for w in range(workers)]
        jobs = [(args, sumocfg, chunks[w], w, parts[w][0], parts[w][1], f"[w{w}] ")
                for w in range(workers)]
        # spawn: 워커마다 깨끗한 프로세스에서 SUMO를 하나씩 띄운다.
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            total = sum(pool.starmap(collect, jobs))
        merge_parts(parts, jsonl_path, npz_path)
    elapsed = time.time() - t0

    mb = lambda p: os.path.getsize(p) / 2**20  # noqa: E731
    print("\n수집 완료")
    print(f"  transitions: {total}  ({elapsed:.1f}s, {total / max(elapsed, 1e-9):.0f} step/s, "
          f"workers={workers})")
    print(f"  JSONL(full POMDP + privileged state): {jsonl_path} ({mb(jsonl_path):.1f} MB)")
    print(f"  NPZ(training-friendly):               {npz_path} ({mb(npz_path):.1f} MB)")
    print(f"  metadata:                             {os.path.join(args.out_dir, name)}.metadata.json")
    if args.csv:
        csv_path = os.path.join(args.out_dir, name + ".csv")
        rows = export_csv(jsonl_path, csv_path)
        print(f"  CSV:                                  {csv_path} ({rows} rows, {mb(csv_path):.1f} MB)")


if __name__ == "__main__":
    main()
