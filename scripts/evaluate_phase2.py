"""Phase 2 모델 평가 — 신호 정상 작동 조건에서 27개 경로 × seed 평가.

Phase 1용 evaluate_variant.py와 달리 all_signals_green이 아닌 정상 신호
주기에서 평가한다. 신호 위반/청신호 통과/적신호 정차 지표를 집계한다.

Usage (프로젝트 루트에서):
    python scripts/evaluate_phase2.py results/signal_XXXX/model.pt --saved-config
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

# scripts/ 아래에서 실행해도 프로젝트 루트의 env/algorithms 모듈을 찾도록 추가.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path, help='model.pt 경로')
parser.add_argument('--seeds', type=int, default=5)
args = parser.parse_args()

MODEL = args.model.resolve()
RUN = MODEL.parent
OUT = RUN / 'analysis'
OUT.mkdir(exist_ok=True)

# 학습 당시 저장된 설정(source/)을 최우선으로 사용해, 현재 env/ 설정 파일이
# 바뀌었어도 그 run의 도로/관측/보상/배경차 조건 그대로 평가한다.
# (evaluate_variant.py와 동일한 방식 — run 폴더의 source/를 sys.path 맨 앞에 삽입)
_SRC = RUN / 'source'
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import traci
from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.road_builder import build
from env.sumo_env import SumoHighwayEnv
from train import HPARAMS, REWARD
from algorithms.ppo import PPO
from utils.evaluator import evaluate_policy

assert NOHYEONG.get('all_signals_green', False) is False, \
    '이 스크립트는 정상 신호 주기(Phase 2) 평가용입니다. all_signals_green=False 이어야 합니다.'

cfg = build(ROAD, EGO, TRAFFIC, str(OUT / 'sumo'))
routes = {r.get('id'): r.get('edges').split() for r in
          ET.parse(str(OUT / 'sumo' / 'highway.ego.rou.xml')).getroot().findall('route')
          if r.get('id').startswith(('ego_d_', 'ego_r_'))}
print(f'경로 {len(routes)}개 평가 (seeds={args.seeds})', flush=True)

env = SumoHighwayEnv(cfg_path=cfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION,
                     mdp_obs=OBSERVATION, action=ACTION, reward=REWARD,
                     traffic=TRAFFIC, gui=False)

agent = PPO(obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0], **HPARAMS)
agent.load(str(MODEL))
agent.policy.eval()

rows = []
try:
    for route in sorted(routes):
        env._choose_ego_route = lambda route=route: route
        for seed in range(args.seeds):
            metrics = evaluate_policy(agent, env, n_episodes=1,
                                      deterministic=True, seed=seed)
            # 종료 원인 추정: 마지막 info로는 안 되므로 지표로 추론
            if metrics.red_light_violation_rate:
                status = 'red_light'
            elif metrics.collision_rate:
                status = 'collision'
            elif metrics.wrong_lane_rate:
                status = 'wrong_lane'
            elif metrics.success_rate:
                status = 'arrival'
            elif metrics.timeout_rate:
                status = 'timeout'
            else:
                status = 'unknown'
            row = dict(route=route, seed=seed, status=status, **metrics.as_dict())
            rows.append(row)
            print(f"{route} seed={seed}: {status}, return={metrics.ep_return:.2f}, "
                  f"steps={metrics.ep_length:.0f}, green={metrics.green_passes:.0f}, "
                  f"red_viol={metrics.red_light_violation_rate:.0f}", flush=True)
finally:
    env.close()

    # 결과 저장
    with (OUT / 'phase2_evaluation.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['route'])
        w.writeheader()
        w.writerows(rows)

    def group(selected):
        if not selected:
            return dict(episodes=0)
        return dict(episodes=len(selected),
                    statuses=dict(Counter(r['status'] for r in selected)),
                    **{k: float(np.mean([r[k] for r in selected])) for k in
                       ['success_rate', 'wrong_lane_rate', 'timeout_rate',
                        'collision_rate', 'red_light_violation_rate',
                        'ep_return', 'lane_changes', 'mean_speed', 'ep_length',
                        'green_passes', 'red_stop_cost']})

    summary = dict(
        model=str(MODEL),
        model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        observation_dim=env.obs_dim,
        signal_count=env.signal_count,
        all_green=NOHYEONG.get('all_signals_green', False),
        background=TRAFFIC.get('background_enabled', False),
        seeds=list(range(args.seeds)),
        groups=dict(all=group(rows),
                    direct_od=group([r for r in rows if r['route'].startswith('ego_d_')]),
                    explicit_ring=group([r for r in rows if r['route'].startswith('ego_r_')])))
    (OUT / 'phase2_evaluation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
