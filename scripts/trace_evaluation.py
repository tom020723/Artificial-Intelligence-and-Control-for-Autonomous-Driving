"""Replay one episode deterministically and record a per-step trace CSV.

Fixes a single (route, departure lane, signal offset) and steps the saved
policy deterministically, recording observation, action, lane/signal state,
reward (total and per-component), and termination reason per step.

Usage:
    python scripts/trace_evaluation.py results/<run> --route ego_r_RING_A2_A4_A2_A4 \
        --depart-lane 1 --signal-offset 10.0 --output traces/case.csv
"""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from algorithms.ppo import PPO
from env import road_builder
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo
from env.parallel_traffic import TrafficCurriculum
from test import load_saved_configs

STEP = 0.5

# Reward component keys present in step info; the total "reward" is recorded
# separately so it is never overwritten by this list.
REWARD_COMPONENT_KEYS = (
    "early_lane_change_reward", "green_pass_reward", "green_stop_penalty",
    "red_stop_penalty", "red_stop_close_penalty", "red_park_penalty",
    "red_stop_clearance_reward", "red_stop_progress_reward",
    "red_speed_tracking_reward", "red_stopped_brake_penalty",
    "red_safe_zone_first_reward", "red_safe_zone_hold_reward",
)


def _reward_component_values(info):
    """Return {name: value} for reward/penalty components, excluding the total."""
    return {key: float(info[key]) for key in REWARD_COMPONENT_KEYS if key in info}


def _signal_state(info):
    signals = info.get("observed_signals", ())
    if not signals:
        return "", None
    signal = signals[0]
    return signal["state"], signal["distance"]


def _outcome(info, terminated, truncated):
    for key in ("red_light_violation", "collided", "arrived", "wrong_lane"):
        if info.get(key):
            return key
    if truncated:
        return "timeout"
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="학습 결과 폴더 또는 model.pt")
    parser.add_argument("--route", required=True, help="고정 경로 ID (예: ego_d_A1_A3)")
    parser.add_argument("--depart-lane", type=int, required=True, help="출발 차선 번호 (0부터)")
    parser.add_argument("--signal-offset", type=float, required=True,
                        help="에피소드 시작 시 신호 주기 내 시점(초)")
    parser.add_argument("--output", type=Path, default=None,
                        help="출력 CSV 경로 (기본: traces/<route>_<lane>_<offset>.csv)")
    args = parser.parse_args()

    run = args.run.resolve()
    if run.is_file() and run.name == "model.pt":
        run = run.parent
    model = run / "model.pt"
    if not model.is_file():
        parser.error(f"모델을 찾을 수 없습니다: {model}")

    road, mdp, train = load_saved_configs(str(model))
    saved = json.loads((run / "parallel_config.json").read_text()) \
        if (run / "parallel_config.json").is_file() else {}
    settings = saved.get("traffic_curriculum")
    curriculum = TrafficCurriculum(**settings) if settings else None

    import tempfile
    with tempfile.TemporaryDirectory(prefix="sumo-trace-") as directory:
        root = Path(directory)
        config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"],
                                    str(root / "base"))
        non_ring, ring = read_ego_route_groups(root / "base/highway.ego.rou.xml")
        routes = non_ring + ring
        if args.route not in routes:
            parser.error(f"존재하지 않는 고유 경로 ID: {args.route} (총 {len(routes)}개)")

        kwargs = dict(cfg_path=config, road=road["ROAD"], ego=road["EGO"],
                      mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
                      action=mdp["ACTION"], reward=train["REWARD"], traffic=road["TRAFFIC"])
        env = ParallelSumo(1, kwargs, root / "parallel", seed=0, randomize_signals=True,
                           randomize_departures=True,
                           independent_episode_rng=True,
                           traffic_curriculum=curriculum,
                           fixed_traffic_vph=None)
        if env.prototype.step_length != STEP:
            raise ValueError("trace requires a 0.5 s simulation step")
        agent = PPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim,
                    **(train["HPARAMS"] | {"device": "cpu"}))
        agent.policy.load_state_dict(torch.load(model, map_location="cpu", weights_only=True))
        agent.policy.eval()

        # Fix the departure to exactly the requested lane.
        env._find_departures()
        lanes = tuple(item for item in env.departures[args.route]
                      if item[2] == args.depart_lane)
        if not lanes:
            parser.error(f"{args.route}: 출발 차선 {args.depart_lane} 사용 불가")
        env.departures[args.route] = lanes

        tick = args.signal_offset / STEP
        if not (0 <= tick < env.signal_end_ticks[-1] and float(tick).is_integer()):
            parser.error("--signal-offset은 신호 주기 안의 0.5초 배수여야 합니다.")

        obs = env.reset(routes=[args.route], signal_ticks=[int(tick)])
        obs = obs[0]  # single-replica batch -> single observation
        ego = env.egos[0]

        def current_state():
            try:
                edge = env.prototype._api.vehicle.getRoadID(ego.ego_id)
                lane = env.prototype._api.vehicle.getLaneIndex(ego.ego_id)
                position = env.prototype._api.vehicle.getLanePosition(ego.ego_id)
                speed = env.prototype._api.vehicle.getSpeed(ego.ego_id)
            except Exception:
                return "", None, None, None
            return edge.removeprefix(ego.prefix("")), lane, position, speed

        rows = []
        step = 0
        terminated = truncated = False
        while True:
            edge, lane, position, speed = current_state()
            action = agent.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            transitions = list(env.step_evaluation([0], [action]))
            next_obs, reward, terminated, truncated, info = transitions[0]
            state, dist = _signal_state(info)
            rows.append({
                "step": step,
                "time_s": round((step + 1) * STEP, 2),
                "edge": edge,
                "lane_index": lane if lane is not None else "",
                "position_m": round(position, 3) if position is not None else "",
                "speed_m_s": round(speed, 3) if speed is not None else "",
                "observation": json.dumps([round(float(x), 6) for x in obs]),
                "action_accel_raw": round(float(action[0]), 6),
                "action_lane_raw": round(float(action[1]), 6),
                "lane_change_cmd": info.get("lane_change", 0),
                "lane_change_applied": bool(info.get("lane_change_applied")),
                "signal_state": state,
                "signal_distance_m": dist,
                "red_signal_distance_m": info.get("red_signal_distance"),
                "reward": reward,
                "reward_components": json.dumps(_reward_component_values(info)),
                "outcome": _outcome(info, terminated, truncated),
            })
            step += 1
            if terminated or truncated:
                break
            obs = next_obs

        env.close()

    output = args.output or Path("traces") / f"{args.route}_{args.depart_lane}_{args.signal_offset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "time_s", "edge", "lane_index", "position_m", "speed_m_s",
                  "observation", "action_accel_raw", "action_lane_raw", "lane_change_cmd",
                  "lane_change_applied", "signal_state", "signal_distance_m",
                  "red_signal_distance_m", "reward", "reward_components", "outcome"]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    ep_return = sum(row["reward"] for row in rows)
    outcome = rows[-1]["outcome"]
    print(f"{args.route} | lane {args.depart_lane} | offset {args.signal_offset}s | "
          f"{len(rows)} steps | outcome={outcome} | return={ep_return:.4f} | {output}")


if __name__ == "__main__":
    main()
