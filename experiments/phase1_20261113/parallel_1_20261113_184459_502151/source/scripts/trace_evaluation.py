"""Trace one deterministic evaluation episode to CSV."""

import argparse
import csv
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.ppo import PPO
from env import road_builder
from env.ego_routes import read_ego_route_groups
from env.parallel_sumo import ParallelSumo
from test import load_saved_configs


TRACE_FIELDS = (
    "step", "route", "signal_offset_s", "edge_before", "lane_before", "lane_after",
    "speed", "signal_state", "signal_distance", "stop_clearance_m",
    "route_connected_before", "accel_raw", "lane_change_raw", "lane_change",
    "lane_change_applied", "lane_change_rejected", "reward",
    "red_stop_penalty", "red_stop_clearance_reward", "red_stop_progress_reward",
    "red_speed_tracking_reward", "red_stopped_brake_penalty",
    "red_safe_zone_first_reward", "red_safe_zone_hold_reward", "green_stop_penalty",
    "outcome",
)


def _reward_component_values(info):
    """Reward term values from info, excluding the total-reward key.

    The total reward is taken directly from env.step's return value and must
    not be overwritten by a None-valued info entry of the same name.
    """
    return {key: info.get(key) for key in TRACE_FIELDS
            if key != "reward"
            and (key.endswith("penalty") or key.endswith("reward"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("route")
    parser.add_argument("offset", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--randomize-departures", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    model = run / "model.pt"
    road, mdp, train = load_saved_configs(str(model))
    saved = json.loads((run / "parallel_config.json").read_text())
    with TemporaryDirectory(prefix="sumo-trace-") as directory:
        root = Path(directory)
        config = road_builder.build(road["ROAD"], road["EGO"], road["TRAFFIC"],
                                    str(root / "base"))
        non_ring, ring = read_ego_route_groups(root / "base/highway.ego.rou.xml")
        routes = non_ring + ring
        if args.route not in routes:
            parser.error(f"unknown route: {args.route}")
        route_index = routes.index(args.route)
        kwargs = dict(cfg_path=config, road=road["ROAD"], ego=road["EGO"],
                      mdp_sim=mdp["SIMULATION"], mdp_obs=mdp["OBSERVATION"],
                      action=mdp["ACTION"], reward=train["REWARD"],
                      traffic=road["TRAFFIC"])
        env = ParallelSumo(
            1, kwargs, root / "worker", seed=0, randomize_signals=True,
            randomize_departures=saved.get("randomize_departures", False)
            or args.randomize_departures,
            independent_episode_rng=saved.get("independent_episode_rng", False),
        )
        agent = PPO(obs_dim=env.prototype.obs_dim, act_dim=env.prototype.act_dim,
                    **(train["HPARAMS"] | {"device": "cpu"}))
        agent.policy.load_state_dict(torch.load(model, map_location="cpu", weights_only=True))
        obs = env.reset(
            routes=[args.route], signal_ticks=[round(args.offset / 0.5)],
            replica_seeds=[np.random.SeedSequence([0, route_index, round(args.offset / 0.5)])],
        )
        rows = []
        step = 0
        with torch.inference_mode():
            while True:
                action = agent.predict(obs, deterministic=True)
                transition = next(env.step_evaluation([0], [action]))
                obs, reward, terminated, truncated, info = transition
                signals = info.get("observed_signals", ())
                rows.append({
                    "step": step,
                    "route": args.route,
                    "signal_offset_s": args.offset,
                    "edge_before": info.get("edge_before"),
                    "lane_before": info.get("lane_before"),
                    "lane_after": info.get("lane_after"),
                    "speed": info.get("speed"),
                    "signal_state": signals[0]["state"] if signals else None,
                    "signal_distance": signals[0]["distance"] if signals else None,
                    "stop_clearance_m": info.get("stop_clearance_m"),
                    "route_connected_before": info.get("route_connected_before"),
                    "accel_raw": info.get("accel_raw"),
                    "lane_change_raw": info.get("lane_change_raw"),
                    "lane_change": info.get("lane_change"),
                    "lane_change_applied": info.get("lane_change_applied"),
                    "lane_change_rejected": info.get("lane_change_rejected"),
                    "reward": reward,
                    **_reward_component_values(info),
                    "outcome": ("arrival" if info.get("arrived") else
                                "collision" if info.get("collided") else
                                "red_light" if info.get("red_light_violation") else
                                "wrong_lane" if info.get("wrong_lane") else
                                "timeout" if truncated else None),
                })
                step += 1
                if terminated or truncated:
                    break
                obs = np.asarray(obs)
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    get_context("spawn")
    main()
