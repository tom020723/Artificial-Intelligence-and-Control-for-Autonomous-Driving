"""Validate a frozen Phase 1 observation experiment, optionally with a GPU smoke run."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    manifest = json.loads((source / "experiment.json").read_text())
    for name, digest in manifest["source_hashes"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    sys.path.insert(0, str(source))

    import numpy as np
    import torch
    from env import road_builder
    from env.road_config import ROAD, EGO, TRAFFIC, NOHYEONG, ROUTE_ONLY_EXPERIMENT, SIGNAL_EXPERIMENT
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    from algorithms.ppo import PPO
    from utils.buffer import RolloutBuffer
    from train import TOTAL_TIMESTEPS, REWARD, HPARAMS, RESULTS_DIR, RUN_NAME

    assert TOTAL_TIMESTEPS == manifest["target_steps"] == 300_000
    assert str(Path(RESULTS_DIR) / RUN_NAME) == manifest["run_dir"]
    assert ROUTE_ONLY_EXPERIMENT and SIGNAL_EXPERIMENT and NOHYEONG["all_signals_green"]
    assert not TRAFFIC["background_enabled"] and TRAFFIC["vehs_per_hour"] == 0
    assert not TRAFFIC["prefill"]["enabled"]
    assert all(d.get("vph", 0) == 0 for d in NOHYEONG["ring_segments"] + NOHYEONG["loop_routes"])
    assert not {"A2_in", "A2_out"}.intersection(NOHYEONG["no_lane_change_edges"])
    assert REWARD["lane_change_penalty"] == 0.02 and REWARD["invalid_action_penalty"] == 0.2
    assert OBSERVATION["lookahead_connectivity_enabled"]
    assert OBSERVATION["destination_enabled"] == manifest["destination_enabled"]
    assert OBSERVATION["signal_enabled"] and OBSERVATION["signal_count"] == manifest["signal_count"]
    assert OBSERVATION["connectivity_half_width"] == 4
    assert HPARAMS["seed"] == 0 and HPARAMS["hidden_sizes"] == (256, 256)
    assert HPARAMS["n_steps"] == 2048
    print(f"Verified frozen variant {manifest['variant']}: obs_dim={manifest['observation_dim']}, "
          f"goal={manifest['destination_enabled']}, signals={manifest['signal_count']}, "
          "fresh initialization, target=300000", flush=True)
    if args.verify_only:
        return

    folder = source / "preflight"
    folder.mkdir(exist_ok=True)
    cfg = road_builder.build(ROAD, EGO, TRAFFIC, str(folder / "sumo"))
    env = SumoHighwayEnv(cfg, ROAD, EGO, SIMULATION, OBSERVATION, ACTION, REWARD, traffic=TRAFFIC)
    try:
        expected_dim = manifest["observation_dim"]
        assert env.obs_dim == expected_dim
        env._choose_ego_route = lambda: "ego_d_A2_A3"
        obs, _ = env.reset(seed=0)
        assert obs.shape == (expected_dim,) and env.observation_space.contains(obs)
        reference_env = SumoHighwayEnv(
            cfg, ROAD, EGO, SIMULATION,
            {**OBSERVATION, "destination_enabled": True, "signal_count": 2},
            ACTION, REWARD, traffic=TRAFFIC)
        full = reference_env._get_obs()
        assert full.shape == (59,)
        pieces = [full[:35 + 5 * manifest["signal_count"]], full[45:54]]
        if manifest["destination_enabled"]:
            pieces.append(full[54:59])
        np.testing.assert_array_equal(obs, np.concatenate(pieces))

        agent = PPO(obs_dim=expected_dim, act_dim=env.action_space.shape[0],
                    buffer=RolloutBuffer(), **dict(HPARAMS, n_steps=64, n_epochs=1))
        assert agent.device.type == "cuda"
        before = {k: v.detach().clone() for k, v in agent.policy.state_dict().items()}
        agent.learn(env, total_timesteps=64)
        assert agent.num_timesteps == 64
        assert all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
        assert any(not torch.equal(before[k], v) for k, v in agent.policy.state_dict().items())
        result = dict(status="PASS", variant=manifest["variant"], observation_dim=expected_dim,
                      destination_enabled=env.destination_enabled, signal_count=env.signal_count,
                      ablation_slices_verified=True, device=str(agent.device),
                      smoke_steps=64, smoke_epochs=1, smoke_weights_discarded=True,
                      production_target_steps=300000, expected_actual_steps=301056,
                      from_scratch=True, checkpoint_loaded=None)
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
