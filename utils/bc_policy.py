"""Load a saved BC/PPO actor, and build the settings folder an evaluation runs it with."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import torch

from algorithms.ppo import PPO
from env.parallel_traffic import TrafficCurriculum


def load_actor(checkpoint, hparams, device, *, obs_dim=51):
    actor = PPO(obs_dim=obs_dim, act_dim=2, **(hparams | {"device": device})).policy
    actor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return actor


def evaluation_context(run, context, data_dir=None, *, max_vph=3000):
    """Copy a BC run's settings into `context` with its collection's traffic and observation.

    The road, observation, and simulation come from the run's dataset metadata;
    the ego departs from the center lane and executes its lane commands without
    any rule-based restriction (the expert's entry restriction is only recorded).
    Returns (evaluation_config, traffic_curriculum, dataset_directory).
    """
    config = json.loads((run / "config.json").read_text())
    if config.get("algorithm") != "supervised_bc_on_ppo_hybrid_actor":
        raise ValueError("Use a BC result folder (scripts/train_bc_shards.py); legacy regression BC is unsupported")
    data_dir = Path(data_dir) if data_dir else (
        Path(config["dataset_directory"]) if config.get("dataset_directory") else Path(config["data"]).parent)
    collection_file = data_dir / "config.json"
    if not collection_file.is_file() and (run / "collection_config.json").is_file():
        # The dataset is not on this machine: use the copy of its config saved beside the model.
        collection_file = run / "collection_config.json"
    collection = json.loads(collection_file.read_text())
    if collection.get("observation", {}).get("layout") not in ("bc43", "bc49"):
        raise ValueError("The dataset must use the bc43 or bc49 observation layout")
    expert_spawn_protect = float(collection.get("expert_control_timing", {}).get("spawn_protect", 0.0))
    context.mkdir(parents=True, exist_ok=False)
    for name in ("model.pt", "road_config.py", "mdp_config.py", "train.py"):
        shutil.copy2(run / name, context / name)
    # The run's road geometry stays; collection restores cyclic signals and the background traffic.
    with (context / "road_config.py").open("a") as file:
        file.write("\n# BC evaluation context (not a PPO-trained run).\n")
        file.write("ROUTE_ONLY_EXPERIMENT = False\nSIGNAL_EXPERIMENT = True\n")
        file.write("TRAFFIC = " + repr(collection["traffic"]) + "\n")
        file.write("NOHYEONG['all_signals_green'] = False\nNOHYEONG['warmup_seconds'] = 0\n")
        file.write("for demand, vph in zip(NOHYEONG['ring_segments'] + NOHYEONG['loop_routes'], "
                   "_RING_BACKGROUND_VPH, strict=True):\n    demand['vph'] = vph\n")
        file.write("\nEGO = dict(EGO, depart_lane='center')\n")
    with (context / "mdp_config.py").open("a") as file:
        file.write("\nOBSERVATION = " + repr(collection["observation"]) + "\n")
        file.write("SIMULATION = " + repr(collection["simulation"]) + "\n")
        file.write("\n# The ego executes its lane commands without rule-based restrictions.\n")
        file.write("from copy import deepcopy as _copy_action\nACTION = _copy_action(ACTION)\n"
                   "ACTION['lane_change_safety'] = dict(ACTION.get('lane_change_safety', {}), "
                   "enabled=False, spawn_protect_only=False, spawn_protect=0.0)\n")
        file.write("\nACTION['lane_change_safety'].update("
                   "enabled=False, spawn_protect_only=False, spawn_protect=0.0)\n")
    with (context / "train.py").open("a") as file:
        file.write("\nHPARAMS.update(" + repr(dict(hidden_sizes=tuple(config["hidden_sizes"]),
                                                  activation=config["activation"], policy_type="hybrid")) + ")\n")
    curriculum = TrafficCurriculum(max_vph=max_vph, total_vph=True, snapshot_warmup_seconds=250,
                                   ring_reference_vph=collection["traffic"]["vehs_per_hour"])
    saved = dict(evaluation_only=True, bc_spawn_protect_m=0.0, expert_spawn_protect_m=expert_spawn_protect,
                 training_algorithm=config["algorithm"], traffic_curriculum=asdict(curriculum),
                 randomize_departures=False, independent_episode_rng=True, seed=0,
                 fixed_departure="center", spawn_protect_m=0.0, target_lane_safety_enabled=False)
    (context / "evaluation_config.json").write_text(json.dumps(saved, indent=2) + "\n")
    return saved, curriculum, data_dir.resolve()
