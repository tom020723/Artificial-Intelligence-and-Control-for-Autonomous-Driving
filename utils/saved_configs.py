"""Restore the settings snapshot (road_config.py, mdp_config.py, train.py) saved beside a model."""

import os
import runpy

from env import road_builder


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
