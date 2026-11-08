#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export TZ=Asia/Seoul
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SUMO_RUN_PREFIX="${SUMO_RUN_PREFIX:-exp8_destination}"

/home/lair15/anaconda3/bin/conda run -n sumo python -c '
from env.road_config import NOHYEONG, TRAFFIC, ROUTE_ONLY_EXPERIMENT, SIGNAL_EXPERIMENT
from env.mdp_config import OBSERVATION
from train import TOTAL_TIMESTEPS
assert ROUTE_ONLY_EXPERIMENT and SIGNAL_EXPERIMENT
assert NOHYEONG["all_signals_green"], "Experiment 8 requires all signals green"
assert not TRAFFIC["background_enabled"] and TRAFFIC["vehs_per_hour"] == 0
assert not TRAFFIC["prefill"]["enabled"]
assert all(d.get("vph", 0) == 0 for d in NOHYEONG["ring_segments"] + NOHYEONG["loop_routes"])
assert OBSERVATION["signal_enabled"] and OBSERVATION["signal_count"] == 2
assert OBSERVATION["connectivity_half_width"] == 4 and OBSERVATION["destination_enabled"]
assert tuple(OBSERVATION["destination_arms"]) == ("A1", "A2", "A3", "A4", "A5")
assert TOTAL_TIMESTEPS == 200_000
print("Experiment 8: all green, no background, connectivity +/-4 lanes, destination one-hot, 50 observations, 200000 target steps")
'
exec /home/lair15/anaconda3/bin/conda run --no-capture-output -n sumo python -u train.py "$@"
