#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export TZ=Asia/Seoul
export PYTHONUNBUFFERED=1
# Small MLP updates are faster without excessive CPU thread contention.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

/home/lair15/anaconda3/bin/conda run -n sumo python -c 'from env.road_config import SIGNAL_EXPERIMENT; assert SIGNAL_EXPERIMENT, "Enable SIGNAL_EXPERIMENT first"'
exec /home/lair15/anaconda3/bin/conda run --no-capture-output -n sumo python -u train.py "$@"
