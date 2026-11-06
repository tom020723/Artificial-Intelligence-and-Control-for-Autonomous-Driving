#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export TZ=Asia/Seoul
export PYTHONUNBUFFERED=1
# Small MLP updates are faster without excessive CPU thread contention.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec /home/lair15/anaconda3/bin/conda run --no-capture-output -n sumo python -u train.py
