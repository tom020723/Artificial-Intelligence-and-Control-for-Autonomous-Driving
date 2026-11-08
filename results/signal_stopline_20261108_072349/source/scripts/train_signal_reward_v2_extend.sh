#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SUMO_RUN_PREFIX=signal_reward_v2_extend
exec bash "$PROJECT_DIR/scripts/train_signal.sh" \
  --init-model "$PROJECT_DIR/results/signal_reward_v2_20261107_143114/model.pt"
