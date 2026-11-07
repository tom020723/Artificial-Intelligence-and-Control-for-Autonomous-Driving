#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SUMO_RUN_PREFIX=signal_reward_v2
exec bash "$PROJECT_DIR/scripts/train_signal.sh"
