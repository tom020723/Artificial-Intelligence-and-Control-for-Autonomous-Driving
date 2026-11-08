#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SUMO_RUN_PREFIX=exp8_destination_extend
exec bash "$PROJECT_DIR/scripts/train_exp8_destination.sh" \
  --init-model "$PROJECT_DIR/results/exp8_destination_20261108_163247/model.pt"
