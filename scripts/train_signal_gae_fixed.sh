#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# 실험 2의 설정/초기화를 그대로 사용하고 저장 폴더 이름만 구분한다.
export SUMO_RUN_PREFIX=signal_gae_fixed
exec bash "$PROJECT_DIR/scripts/train_signal.sh"
