#!/usr/bin/env bash
# Expert collection -> BC training -> BC evaluation -> PPO fine-tuning -> PPO evaluation.
#
#   bash scripts/reproduce_pipeline.sh                 # full run (about 4 hours)
#   SMOKE=1 bash scripts/reproduce_pipeline.sh         # every stage at minimum scale (about 3 minutes)
#
# Environment:
#   OUT      output root (default results/reproduce_<timestamp>, or results/reproduce_smoke_<timestamp>)
#   PY       Python for stages 0-3 (default: python)
#   PPO_PY   Python for stages 4-5, PPO training and evaluation (default: $PY)
#            The final models were built with torch 2.7.0+cu128 for 0-3 and torch
#            2.14.1+cu130 for 4-5; other torch/CUDA builds change GPU results in
#            the last bits, which makes PPO training diverge (docs/REPRODUCE.md).
#   SUMO_HOME SUMO install whose tools/ go on PYTHONPATH (default /usr/share/sumo)
#   WORKERS  collection processes (default 12)
#   SUMO_CHECK_JUNCTIONS  1 = SUMO also detects collisions inside junctions in collection and
#            PPO training too (default 0, as when the final models were built). Evaluation
#            always counts junction collisions unless run with --no-junction-check.
#
# Each finished stage leaves OUT/.done_<stage>; rerunning with the same OUT resumes
# after the last finished stage. docs/REPRODUCE.md describes every stage.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SMOKE="${SMOKE:-0}"
PY="${PY:-python}"
PPO_PY="${PPO_PY:-$PY}"
WORKERS="${WORKERS:-12}"
if [[ "$SMOKE" == 1 ]]; then
  OUT="${OUT:-results/reproduce_smoke_$(date +%Y%m%d_%H%M%S)}"
else
  OUT="${OUT:-results/reproduce_$(date +%Y%m%d_%H%M%S)}"
fi
export PYTHONPATH="$REPO:${SUMO_HOME:-/usr/share/sumo}/tools${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SUITE="$OUT/suite_fixed_routes"
TEST_SUITE="$OUT/suite_diverse_test"
VALIDATION_SUITE="$OUT/suite_diverse_validation"
BC="$OUT/BC"
PPO="$OUT/PPO"

if [[ "$SMOKE" == 1 ]]; then
  COLLECT=(--train-quota 2 --validation-quota 1 --epochs 1 --eval-smoke 1)
  TEST_SEEDS=(52101 52102)          # prepare needs at least two seeds
  VALIDATION_SEEDS=(62101 62102)
  EVAL=(--smoke 1)
  FINETUNE=(--iterations 10 --critic-warmup 1 --envs 16 --workers 1)
  ITERATIONS=(10)
else
  COLLECT=(--train-quota 19000 --validation-quota 1000 --epochs 100)
  TEST_SEEDS=($(seq 52101 52125))
  VALIDATION_SEEDS=($(seq 62101 62110))
  EVAL=()
  FINETUNE=(--iterations 100 --workers 16)   # 16 SUMO processes x 32 replicas; 2.5x faster than 4 x 128
  ITERATIONS=(50 60 70 80 90 100)
fi

mkdir -p "$OUT"
export SUMO_CHECK_JUNCTIONS="${SUMO_CHECK_JUNCTIONS:-0}"
echo "SUMO_CHECK_JUNCTIONS=$SUMO_CHECK_JUNCTIONS" | tee -a "$OUT/settings.log"
stage() {
  local name="$1"; shift
  if [[ -e "$OUT/.done_$name" ]]; then
    echo "== $name: already done"
    return
  fi
  echo "== $name: $(date '+%F %T')"
  "$@" 2>&1 | tee -a "$OUT/$name.log"
  touch "$OUT/.done_$name"
}

# 0. Fixed-route suite: road, background at CALIBRATED_VPH (env/background_demand.py), saved
#    traffic states. Built only from
#    the repository code (env/, train.py) and benchmark/.
stage suite "$PY" scripts/evaluate_fixed_routes.py prepare "$SUITE" \
  --idm-only-speed-gain

# 1-2. IDM expert collection, BC training, and the built-in fixed-route evaluation.
stage bc "$PY" -u scripts/measure_calibrated_idm_coverage.py "$SUITE" "$BC" \
  --workers "$WORKERS" "${COLLECT[@]}" --lane-weight-root 4 \
  --connected-lane-bumper-gaps --signed-adjacent-gaps --idm-spawn-protect-m 20

# 3. Diverse start-lane test and validation suites, and the BC test evaluation.
stage diverse_suites bash -c "
  '$PY' scripts/evaluate_diverse_start_lanes.py prepare '$SUITE' '$TEST_SUITE' --seeds ${TEST_SEEDS[*]} &&
  '$PY' scripts/evaluate_diverse_start_lanes.py prepare '$SUITE' '$VALIDATION_SUITE' --seeds ${VALIDATION_SEEDS[*]}"
stage bc_eval "$PY" scripts/evaluate_diverse_start_lanes.py evaluate "$TEST_SUITE" \
  --model "$BC/training/model.pt" --output "$BC/evaluation_diverse_start_lanes" "${EVAL[@]}"

# 4. Anchored PPO fine-tuning from the BC actor; each executed lane change costs 0.01.
stage ppo "$PPO_PY" -u -m scripts.finetune_bc_ppo "$BC/training/model.pt" "$SUITE" \
  --init-std 0.05 --lane-sharpness 2 --lr 5e-5 --anchor-coef 0.03 --overspeed-penalty 0 --lane-change-penalty 0.01 --seed 0 \
  --output "$PPO" "${FINETUNE[@]}"

# 5. Pick the checkpoint on the validation suite, then evaluate it on the test suite.
stage ppo_select bash -c "
  '$PPO_PY' scripts/ppo_checkpoints.py export '$PPO' --iterations ${ITERATIONS[*]} &&
  for i in ${ITERATIONS[*]}; do
    dir=\$(printf '$PPO/eval_runs/iter%03d' \$i)
    '$PPO_PY' scripts/evaluate_diverse_start_lanes.py evaluate '$VALIDATION_SUITE' \
      --model \$dir/model.pt --output \$dir/evaluation_validation ${EVAL[*]} || exit 1
  done &&
  '$PPO_PY' scripts/ppo_checkpoints.py select '$PPO' --evaluation evaluation_validation"
BEST=$("$PPO_PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['best']['model'])" "$PPO/selection.json")
stage ppo_eval "$PPO_PY" scripts/evaluate_diverse_start_lanes.py evaluate "$TEST_SUITE" \
  --model "$BEST" --output "$(dirname "$BEST")/evaluation_diverse_start_lanes" "${EVAL[@]}"

"$PY" - "$BC/evaluation_diverse_start_lanes" "$(dirname "$BEST")/evaluation_diverse_start_lanes" <<'EOF'
import json, sys
for name, path in zip(("BC", "PPO"), sys.argv[1:]):
    overall = json.load(open(f"{path}/evaluation_summary.json"))["overall"]
    print(f"{name}: {overall['success_rate']:.2%} of {overall['episodes']} cases  {overall['statuses']}")
EOF
echo "Done: $OUT"
