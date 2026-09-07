#!/usr/bin/env bash
# Re-fit each frame-wise arm's event operating point WITH hysteresis, on the
# training corpora, then re-score the test split.
#
# Task 1 of tmp/plan.md. The hysteresis numbers measured so far were swept on
# the dumped *test* signals and are therefore not publishable; this fits the
# (high, low) pair on validation via --shared-threshold-dir, which is what makes
# the resulting table honest.
#
# Sequential: one MPS device, and each arm's eval is memory-hungry enough that
# overlapping them is how the earlier OOMs happened.
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS=logs/refit.status
mkdir -p logs
: > "$STATUS"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

for arm in fw-bce fw-bce-augment fw-focal fw-focal-augment; do
  ckpt="results/frame-wise/$arm/checkpoints/best.ckpt"
  thr="results/frame-wise/$arm/hysteresis"
  if [ ! -f "$ckpt" ]; then say "$arm: SKIP (no checkpoint)"; continue; fi

  rm -rf "$thr"; mkdir -p "$thr"
  say "$arm: fitting (high, low) on validation"
  start=$(date +%s)

  if uv run python -m blinklinmult.train.cli \
       --data config/data/stills_all.yaml \
       --model config/model/blinkcnn.yaml \
       --train config/train/frame_wise.yaml \
       --eval-only "$ckpt" \
       --resumable \
       --shared-threshold-dir "$thr" \
       --set 'data.datasets=[rn15,rn30]' \
       --set 'data.eval_datasets=[]' \
       --set "train.mlflow.run_name=$arm-hysteresis" \
       >> "logs/refit-$arm.log" 2>&1
  then
    mins=$(( ($(date +%s) - start) / 60 ))
    pair=$(cat "$thr/event_threshold.json" 2>/dev/null | tr -d '\n ')
    say "$arm: ok after ${mins} min  $pair"
  else
    say "$arm: FAILED (see logs/refit-$arm.log)"
  fi
done

say "refit: all done"
