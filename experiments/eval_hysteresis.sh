#!/usr/bin/env bash
# Score each arm on EVERY corpus, one corpus per process, reusing the
# hysteresis pair already fitted on that arm's validation split.
#
# Two reasons this cannot be one process:
#   * memory -- the full test split is 2.4M frames, MPEblink alone 1.9M behind
#     a 40 GB h5, which SIGKILLs a 34 GB host;
#   * attribution -- rn15 and rn30 use identical recording ids (`test_N`), so a
#     pooled signal dump cannot be split by corpus afterwards. Scoring each in
#     its own run is the only way to get a trustworthy per-dataset row.
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS=logs/hyst_eval.status
mkdir -p logs
: > "$STATUS"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

TRAIN_CORPORA="rn15 rn30"
EVAL_CORPORA="talkingface mpeblink"

for arm in fw-bce fw-bce-augment fw-focal fw-focal-augment; do
  ckpt="results/frame-wise/$arm/checkpoints/best.ckpt"
  thr="results/frame-wise/$arm/hysteresis"
  [ -f "$ckpt" ] || { say "$arm: SKIP (no checkpoint)"; continue; }
  [ -f "$thr/event_threshold.json" ] || { say "$arm: SKIP (no fitted pair)"; continue; }

  for corpus in $TRAIN_CORPORA; do
    say "$arm / $corpus (train-corpus route)"
    uv run python -m blinklinmult.train.cli \
      --data config/data/stills_all.yaml --model config/model/blinkcnn.yaml \
      --train config/train/frame_wise.yaml \
      --set train.event_carrier_only=true --eval-only "$ckpt" --reuse-threshold --resumable \
      --shared-threshold-dir "$thr" \
      --set "data.datasets=[$corpus]" --set 'data.eval_datasets=[]' \
      --set "train.mlflow.run_name=$arm-hyst-$corpus" \
      >> "logs/hyst-$arm-$corpus.log" 2>&1 \
      && say "$arm / $corpus: ok" || say "$arm / $corpus: FAILED"
  done

  for corpus in $EVAL_CORPORA; do
    say "$arm / $corpus (eval-only route)"
    uv run python -m blinklinmult.train.cli \
      --data config/data/stills_all.yaml --model config/model/blinkcnn.yaml \
      --train config/train/frame_wise.yaml \
      --eval-only "$ckpt" --reuse-threshold --resumable \
      --shared-threshold-dir "$thr" \
      --set 'data.datasets=[cew]' --set "data.eval_datasets=[$corpus]" \
      --set "train.mlflow.run_name=$arm-hyst-$corpus" \
      >> "logs/hyst-$arm-$corpus.log" 2>&1 \
      && say "$arm / $corpus: ok" || say "$arm / $corpus: FAILED"
  done
done

say "hysteresis eval: all done"
