#!/usr/bin/env bash
# Quick BlinkLinT sweep: 10% of train/valid, FULL test.
#
# Two corpus sets x three initialisations, six runs, in order:
#
#   rn      = rn15 + rn30 + hust_lebw          7 028 train windows, all full
#   rnmpe   = rn      + mpeblink at 10%       13 404 train windows
#
#   scratch -> fine-tuned -> frozen, within each set.
#
# **Only MPEblink is subsampled.** It is 63 758 of the 70 786 training windows
# (90%), so thinning it alone cuts the cost of an epoch by 5x while RN15, RN30
# and HUST-LEBW -- the corpora the result is actually reported on -- stay at
# full size. `data.fit_fractions` applies to train and validation only; **the
# test split is never touched**, for any corpus, because scoring each arm on a
# different subset would make the arms incomparable.
#
# Sequential: one MPS device, and the step is compute-bound (measured 5.7 s per
# batch fine-tuned, 0.74 s frozen), so parallel runs would only contend.
#
# Resumable: each arm picks up its own last.ckpt, so re-running this script
# continues wherever it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS=logs/video_quick.status
mkdir -p logs
touch "$STATUS"
say() { echo "$(date '+%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

SEED=results/frame-wise/fw-focal-augment/checkpoints/best.ckpt
MPE_FRACTION=0.1

run() {
  local name=$1 corpora=$2 init=$3
  local dir="results/blink-video/$name"
  local extra=()

  case "$init" in
    scratch) ;;
    ft)      extra=(--set "model.encoder_weights=$SEED") ;;
    frozen)  extra=(--set "model.encoder_weights=$SEED" --set model.encoder_freeze=true) ;;
  esac

  local fractions=()
  if [[ "$corpora" == *mpeblink* ]]; then
    fractions=(--set "data.fit_fractions={mpeblink: $MPE_FRACTION}")
  fi

  local resume=()
  if [ -f "$dir/checkpoints/last.ckpt" ]; then
    resume=(--resume); say "$name: resuming"
  else
    say "$name: starting ($corpora, $init)"
  fi

  local start; start=$(date +%s)
  if uv run python -m blinklinmult.train.cli \
       --data config/data/video_all.yaml \
       --model config/model/blinklint.yaml \
       --train config/train/video.yaml \
       --set "data.datasets=[$corpora]" \
       ${fractions[@]+"${fractions[@]}"} \
       ${extra[@]+"${extra[@]}"} ${resume[@]+"${resume[@]}"} \
       --set "train.mlflow.run_name=$name" \
       >> "logs/$name.log" 2>&1
  then
    local f1; f1=$(grep -aoE "test/mean_f1 = [0-9.]+" "logs/$name.log" | tail -1)
    say "$name: ok after $(( ($(date +%s) - start) / 60 )) min  $f1"
  else
    say "$name: FAILED after $(( ($(date +%s) - start) / 60 )) min"
  fi
}

RN="rn15,rn30,hust_lebw"
RNMPE="rn15,rn30,hust_lebw,mpeblink"

run vid-lint-scratch-rn    "$RN"    scratch
run vid-lint-ft-rn         "$RN"    ft
run vid-lint-frozen-rn     "$RN"    frozen

run vid-lint-scratch-rnmpe "$RNMPE" scratch
run vid-lint-ft-rnmpe      "$RNMPE" ft
run vid-lint-frozen-rnmpe  "$RNMPE" frozen

say "quick sweep: all done"
