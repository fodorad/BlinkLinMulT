#!/usr/bin/env bash
# The frozen-encoder search from tmp/overview.md, section 9.
#
# Every arm is frozen and cache-backed (~2.7 min/epoch), so the whole search is
# a working day of compute rather than a week. Each arm trains, then scores
# itself per corpus in the same pass -- the single-process test was verified to
# agree with the per-corpus route to four decimals, and it is the only route
# that can score mpeblink and hust_lebw, which carry no `eye_state` field and so
# cannot be evaluated in isolation.
#
# Usage:  experiments/run_lint_search.sh step1
#         experiments/run_lint_search.sh dmodel
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS=logs/lint_search.status
mkdir -p logs
touch "$STATUS"
say() { echo "$(date '+%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

SEED=results/frame-wise/fw-focal-augment/checkpoints/best.ckpt
CACHE=cache/embeddings

# Common to every arm: frozen encoder, cached embeddings, no augmentation
# (the cache requires it off), MPEblink thinned so the ESR head is actually
# supervised -- see step 1 of the plan.
base_args() {
  echo "--set model.encoder_weights=$SEED \
        --set model.encoder_freeze=true \
        --set data.augment=null \
        --set data.embedding_cache=$CACHE"
}

run() {
  local name=$1; shift
  local dir="results/blink-video/$name"
  local resume=()
  [ -f "$dir/checkpoints/last.ckpt" ] && resume=(--resume)

  say "$name: starting"
  local start; start=$(date +%s)
  if uv run python -m blinklinmult.train.cli \
       --data config/data/video_all.yaml \
       --model config/model/blinklint.yaml \
       --train config/train/video.yaml \
       $(base_args) "$@" \
       ${resume[@]+"${resume[@]}"} \
       --set "train.mlflow.run_name=$name" \
       >> "logs/$name.log" 2>&1
  then
    local mins=$(( ($(date +%s) - start) / 60 ))
    local f1; f1=$(grep -aoE "test/mean_f1 = [0-9.]+" "logs/$name.log" | tail -1)
    say "$name: ok after ${mins} min  $f1"
  else
    say "$name: FAILED after $(( ($(date +%s) - start) / 60 )) min"
  fi
}

case "${1:-step1}" in
  step1)
    # Step 1a: thin MPEblink so the eye_state head sees gradient in ~60% of
    # batches rather than 28%.
    run vid-lint-s1-mpe10 --set 'data.fit_fractions={mpeblink: 0.1}'
    ;;
  step15)
    # Step 1.5: the same data as step 1, with the event head stacked on the ESR
    # signal. The only difference from `vid-lint-s1-mpe10` is the head, so the
    # delta is attributable. Writes two event reports -- `test_events.json` for
    # the learned route and `test_events_eye_state.json` for the hand-fitted
    # hysteresis extractor -- so the two can be compared on one test set.
    run vid-lint-s15-evhead \
      --set 'data.fit_fractions={mpeblink: 0.1}' \
      --set model.event_head=conv
    ;;
  evhead)
    # Step 1.6: the two event-head variants, each one flag from `step15` so the
    # delta is attributable. Both address something step15 measured rather than
    # a guess:
    #
    #   detach=false -- step15 left rn30 ESR flat (0.5029 against BlinkCNN's
    #     0.6861). Letting event gradient reach the ESR head may sharpen the
    #     blink boundaries rn30 under-specifies.
    #   attention    -- step15 cost HUST-LEBW recall (0.8530 -> 0.4426) at
    #     unchanged precision, i.e. it turned conservative on short clips. A
    #     global-context head may fire where a 9-frame conv window cannot.
    run vid-lint-s16-attached \
      --set 'data.fit_fractions={mpeblink: 0.1}' \
      --set model.event_head=conv \
      --set model.event_head_detach=false
    run vid-lint-s16-attn \
      --set 'data.fit_fractions={mpeblink: 0.1}' \
      --set model.event_head=attention
    ;;
  dmodel)
    # Step 2: the trainable part is 1.6% of the model at d_model=32. Sweep it.
    for d in 64 128 256; do
      run "vid-lint-s2-d$d" \
        --set 'data.fit_fractions={mpeblink: 0.1}' \
        --set "model.d_model=$d"
    done
    ;;
  all)
    # Both arms back to back. They differ in exactly one thing -- the event
    # head -- so the delta between them is attributable to it and nothing else.
    bash "$0" step1
    bash "$0" step15
    ;;
  *)
    say "unknown stage ${1}"; exit 1 ;;
esac

say "search stage ${1:-step1}: done"
