#!/usr/bin/env bash
# Score a trained video arm on every corpus, one corpus per process.
#
# Usage:  experiments/eval_video.sh <arm> [<arm> ...]
#         experiments/eval_video.sh vid-lint-ft vid-mult-ft
#
# Same discipline as the frame-wise benchmark, for the same two reasons:
#
#   * memory -- the full test split is 2.4M frames, MPEblink alone 1.9M behind a
#     40 GB h5, which SIGKILLs a 34 GB host when scored in one pass;
#   * attribution -- rn15 and rn30 use identical recording ids (`test_N`), so a
#     pooled signal dump cannot be split by corpus afterwards.
#
# The operating point is fitted ONCE per arm on the training corpora and shared
# across every corpus via --shared-threshold-dir. A per-corpus threshold would
# measure calibration rather than detection, and TalkingFace has no validation
# split to fit one on at all.
set -uo pipefail
cd "$(dirname "$0")/.."

# The unfreeze arms are stage 2 of the two-stage recipe and are scored as
# separate runs, so the improvement over their frozen stage 1 is a direct
# comparison rather than a claim.
ARMS=${*:-vid-lint-scratch vid-lint-frozen vid-lint-ft vid-lint-unfreeze \
          vid-mult-scratch vid-mult-frozen vid-mult-ft vid-mult-unfreeze}

STATUS=logs/video_eval.status
mkdir -p logs
: > "$STATUS"
say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

# Corpora that supply a validation split, and so can carry the head themselves.
TRAIN_CORPORA=${TRAIN_CORPORA:-"rn15 rn30 mpeblink hust_lebw"}
# Held out entirely; scored through a carrier, whose samples the event report
# now drops (see EventReport._by_recording).
EVAL_CORPORA=${EVAL_CORPORA-"talkingface"}
CARRIER=rn30

for arm in $ARMS; do
  case "$arm" in
    *lint*) model=config/model/blinklint.yaml ;;
    *mult*) model=config/model/blinklinmult.yaml ;;
    *) say "$arm: SKIP (cannot infer model from name)"; continue ;;
  esac

  ckpt="results/blink-video/$arm/checkpoints/best.ckpt"
  thr="results/blink-video/$arm/threshold"

  # A `-cached` arm was trained against precomputed embeddings, so scoring it
  # reads them too. Not for correctness -- its encoder is frozen, so encoding
  # live yields the identical vectors -- but for speed: the eval splits are
  # 64 027 windows and the CNN is the whole cost.
  extra=()
  case "$arm" in
    *-cached)
      # `encoder_freeze` is set for the *config check*, not to change the
      # model: `--eval-only` rebuilds the module from the checkpoint's own
      # stored config, so the loaded encoder is frozen either way. Without it
      # the cache guard refuses the run, because blinklint.yaml defaults the
      # flag to false and validation happens before the checkpoint is read.
      # `encoder_weights` accompanies it because `encoder_freeze` alone is
      # refused: freezing an encoder with no trained weights would pin it to
      # ImageNet features. Under `--eval-only` the value is inert -- the
      # checkpoint supplies the real weights -- but config validation runs
      # before the checkpoint is read, so it must still be satisfiable.
      extra=(--set model.encoder_freeze=true
             --set "model.encoder_weights=${VIDEO_SEED:-results/frame-wise/fw-focal-augment/checkpoints/best.ckpt}"
             --set "data.embedding_cache=${CACHE_ROOT:-cache/embeddings}"
             --set data.augment=null
             --set data.batch_size=64
             --set data.num_workers=6)
      ;;
  esac
  [ -f "$ckpt" ] || { say "$arm: SKIP (no checkpoint at $ckpt)"; continue; }
  mkdir -p "$thr"

  # mpeblink and hust_lebw carry no `eye_state` field at all -- they annotate
  # blink intervals and nothing else -- so asking for both targets fails when
  # one of them is scored alone. Scoring them on `blink_presence` only is not a
  # workaround: it is the same routing that lets them train the event head and
  # not the closure head.
  targets_for() {
    case "$1" in
      # `task` moves with `targets`: config refuses a joint task that
      # supervises two targets while only one is listed.
      mpeblink|hust_lebw) echo '--set train.task=blink_presence --set train.targets=[blink_presence]' ;;
      *) echo "" ;;
    esac
  }

  for corpus in $TRAIN_CORPORA; do
    say "$arm / $corpus"
    uv run python -m blinklinmult.train.cli \
      --data config/data/video_all.yaml --model "$model" \
      --train config/train/video.yaml \
      --set train.event_carrier_only=true --eval-only "$ckpt" --reuse-threshold --resumable \
      --shared-threshold-dir "$thr" \
      --set "data.datasets=[$corpus]" --set 'data.eval_datasets=[]' \
      $(targets_for "$corpus") \
      ${extra[@]+"${extra[@]}"} \
      --set "train.mlflow.run_name=$arm-eval-$corpus" \
      >> "logs/video-$arm-$corpus.log" 2>&1 \
      && say "$arm / $corpus: ok" || say "$arm / $corpus: FAILED"
  done

  for corpus in $EVAL_CORPORA; do
    say "$arm / $corpus (held out)"
    uv run python -m blinklinmult.train.cli \
      --data config/data/video_all.yaml --model "$model" \
      --train config/train/video.yaml \
      --eval-only "$ckpt" --reuse-threshold --resumable \
      --shared-threshold-dir "$thr" \
      --set "data.datasets=[$CARRIER]" --set "data.eval_datasets=[$corpus]" \
      ${extra[@]+"${extra[@]}"} \
      --set "train.mlflow.run_name=$arm-eval-$corpus" \
      >> "logs/video-$arm-$corpus.log" 2>&1 \
      && say "$arm / $corpus: ok" || say "$arm / $corpus: FAILED"
  done
done

say "video eval: all done"
