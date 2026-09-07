#!/usr/bin/env bash
# Re-run the test pass for arms whose training finished but whose evaluation hit
# an MPS out-of-memory.
#
# Both BCE arms died at almost exactly the same point -- batch 15 865 and 15 862
# of 34 819 -- against a 42.43 GiB ceiling, while both focal arms passed the
# identical split. Only the evaluation is repeated: each arm's `best.ckpt`
# survived, so the hours of training are intact. `--eval-only` loads it, fits
# the operating point on validation, and scores the test split, producing the
# same artifacts a run that finished normally would.
#
# Waits for the arm chain first: the failure is allocator pressure on a GPU that
# is already busy, so overlapping would invite the same OOM and slow both.

set -u

LOG_DIR="logs"
STATUS="${LOG_DIR}/arms.status"
ARMS=(fw-bce fw-bce-augment)

mkdir -p "${LOG_DIR}"

while pgrep -f run_arms.sh > /dev/null 2>&1; do
    sleep 300
done

echo "$(date '+%F %T')  re-eval: chain finished, starting" | tee -a "${STATUS}"

for arm in "${ARMS[@]}"; do
    ckpt="results/frame-wise/${arm}/checkpoints/best.ckpt"
    if [ ! -f "${ckpt}" ]; then
        echo "$(date '+%F %T')  ${arm} eval: no checkpoint, skipped" | tee -a "${STATUS}"
        continue
    fi

    echo "$(date '+%F %T')  ${arm} eval: started" | tee -a "${STATUS}"
    start=$(date +%s)

    if make eval-frame-wise CKPT="${ckpt}" \
            ARGS="--set train.mlflow.run_name=${arm}" \
            > "${LOG_DIR}/eval-${arm}.log" 2>&1; then
        verdict="ok"
    else
        verdict="FAILED (exit $?)"
    fi

    elapsed=$(( ($(date +%s) - start) / 60 ))
    score=$(grep -oE "test/mean_f1 = [0-9.]+" "${LOG_DIR}/eval-${arm}.log" | tail -1 || true)
    echo "$(date '+%F %T')  ${arm} eval: ${verdict} after ${elapsed} min  ${score}" | tee -a "${STATUS}"
done

echo "$(date '+%F %T')  re-eval: all done" | tee -a "${STATUS}"
