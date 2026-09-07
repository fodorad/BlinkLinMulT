#!/usr/bin/env bash
# Run the four frame-wise arms one after another, unattended.
#
# Sequential rather than parallel: the arms share one GPU and one 45 GB corpus,
# so running them together would make each slower and the timings
# incomparable — the point of the comparison is that the arms differ in one
# thing, and contention is a second thing.
#
# An arm that fails does not stop the rest. Each writes its own log under
# logs/, and this script appends a one-line verdict per arm to logs/arms.status
# so progress is readable without opening anything.

set -u

ARMS=(train-fw-bce train-fw-focal train-fw-bce-augment train-fw-focal-augment)
LOG_DIR="logs"
STATUS="${LOG_DIR}/arms.status"

mkdir -p "${LOG_DIR}"
: > "${STATUS}"

echo "$(date '+%F %T')  starting ${#ARMS[@]} arms" | tee -a "${STATUS}"

for arm in "${ARMS[@]}"; do
    log="${LOG_DIR}/${arm}.log"
    echo "$(date '+%F %T')  ${arm}: started" | tee -a "${STATUS}"

    # Resume if this arm already has a checkpoint: a run killed overnight
    # continues from its last completed epoch instead of discarding the hours
    # it already spent. A first launch has no checkpoint and starts at 0.
    run_name="${arm#train-}"
    ckpt="results/frame-wise/${run_name}/checkpoints/last.ckpt"
    extra=""
    if [ -f "${ckpt}" ]; then
        extra="--resume"
        echo "$(date '+%F %T')  ${arm}: resuming from ${ckpt}" | tee -a "${STATUS}"
    fi

    start=$(date +%s)
    if make "${arm}" ARGS="${extra}" > "${log}" 2>&1; then
        verdict="ok"
    else
        verdict="FAILED (exit $?)"
    fi
    elapsed=$(( ($(date +%s) - start) / 60 ))

    # The headline number, pulled from the log so the status file alone tells
    # you how each arm did.
    score=$(grep -oE "test/mean_f1 = [0-9.]+" "${log}" | tail -1 || true)
    echo "$(date '+%F %T')  ${arm}: ${verdict} after ${elapsed} min  ${score}" | tee -a "${STATUS}"
done

echo "$(date '+%F %T')  all arms done" | tee -a "${STATUS}"
