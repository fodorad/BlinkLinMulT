#!/usr/bin/env bash
# Train the BlinkLinT arms sequentially: scratch -> fine-tuned -> frozen.
#
# Sequential, not parallel: one MPS device, and two runs would contend for both
# it and the page cache behind a 51 GB corpus set.
#
# Every arm resumes from its own last.ckpt if one exists, so an interrupted
# chain -- or a machine shut down for the night -- costs at most the epoch in
# progress. Re-running this script continues where it stopped.
set -uo pipefail
cd "$(dirname "$0")/.."

STATUS=logs/video_arms.status
mkdir -p logs
touch "$STATUS"
say() { echo "$(date '+%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

run_arm() {
  local target=$1 name=$2
  local ckpt="results/blink-video/$name/checkpoints/last.ckpt"
  local resume=""
  if [ -f "$ckpt" ]; then
    resume='ARGS=--resume'
    say "$name: resuming from last.ckpt"
  else
    say "$name: starting fresh"
  fi

  local start
  start=$(date +%s)
  if make "$target" $resume >> "logs/video-$name.log" 2>&1; then
    say "$name: ok after $(( ($(date +%s) - start) / 60 )) min"
  else
    say "$name: FAILED after $(( ($(date +%s) - start) / 60 )) min (see logs/video-$name.log)"
    return 1
  fi
}

run_arm video-lint-scratch vid-lint-scratch
run_arm video-lint-ft      vid-lint-ft
run_arm video-lint-frozen  vid-lint-frozen

say "video arms: all done"
