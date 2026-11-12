#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/experiments/convex_stress_sharpen"
parents=$(realpath "${1:?usage: run_pilot.sh PARENTS FRESH_COHORT [WORKERS]}")
cohort=${2:?usage: run_pilot.sh PARENTS FRESH_COHORT [WORKERS]}
workers=${3:-2}
binary=${SHARPEN_BIN:-"$source_dir/target/release/convex-stress-sharpen"}
mkdir "$cohort"
cohort=$(realpath "$cohort")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$binary" prepare "$parents" > "$cohort/prepared.json"
"$binary" fingerprint > "$cohort/source_fingerprint.txt"
cp "$source_dir/protocol.json" "$cohort/protocol.json"
jq -r '.tasks[] as $t | .arms[] as $a | .budgets[] | [$a,$t,.] | @tsv' \
  "$source_dir/protocol.json" > "$cohort/planned.tsv"
export SHARPEN_RUN_BINARY="$binary" SHARPEN_RUN_COHORT="$cohort" SHARPEN_PARENTS="$parents"
# Independent numerical endpoint checks, not additional agents or training runs.
xargs -r -P "$workers" -n3 bash -c '
  set -euo pipefail
  arm=$1; task=$2; budget=$3
  out="$SHARPEN_RUN_COHORT/$arm--$task--$budget"
  if "$SHARPEN_RUN_BINARY" run "$SHARPEN_PARENTS" "$arm" "$task" "$budget" "$out" > "$out.log" 2>&1; then
    if "$SHARPEN_RUN_BINARY" replay "$SHARPEN_PARENTS" "$out" > "$out/replay.json" 2> "$out/replay-error.log"; then
      printf "0\n" > "$out.exit"
    else
      status=$?
      printf "%s\n" "$status" > "$out.exit"
      exit "$status"
    fi
  else
    status=$?
    printf "%s\n" "$status" > "$out.exit"
    exit "$status"
  fi
' sharpen-cell < "$cohort/planned.tsv"
jq -s --slurpfile p "$source_dir/protocol.json" --slurpfile prep "$cohort/prepared.json" \
  -f "$source_dir/summarize.jq" "$cohort"/*/result.json > "$cohort/summary.json"
"$binary" select "$cohort" > "$cohort/hull-selection.log"
"$binary" replay-hull "$cohort" > "$cohort/hull-replay.json"
