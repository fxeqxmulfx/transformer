#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/experiments/convex_stress_sharing"
cohort=${1:?usage: run_pilot.sh FRESH_COHORT [WORKERS]}
workers=${2:-2}
binary=${SHARING_BIN:-"$source_dir/target/release/convex-stress-sharing"}
mkdir "$cohort"
cohort=$(realpath "$cohort")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$binary" prepare > "$cohort/prepared.json"
"$binary" fingerprint > "$cohort/source_fingerprint.txt"
cp "$source_dir/protocol.json" "$cohort/protocol.json"
jq -r '.tasks[] as $t | .arms[] as $a | .budgets[] | [$a,$t,.] | @tsv' \
  "$source_dir/protocol.json" > "$cohort/planned.tsv"
export BUDGETS_RUN_BINARY="$binary" BUDGETS_RUN_COHORT="$cohort"
# Independent numerical experiments, not additional agents.
xargs -r -P "$workers" -n3 bash -c '
  set -euo pipefail
  arm=$1; task=$2; budget=$3
  out="$BUDGETS_RUN_COHORT/$arm--$task--$budget"
  if "$BUDGETS_RUN_BINARY" run "$arm" "$task" "$budget" "$out" > "$out.log" 2>&1; then
    if "$BUDGETS_RUN_BINARY" replay "$out" > "$out/replay.json" 2> "$out/replay-error.log"; then
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
' budgets-cell < "$cohort/planned.tsv"
jq -s --slurpfile p "$source_dir/protocol.json" --slurpfile prep "$cohort/prepared.json" \
  -f "$source_dir/summarize.jq" "$cohort"/*/result.json > "$cohort/summary.json"
"$binary" select "$cohort" > "$cohort/hull-selection.log"
"$binary" replay-hull "$cohort" > "$cohort/hull-replay.json"
