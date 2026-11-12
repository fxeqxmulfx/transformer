#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/experiments/convex_stress_schatten"
cohort=${1:?usage: run_pilot.sh COHORT}
binary=${SCHATTEN_BIN:-"$source_dir/target/release/convex-stress-schatten"}
mkdir -p "$cohort"
cohort=$(realpath "$cohort")
cp "$source_dir/protocol.json" "$cohort/protocol.json"
"$binary" fingerprint > "$cohort/source_fingerprint.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SCHATTEN_RUN_BINARY="$binary" SCHATTEN_RUN_COHORT="$cohort"
jq -r '.tasks[] as $t | .arms[] as $a | .budgets[] | [$a,$t,.] | @tsv' "$source_dir/protocol.json" > "$cohort/planned.tsv"
# Independent experiment processes, not collaborating agents; eight frequency workers each.
xargs -r -P3 -n3 bash -c '
  set -euo pipefail
  arm=$1; task=$2; budget=$3
  out="$SCHATTEN_RUN_COHORT/$arm--$task--$budget"
  "$SCHATTEN_RUN_BINARY" run "$arm" "$task" "$budget" "$out" > "$out.log" 2>&1
  "$SCHATTEN_RUN_BINARY" replay "$out" > "$out/replay.json"
' schatten-cell < "$cohort/planned.tsv"
jq -s --slurpfile p "$source_dir/protocol.json" -f "$source_dir/summarize.jq" "$cohort"/*/result.json > "$cohort/summary.json"
"$binary" select "$cohort" > "$cohort/hull-selection.log"
"$binary" replay-hull "$cohort" > "$cohort/hull-replay.json"
