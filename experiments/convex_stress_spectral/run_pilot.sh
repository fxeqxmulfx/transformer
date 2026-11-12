#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/experiments/convex_stress_spectral"
cohort=${1:?usage: run_pilot.sh COHORT}
binary=${SPECTRAL_BIN:-"$source_dir/target/release/convex-stress-spectral"}
mkdir -p "$cohort"
cohort=$(realpath "$cohort")
cp "$source_dir/protocol.json" "$cohort/protocol.json"
"$binary" fingerprint > "$cohort/source_fingerprint.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export SPECTRAL_RUN_BINARY="$binary" SPECTRAL_RUN_COHORT="$cohort"
jq -r '.tasks[] as $t | .arms[] as $a | .radii[] | [$a,$t,.] | @tsv' "$source_dir/protocol.json" > "$cohort/planned.tsv"
# Independent numerical cells; these are processes, not collaborating agents.
xargs -r -P3 -n3 bash -c '
  set -euo pipefail
  arm=$1; task=$2; radius=$3
  out="$SPECTRAL_RUN_COHORT/$arm--$task--$radius"
  "$SPECTRAL_RUN_BINARY" run "$arm" "$task" "$radius" "$out" > "$out.log" 2>&1
  "$SPECTRAL_RUN_BINARY" replay "$out" > "$out/replay.json"
' spectral-cell < "$cohort/planned.tsv"
jq -s --slurpfile p "$source_dir/protocol.json" -f "$source_dir/summarize.jq" "$cohort"/*/result.json > "$cohort/summary.json"
