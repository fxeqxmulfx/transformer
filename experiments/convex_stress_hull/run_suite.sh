#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/experiments/convex_stress_hull"
cohort=${1:?usage: run_suite.sh FRESH_COHORT [WORKERS]}
workers=${2:-1}
memory_gib=${HULL_MEMORY_GIB:-16}
if [[ ! "$workers" =~ ^[12]$ ]]; then
  echo 'Use one or two native workers after the audit OOM.' >&2
  exit 2
fi
binary=${HULL_BIN:-"$source_dir/target/release/convex-stress-hull"}
mkdir "$cohort"
cohort=$(realpath "$cohort")
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bash "$source_dir/limited.sh" "$memory_gib" "$binary" prepare > "$cohort/prepared.json"
"$binary" fingerprint > "$cohort/source_fingerprint.txt"
cp "$source_dir/protocol.json" "$cohort/protocol.json"
jq -r '.protocol as $p | .tasks[].task as $t | $p.arms[] as $a | $p.candidates[] | [$a,$t,.] | @tsv' \
  "$cohort/prepared.json" > "$cohort/planned.tsv"
export HULL_RUN_BINARY="$binary" HULL_RUN_COHORT="$cohort" HULL_RUN_LIMIT="$source_dir/limited.sh" HULL_RUN_MEMORY="$memory_gib"
# Independent numerical experiments, not additional agents.
xargs -r -P "$workers" -n3 bash -c '
  set -euo pipefail
  arm=$1; task=$2; candidate=$3
  out="$HULL_RUN_COHORT/$arm--$task--$candidate"
  if /usr/bin/time -v -o "$out.resources" bash "$HULL_RUN_LIMIT" "$HULL_RUN_MEMORY" "$HULL_RUN_BINARY" run "$arm" "$task" "$candidate" "$out" > "$out.log" 2>&1; then
    if bash "$HULL_RUN_LIMIT" "$HULL_RUN_MEMORY" "$HULL_RUN_BINARY" replay "$out" > "$out/replay.json" 2> "$out/replay-error.log"; then
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
' hull-cell < "$cohort/planned.tsv"
jq -s --slurpfile prep "$cohort/prepared.json" -f "$source_dir/summarize.jq" \
  "$cohort"/*/result.json > "$cohort/summary.json"
bash "$source_dir/limited.sh" "$memory_gib" "$binary" select "$cohort" > "$cohort/selection.log"
bash "$source_dir/limited.sh" "$memory_gib" "$binary" replay-selection "$cohort" > "$cohort/selection-replay.json"
