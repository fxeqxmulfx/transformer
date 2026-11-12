#!/usr/bin/env bash
# Launch only a fresh, committed loss ablation. Keep every exit and endpoint.
set -euo pipefail
cal_experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cal_binary=${1:?frozen binary required}
cal_cohort=${2:?fresh cohort required}
cal_workers=${3:-1}
[[ "$cal_workers" == 1 || "$cal_workers" == 2 ]]
mkdir -- "$cal_cohort"
"$cal_binary" fingerprint > "$cal_cohort/fingerprint.txt"
sha256sum "$cal_binary" > "$cal_cohort/binary.sha256"
cp -- "$cal_experiment/protocol.json" "$cal_cohort/protocol.json"
jq -r '.tasks[] | .name as $name | ["scaled", "calibrated"][] | [., $name] | @tsv' \
  "$cal_experiment/../convex_stress/suite.json" > "$cal_cohort/jobs.tsv"
cal_memory=$(jq -r '.memory_gib * 1024 * 1024' "$cal_experiment/protocol.json")
cal_timeout=$(jq -r '.hard_cell_seconds' "$cal_experiment/protocol.json")
export cal_binary cal_cohort cal_memory cal_timeout
xargs -P "$cal_workers" -n 2 bash -c '
  set -uo pipefail
  ulimit -c 0
  ulimit -v "$cal_memory"
  id="$1--$2"
  cell="$cal_cohort/$id"
  /usr/bin/time -v timeout --kill-after=5 "$cal_timeout" "$cal_binary" run "$1" "$2" "$cell" > "$cell.log" 2>&1
  code=$?
  printf "%s\n" "$code" > "$cell.exit"
  if [[ "$code" == 0 ]]; then
    /usr/bin/time -v timeout --kill-after=5 "$cal_timeout" "$cal_binary" replay "$cell" > "$cell.replay.log" 2>&1
    printf "%s\n" "$?" > "$cell.replay.exit"
  fi
  printf "%s exit=%s\n" "$id" "$code"
' _ < "$cal_cohort/jobs.tsv"
"$cal_binary" summary "$cal_cohort" > "$cal_cohort/summary.json"
