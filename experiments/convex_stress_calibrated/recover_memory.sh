#!/usr/bin/env bash
# Resource-only retries with the unchanged producer; preserve original failures.
set -euo pipefail
rec_experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rec_binary=${1:?frozen producer required}
rec_original=${2:?original cohort required}
rec_output=${3:?fresh recovery directory required}
rec_expected=$(jq -r '.producer_fingerprint' "$rec_experiment/recovery.json")
[[ "$("$rec_binary" fingerprint)" == "$rec_expected" ]]
[[ "$(cat "$rec_original/fingerprint.txt")" == "$rec_expected" ]]
[[ "$(jq -r '.planned' "$rec_original/summary.json")" == 70 ]]
while IFS= read -r id; do
  [[ "$(cat "$rec_original/$id.exit")" == 134 ]]
  rg -q 'memory allocation|failed to allocate' "$rec_original/$id.log"
done < <(jq -r '.cells[]' "$rec_experiment/recovery.json")
mkdir -- "$rec_output"
cp -- "$rec_experiment/recovery.json" "$rec_output/resource_override.json"
cp -- "$rec_original/fingerprint.txt" "$rec_output/fingerprint.txt"
sha256sum "$rec_binary" > "$rec_output/binary.sha256"
rec_memory=$(jq -r '.memory_gib * 1024 * 1024' "$rec_experiment/recovery.json")
rec_timeout=$(jq -r '.hard_cell_seconds' "$rec_experiment/recovery.json")
ulimit -c 0
ulimit -v "$rec_memory"
while IFS= read -r id; do
  rec_arm=${id%%--*}
  rec_task=${id#*--}
  rec_cell="$rec_output/$id"
  if /usr/bin/time -v timeout --kill-after=5 "$rec_timeout" "$rec_binary" run "$rec_arm" "$rec_task" "$rec_cell" > "$rec_cell.log" 2>&1; then
    printf '0\n' > "$rec_cell.exit"
    if /usr/bin/time -v timeout --kill-after=5 "$rec_timeout" "$rec_binary" replay "$rec_cell" > "$rec_cell.replay.log" 2>&1; then
      printf '0\n' > "$rec_cell.replay.exit"
    else
      printf '%s\n' "$?" > "$rec_cell.replay.exit"
    fi
  else
    printf '%s\n' "$?" > "$rec_cell.exit"
  fi
  printf '%s recovery finished\n' "$id"
done < <(jq -r '.cells[]' "$rec_experiment/recovery.json")
