#!/usr/bin/env bash
# A separate frozen follow-up, started only after the original cohort closes.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: run_retained.sh FRESH_COHORT_DIRECTORY' >&2; exit 2; fi
: "${DEPTH_PRODUCER_COMMIT:?set the committed retained producer}"
: "${DEPTH_REFERENCE_BINARY:?set the frozen 032762e5 baseline executable}"
binary=${DEPTH_BINARY:-experiments/convex_stress_depth/target/release/convex-stress-depth}
reference=${DEPTH_REFERENCE_ROOT:-experiments/convex_stress_depth/runs/cohort-v1}
destination=$1
if [[ -e "$destination" || -e "$destination-reference-summary.json" ]]; then
  echo 'Retained cohort requires a fresh destination and reference summary.' >&2; exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
expected=d6931b95c2b1d4971c02b718772fd08105c554d2d0bb66b99aedbf1b07b3e56f
if [[ $("$DEPTH_REFERENCE_BINARY" cohort-fingerprint) != "$expected" ]]; then
  echo 'Reference executable does not match the frozen original producer.' >&2; exit 2
fi
bash experiments/convex_stress_hull/limited.sh 8 "$DEPTH_REFERENCE_BINARY" summarize-cohort "$reference" \
  > "$destination-reference-summary.json"
jq -e --arg expected "$expected" '.source_fingerprint==$expected and .cells==288 and (.complete+.memory_ineligible)==288 and (.choices|length)==72' \
  "$destination-reference-summary.json" > /dev/null
sha256sum "$destination-reference-summary.json" > "$destination-reference-summary.sha256"
/usr/bin/time -v -o "$destination-prepare.resources" \
  bash experiments/convex_stress_hull/limited.sh 8 "$binary" prepare-retained-cohort "$destination"
need=$(jq -r .projected_disk_bytes "$destination/inventory.json")
available=$(df --output=avail -B1 "$destination" | tail -n 1)
reserve=$((16 * 1024 * 1024 * 1024))
if (( need > 96 * 1024 * 1024 * 1024 || available < need + reserve )); then
  echo 'Retained cohort would exceed its disk budget or reserve.' >&2; exit 2
fi
jq -r '.allocations[].id' "$destination/inventory.json" > "$destination/cells.txt"
while IFS= read -r cell; do
  need=$(jq -r --arg cell "$cell" '.allocations[] | select(.id==$cell) | .disk_bytes_upper_bound' "$destination/inventory.json")
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < need + reserve )); then
    echo 'Disk reserve reached; preserve existing outcomes and stop.' >&2; exit 2
  fi
  /usr/bin/time -v -o "$destination/$cell.resources" \
    bash experiments/convex_stress_hull/limited.sh 8 "$binary" run-cell "$destination" "$cell" \
    > "$destination/$cell.log" 2>&1
  /usr/bin/time -v -o "$destination/$cell-replay.resources" \
    bash experiments/convex_stress_hull/limited.sh 8 "$binary" replay-cell "$destination" "$cell" \
    > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
done < "$destination/cells.txt"
bash experiments/convex_stress_hull/limited.sh 8 "$binary" summarize-cohort "$destination" > "$destination/summary.json"
