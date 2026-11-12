#!/usr/bin/env bash
# Serial CM15 construction and raw-table replay after the frozen phase cohort.
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: run_cohort.sh FRESH_ROOT PHASE_ROOT' >&2; exit 2; fi
: "${DECODE_PRODUCER_COMMIT:?set the committed decoder producer}"
binary=${DECODE_BINARY:-experiments/convex_stress_decode/target/release/convex-stress-decode}
destination=$1
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/time -v -o "$destination-prepare.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" prepare "$destination" "$2"
jq -r '.cases[] | .method + "--" + .task' "$destination/inventory.json" > "$destination/cells.txt"
while IFS= read -r cell; do
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < 16 * 1024 * 1024 * 1024 )); then echo 'Disk reserve reached; preserve outcomes and stop.' >&2; exit 2; fi
  /usr/bin/time -v -o "$destination/$cell.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" run-cell "$destination" "$cell" > "$destination/$cell.log" 2>&1
  /usr/bin/time -v -o "$destination/$cell-replay.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" replay-cell "$destination" "$cell" > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
done < "$destination/cells.txt"
bash experiments/convex_stress_hull/limited.sh 8 "$binary" summarize "$destination" > "$destination/summary.json"
