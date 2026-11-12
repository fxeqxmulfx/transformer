#!/usr/bin/env bash
# Frozen CM18 follow-up. No original inputs enter the discovery before prepare.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: run_cohort.sh FRESH_ROOT' >&2; exit 2; fi
: "${CERTIFIED_PRODUCER_COMMIT:?set the committed certified producer}"
binary=${CERTIFIED_BINARY:-experiments/convex_stress_certified/target/release/convex-stress-certified}
destination=$1
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/time -v -o "$destination-prepare.resources" bash experiments/convex_stress_certified/limited.sh "$binary" prepare "$destination"
jq -r '.cases[] | .arm + "--" + .task + "--" + .decoder + "--" + .budget' "$destination/inventory.json" > "$destination/cells.txt"
while IFS= read -r cell; do
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < 16 * 1024 * 1024 * 1024 )); then echo 'Disk reserve reached; preserve outcomes and stop.' >&2; exit 2; fi
  /usr/bin/time -v -o "$destination/$cell.resources" bash experiments/convex_stress_certified/limited.sh "$binary" run-cell "$destination" "$cell" > "$destination/$cell.log" 2>&1
  /usr/bin/time -v -o "$destination/$cell-replay.resources" bash experiments/convex_stress_certified/limited.sh "$binary" replay-cell "$destination" "$cell" > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
done < "$destination/cells.txt"
bash experiments/convex_stress_certified/limited.sh "$binary" summarize "$destination" > "$destination/summary.json"
