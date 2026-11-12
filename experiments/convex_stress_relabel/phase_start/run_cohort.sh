#!/usr/bin/env bash
# Frozen constructor evaluation after the cold cohort and graph interpretation.
set -euo pipefail
if [[ $# != 3 ]]; then echo 'usage: run_cohort.sh FRESH_ROOT COLD_ROOT GRAPH_AUDIT_ROOT' >&2; exit 2; fi
: "${PHASE_PRODUCER_COMMIT:?set the committed phase producer}"
binary=${PHASE_BINARY:-experiments/convex_stress_relabel/phase_start/target/release/convex-phase-start}
destination=$1
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/time -v -o "$destination-prepare.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" prepare "$destination" "$2" "$3"
jq -r '.cases[] | [.arm,.task,.coding,.budget] | join("--")' "$destination/inventory.json" > "$destination/cells.txt"
while IFS= read -r cell; do
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < 16 * 1024 * 1024 * 1024 )); then echo 'Disk reserve reached; preserve outcomes and stop.' >&2; exit 2; fi
  /usr/bin/time -v -o "$destination/$cell.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" run-cell "$destination" "$cell" > "$destination/$cell.log" 2>&1
  /usr/bin/time -v -o "$destination/$cell-replay.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" replay-cell "$destination" "$cell" > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
done < "$destination/cells.txt"
bash experiments/convex_stress_hull/limited.sh 8 "$binary" summarize "$destination" > "$destination/summary.json"
