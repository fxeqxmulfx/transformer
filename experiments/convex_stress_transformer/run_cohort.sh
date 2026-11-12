#!/usr/bin/env bash
# Full original caps and all 35 tasks, with strictly serial saved-state replay.
set -euo pipefail
if [[ $# != 2 ]]; then echo 'usage: run_cohort.sh FRESH_ROOT DECODE_ROOT' >&2; exit 2; fi
: "${TRANSFORMER_PRODUCER_COMMIT:?set the committed transformer producer}"
binary=${TRANSFORMER_BINARY:-experiments/convex_stress_transformer/target/release/convex-stress-transformer}
destination=$1
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/time -v -o "$destination-prepare.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" prepare "$destination" "$2"
jq -r '.cases[] | "transformer--" + .task + "--seed" + (.seed|tostring)' "$destination/inventory.json" > "$destination/cells.txt"
test "$(wc -l < "$destination/cells.txt")" -eq 35
while IFS= read -r cell; do
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < 16 * 1024 * 1024 * 1024 )); then echo 'Disk reserve reached; preserve partial results and stop.' >&2; exit 2; fi
  /usr/bin/time -v -o "$destination/$cell.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" run-cell "$destination" "$cell" > "$destination/$cell.log" 2>&1
  /usr/bin/time -v -o "$destination/$cell-replay.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" replay-cell "$destination" "$cell" > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
done < "$destination/cells.txt"
bash experiments/convex_stress_hull/limited.sh 8 "$binary" summarize "$destination" > "$destination/summary.json"
