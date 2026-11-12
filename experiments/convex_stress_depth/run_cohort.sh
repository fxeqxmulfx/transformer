#!/usr/bin/env bash
# Every cell and replay runs serially under an 8 GiB address-space limit.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: run_cohort.sh FRESH_COHORT_DIRECTORY' >&2; exit 2; fi
: "${DEPTH_PRODUCER_COMMIT:?set the committed producer revision}"
binary=${DEPTH_BINARY:-experiments/convex_stress_depth/target/release/convex-stress-depth}
destination=$1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/time -v -o "$destination-prepare.resources" \
  bash experiments/convex_stress_hull/limited.sh 8 "$binary" prepare-cohort "$destination"
jq -r '.cases[] | [.mode,.task,.architecture,.transport,("d"+(.depth|tostring))] | join("--")' \
  "$destination/inventory.json" > "$destination/cells.txt"
while IFS= read -r cell; do
  /usr/bin/time -v -o "$destination/$cell.resources" \
    bash experiments/convex_stress_hull/limited.sh 8 "$binary" run-cell "$destination" "$cell" \
    > "$destination/$cell.log" 2>&1
  /usr/bin/time -v -o "$destination/$cell-replay.resources" \
    bash experiments/convex_stress_hull/limited.sh 8 "$binary" replay-cell "$destination" "$cell" \
    > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
done < "$destination/cells.txt"
bash experiments/convex_stress_hull/limited.sh 8 "$binary" summarize-cohort "$destination" > "$destination/summary.json"
