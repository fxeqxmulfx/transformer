#!/usr/bin/env bash
# Freeze this script with the producer. Retain every process outcome and refusal.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: run_cohort.sh FRESH_ROOT' >&2; exit 2; fi
: "${GENERAL_BANK_PRODUCER_COMMIT:?set the committed producer}"
binary=${GENERAL_BANK_BINARY:-experiments/convex_stress_general_bank/target/release/convex-stress-general-bank}
destination=$(realpath -m "$1")
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/bin/time -v -o "$destination-prepare.resources" bash experiments/convex_stress_general_bank/limited.sh "$binary" prepare "$destination"
jq -r '.cases[] | .arm + "--" + .task' "$destination/inventory.json" > "$destination/cells.txt"
while IFS= read -r cell; do
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < 16 * 1024 * 1024 * 1024 )); then echo 'Disk reserve reached; preserve outcomes and stop.' >&2; exit 2; fi
  set +e
  /usr/bin/time -v -o "$destination/$cell.resources" bash experiments/convex_stress_general_bank/limited.sh "$binary" run-cell "$destination" "$cell" > "$destination/$cell.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$destination/$cell.run.exit"
  if (( status != 0 )); then continue; fi
  set +e
  /usr/bin/time -v -o "$destination/$cell-replay.resources" bash experiments/convex_stress_general_bank/limited.sh "$binary" replay-cell "$destination" "$cell" > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
  status=$?
  set -e
  printf '%s\n' "$status" > "$destination/$cell.replay.exit"
done < "$destination/cells.txt"
bash experiments/convex_stress_general_bank/limited.sh "$binary" summarize "$destination" > "$destination/summary.json"
