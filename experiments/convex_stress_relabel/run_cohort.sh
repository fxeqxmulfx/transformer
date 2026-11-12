#!/usr/bin/env bash
# Serial training and raw audits, after both frozen depth cohorts complete.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: run_cohort.sh FRESH_COHORT_DIRECTORY' >&2; exit 2; fi
: "${RELABEL_PRODUCER_COMMIT:?set the committed relabel producer}"
: "${DEPTH_REFERENCE_BINARY:?set the frozen original depth binary}"
: "${DEPTH_RETAINED_BINARY:?set the frozen retained depth binary}"
binary=${RELABEL_BINARY:-experiments/convex_stress_relabel/target/release/convex-stress-relabel}
reference=${DEPTH_REFERENCE_ROOT:-experiments/convex_stress_depth/runs/cohort-v1}
retained=${DEPTH_RETAINED_ROOT:-experiments/convex_stress_depth/runs/retained-v1}
destination=$1
if [[ -e "$destination" || -e "$destination-reference-summary.json" || -e "$destination-retained-summary.json" ]]; then
  echo 'Fresh destination and gate summaries are required.' >&2; exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
original_fp=d6931b95c2b1d4971c02b718772fd08105c554d2d0bb66b99aedbf1b07b3e56f
retained_fp=5f95365da8434756d09b79ad485aba32e4cd2addcfec7a76f9571a8e5ab92e33
if [[ $("$DEPTH_REFERENCE_BINARY" cohort-fingerprint) != "$original_fp" || $("$DEPTH_RETAINED_BINARY" cohort-fingerprint) != "$retained_fp" ]]; then
  echo 'Frozen depth producer fingerprint mismatch.' >&2; exit 2
fi
bash experiments/convex_stress_hull/limited.sh 8 "$DEPTH_REFERENCE_BINARY" summarize-cohort "$reference" > "$destination-reference-summary.json"
jq -e --arg fp "$original_fp" '.source_fingerprint==$fp and .cells==288 and (.complete+.memory_ineligible)==288 and (.choices|length)==72' "$destination-reference-summary.json" > /dev/null
bash experiments/convex_stress_hull/limited.sh 8 "$DEPTH_RETAINED_BINARY" summarize-cohort "$retained" > "$destination-retained-summary.json"
jq -e --arg fp "$retained_fp" '.source_fingerprint==$fp and .cells==48 and (.complete+.memory_ineligible)==48 and (.choices|length)==48' "$destination-retained-summary.json" > /dev/null
/usr/bin/time -v -o "$destination-prepare.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" prepare "$destination"
jq -r '.cases[] | [.arm,.task,.coding,.budget] | join("--")' "$destination/inventory.json" > "$destination/cells.txt"
while IFS= read -r cell; do
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < 16 * 1024 * 1024 * 1024 )); then echo 'Disk reserve reached; preserve outcomes and stop.' >&2; exit 2; fi
  /usr/bin/time -v -o "$destination/$cell.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" run-cell "$destination" "$cell" > "$destination/$cell.log" 2>&1
  /usr/bin/time -v -o "$destination/$cell-replay.resources" bash experiments/convex_stress_hull/limited.sh 8 "$binary" replay-cell "$destination" "$cell" > "$destination/$cell.replay.json" 2> "$destination/$cell-replay.log"
done < "$destination/cells.txt"
bash experiments/convex_stress_hull/limited.sh 8 "$binary" summarize "$destination" > "$destination/summary.json"
