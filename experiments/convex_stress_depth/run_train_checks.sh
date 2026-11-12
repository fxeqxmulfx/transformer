#!/usr/bin/env bash
# Run one original task/depth, serially, under the same hard memory ceiling.
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'usage: run_train_checks.sh FRESH_DIRECTORY TASK DEPTH' >&2
  exit 2
fi
: "${DEPTH_PRODUCER_COMMIT:?set the committed producer revision}"
binary=${DEPTH_BINARY:-experiments/convex_stress_depth/target/release/convex-stress-depth}
destination=$1
task=$2
depth=$3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir "$destination"
for mode in full_batch minibatch forward_only; do
  cell="$destination/$mode"
  /usr/bin/time -v -o "$destination/$mode.resources" \
    bash experiments/convex_stress_hull/limited.sh 8 \
    "$binary" train-check "$task" "$depth" "$mode" "$cell" \
    > "$destination/$mode.log" 2>&1
  if [[ $(jq -r .status "$cell/result.json") == training_check_complete ]]; then
    /usr/bin/time -v -o "$destination/$mode-replay.resources" \
      bash experiments/convex_stress_hull/limited.sh 8 \
      "$binary" replay-train "$cell" > "$destination/$mode-replay.json" \
      2> "$destination/$mode-replay.log"
  fi
done
