#!/usr/bin/env bash
# Serial frozen construction and independent raw-state replay under 64 GiB.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'usage: run_cohort.sh FRESH_ROOT' >&2; exit 2; fi
: "${TAGGED_PRODUCER_COMMIT:?set the committed producer}"
binary=${TAGGED_BINARY:-experiments/convex_stress_tagged/cohort/target/release/convex-tagged-cohort}
destination=$1
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
/usr/bin/time -v -o "$destination-prepare.resources" prlimit --as=68719476736:68719476736 -- "$binary" prepare "$destination"
jq -r '.tasks[]' "$destination/inventory.json" > "$destination/tasks.txt"
while IFS= read -r task; do
  available=$(df --output=avail -B1 "$destination" | tail -n 1)
  if (( available < 17179869184 )); then echo 'Disk reserve reached; preserve results and stop.' >&2; exit 2; fi
  code=0
  /usr/bin/time -v -o "$destination/$task.resources" prlimit --as=68719476736:68719476736 -- "$binary" run-cell "$destination" "$task" > "$destination/$task.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$destination/$task.run.exit"
  if (( code == 0 )); then
    code=0
    /usr/bin/time -v -o "$destination/$task-replay.resources" prlimit --as=68719476736:68719476736 -- "$binary" replay-cell "$destination" "$task" > "$destination/$task.replay.json" 2> "$destination/$task-replay.log" || code=$?
    printf '%s\n' "$code" > "$destination/$task.replay.exit"
  fi
done < "$destination/tasks.txt"
prlimit --as=68719476736:68719476736 -- "$binary" summarize "$destination" > "$destination/summary.json"
