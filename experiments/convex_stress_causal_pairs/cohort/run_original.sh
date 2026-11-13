#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: run_original.sh REQUEST FRESH_ROOT' >&2; exit 2; }
for task in 232 233 234; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
binary=${CAUSAL_PAIRS_BINARY:-experiments/convex_stress_causal_pairs/cohort/target/release/convex-stress-causal-pairs-cohort}
request=$1
destination=$2
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
available=$(df -Pk "$(dirname "$destination")" | awk 'NR==2 {print $4}')
test "$available" -ge 16777216 || { echo 'less than 16 GiB disk reserve' >&2; exit 1; }
/usr/bin/time -v -o "$destination-prepare.resources" prlimit --as=68719476736:68719476736 -- "$binary" prepare-original "$request" "$destination"
while IFS= read -r task; do
  available=$(df -Pk "$destination" | awk 'NR==2 {print $4}')
  test "$available" -ge 16777216 || { echo 'less than 16 GiB disk reserve' >&2; exit 1; }
  status=0
  /usr/bin/time -v -o "$destination/$task.run.resources" prlimit --as=68719476736:68719476736 -- "$binary" run-original-case "$destination" "$task" > "$destination/$task.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$destination/$task.run.exit"
  if test "$status" = 0; then
    status=0
    /usr/bin/time -v -o "$destination/$task.replay.resources" prlimit --as=68719476736:68719476736 -- "$binary" replay-original-case "$destination" "$task" > "$destination/$task.replay.json" 2> "$destination/$task.replay.log" || status=$?
    printf '%s\n' "$status" > "$destination/$task.replay.exit"
  fi
done < <(jq -r '.tasks[]' "$destination/inventory.json")
/usr/bin/time -v -o "$destination/summary.resources" prlimit --as=68719476736:68719476736 -- "$binary" summarize-original "$destination" > "$destination/summary.json"
