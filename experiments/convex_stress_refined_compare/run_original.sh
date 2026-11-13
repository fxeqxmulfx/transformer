#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: run_original.sh REQUEST FRESH_ROOT' >&2; exit 2; }
for task in 232 234 237 243 244 245; do agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "; done
binary=${REFINED_COMPARE_BINARY:-experiments/convex_stress_refined_compare/target/release/convex-stress-refined-compare}
request=$1
destination=$2
test ! -e "$destination"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
reserve() { local available; available=$(df -Pk "$1" | awk 'NR==2 {print $4}'); test "$available" -ge 16777216 || { echo 'less than16GiB disk reserve' >&2; exit 1; }; }
reserve "$(dirname "$destination")"
/usr/bin/time -v -o "$destination-prepare.resources" prlimit --as=68719476736:68719476736 -- "$binary" prepare "$request" "$destination"
mv "$destination-prepare.resources" "$destination/prepare.resources"
while IFS= read -r task; do
  reserve "$destination"
  status=0
  /usr/bin/time -v -o "$destination/$task.run.resources" prlimit --as=68719476736:68719476736 -- "$binary" run-case "$destination" "$task" > "$destination/$task.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$destination/$task.run.exit"
  if test "$status" = 0; then
    status=0
    /usr/bin/time -v -o "$destination/$task.replay.resources" prlimit --as=68719476736:68719476736 -- "$binary" replay-case "$destination" "$task" > "$destination/$task.replay.json" 2> "$destination/$task.replay.log" || status=$?
    printf '%s\n' "$status" > "$destination/$task.replay.exit"
  fi
done < <(jq -r '.tasks[]' "$destination/inventory.json")
/usr/bin/time -v -o "$destination/summary.resources" prlimit --as=68719476736:68719476736 -- "$binary" summarize "$destination" > "$destination/summary.json"
