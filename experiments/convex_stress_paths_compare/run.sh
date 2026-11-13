#!/usr/bin/env bash
# Run only after all numerical parent reviews and fixture preparation are done.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: run.sh REQUEST FRESH_ROOT' >&2; exit 2; }
binary=${PATHS_COMPARE_BINARY:-experiments/convex_stress_paths_compare/target/release/convex-stress-paths-compare}
for task in 175 191 225 227; do
  agent-kanban show "$task" | sed -n '1p' | rg -q "^#$task [^ ]+ done "
done
request=$1
destination=$2
test ! -e "$destination"
ulimit -c 0
/usr/bin/time -v -o "$destination-prepare.resources" prlimit --as=68719476736:68719476736 -- "$binary" prepare "$request" "$destination"
/usr/bin/time -v -o "$destination-replay.resources" prlimit --as=68719476736:68719476736 -- "$binary" replay "$destination" > "$destination/replay.json"
