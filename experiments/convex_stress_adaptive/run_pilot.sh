#!/usr/bin/env bash
# Keep every prespecified pilot arm/task/scale, including failures and replays.
set -euo pipefail
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repair_binary="$experiment/target/release/convex-stress-adaptive"
repair_cohort=${1:-"$experiment/runs/pilot-v1"}
repair_workers=${2:-4}
mkdir -- "$repair_cohort"
"$repair_binary" fingerprint > "$repair_cohort/source_fingerprint.txt"
jq -r '.pilot_tasks[] as $task | .arms[] as $arm | .scales[] as $scale |
    [$arm,$task,.pilot_seed,$scale] | @tsv' "$experiment/protocol.json" > "$repair_cohort/jobs.tsv"
export repair_binary repair_cohort
xargs -P "$repair_workers" -n 4 bash -c '
    cell="$repair_cohort/scale$4/$1/$2/seed$3"
    mkdir -p -- "${cell%/*}"
    if "$repair_binary" run "$1" "$2" "$3" "$4" "$cell" > "$cell.log" 2>&1; then
        printf "0\n" > "$cell.exit"
        if "$repair_binary" replay "$cell" > "$cell.replay.log" 2>&1; then
            printf "0\n" > "$cell.replay.exit"
        else
            printf "%s\n" "$?" > "$cell.replay.exit"
        fi
    else
        printf "%s\n" "$?" > "$cell.exit"
    fi
' _ < "$repair_cohort/jobs.tsv"
