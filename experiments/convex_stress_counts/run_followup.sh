#!/usr/bin/env bash
# A separate prospective inventory, evaluated with the unchanged counts binary.
set -euo pipefail
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
counts_binary="$experiment/target/release/convex-stress-counts"
counts_cohort=${1:-"$experiment/runs/remaining-v1"}
counts_workers=${2:-8}
counts_fingerprint=$("$counts_binary" fingerprint)
test "$counts_fingerprint" = "$(jq -r '.expected_fingerprint' "$experiment/followup.json")"
mkdir -- "$counts_cohort"
printf '%s\n' "$counts_fingerprint" > "$counts_cohort/source_fingerprint.txt"
jq -r '.pilot_tasks[] as $task | .arms[] as $arm | .scales[] as $scale |
    [$arm,$task,.pilot_seed,$scale] | @tsv' "$experiment/followup.json" > "$counts_cohort/jobs.tsv"
export counts_binary counts_cohort
xargs -P "$counts_workers" -n 4 bash -c '
    cell="$counts_cohort/scale$4/$1/$2/seed$3"
    mkdir -p -- "${cell%/*}"
    if "$counts_binary" run "$1" "$2" "$3" "$4" "$cell" > "$cell.log" 2>&1; then
        printf "0\n" > "$cell.exit"
        if "$counts_binary" replay "$cell" > "$cell.replay.log" 2>&1; then
            printf "0\n" > "$cell.replay.exit"
        else
            printf "%s\n" "$?" > "$cell.replay.exit"
        fi
    else
        printf "%s\n" "$?" > "$cell.exit"
    fi
' _ < "$counts_cohort/jobs.tsv"
