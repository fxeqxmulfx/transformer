#!/usr/bin/env bash
set -euo pipefail
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
completion_binary="$experiment/target/release/convex-stress-completion"
completion_cohort=${1:-"$experiment/runs/associative-v1"}
completion_fingerprint=$("$completion_binary" fingerprint)
test "$completion_fingerprint" = "$(jq -r '.expected_fingerprint' "$experiment/followup.json")"
mkdir -- "$completion_cohort"
printf '%s\n' "$completion_fingerprint" > "$completion_cohort/source_fingerprint.txt"
jq -r '.pilot_tasks[] as $task | .arms[] as $arm | .scales[] as $scale |
    [$arm,$task,.pilot_seed,$scale] | @tsv' "$experiment/followup.json" > "$completion_cohort/jobs.tsv"
while IFS=$'\t' read -r arm task seed scale; do
    cell="$completion_cohort/scale$scale/$arm/$task/seed$seed"
    mkdir -p -- "${cell%/*}"
    if "$completion_binary" run "$arm" "$task" "$seed" "$scale" "$cell" > "$cell.log" 2>&1; then
        printf '0\n' > "$cell.exit"
        if "$completion_binary" replay "$cell" > "$cell.replay.log" 2>&1; then
            printf '0\n' > "$cell.replay.exit"
        else
            printf '%s\n' "$?" > "$cell.replay.exit"
        fi
    else
        printf '%s\n' "$?" > "$cell.exit"
    fi
done < "$completion_cohort/jobs.tsv"
