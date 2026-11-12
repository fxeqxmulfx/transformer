#!/usr/bin/env bash
# Posthoc attribution control; never a new version of the counts trainer.
set -euo pipefail
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
control_binary="$experiment/../convex_stress_completion/target/release/convex-stress-completion"
control_cohort=${1:-"$experiment/runs/completion-control-v1"}
control_fingerprint=$("$control_binary" fingerprint)
test "$control_fingerprint" = "$(jq -r '.expected_fingerprint' "$experiment/completion_control.json")"
mkdir -- "$control_cohort"
printf '%s\n' "$control_fingerprint" > "$control_cohort/source_fingerprint.txt"
jq -r '.pilot_tasks[] as $task | .arms[] as $arm | .scales[] as $scale |
    [$arm,$task,.pilot_seed,$scale] | @tsv' "$experiment/completion_control.json" > "$control_cohort/jobs.tsv"
export control_binary control_cohort
xargs -P 6 -n 4 bash -c '
    cell="$control_cohort/scale$4/$1/$2/seed$3"
    mkdir -p -- "${cell%/*}"
    if "$control_binary" run "$1" "$2" "$3" "$4" "$cell" > "$cell.log" 2>&1; then
        printf "0\n" > "$cell.exit"
        if "$control_binary" replay "$cell" > "$cell.replay.log" 2>&1; then
            printf "0\n" > "$cell.replay.exit"
        else
            printf "%s\n" "$?" > "$cell.replay.exit"
        fi
    else
        printf "%s\n" "$?" > "$cell.exit"
    fi
' _ < "$control_cohort/jobs.tsv"
