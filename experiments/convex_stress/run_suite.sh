#!/usr/bin/env bash
# Reproduce every frozen task/seed for both learners; retain every cell's logs.
set -euo pipefail
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stress_binary="$experiment/target/release/convex-stress"
stress_cohort=${1:-"$experiment/runs/stress-v1"}
stress_workers=${2:-4}
mkdir -- "$stress_cohort"
"$stress_binary" verify > "$stress_cohort/preflight.json" 2> "$stress_cohort/preflight.log"
jq -r '.tasks[].name as $task | ["frank_wolfe","forward_only"][] as $arm | range(0;3) as $seed | [$arm,$task,$seed] | @tsv' \
    "$experiment/suite.json" > "$stress_cohort/jobs.tsv"
export stress_binary stress_cohort
xargs -P "$stress_workers" -n 3 bash -c '
    arm=$1
    task=$2
    seed=$3
    parent="$stress_cohort/$arm/$task"
    mkdir -p -- "$parent"
    if "$stress_binary" run "$arm" "$task" "$seed" "$parent/seed$seed" > "$parent/seed$seed.log" 2>&1; then
        printf "0\n" > "$parent/seed$seed.exit"
    else
        result=$?
        printf "%s\n" "$result" > "$parent/seed$seed.exit"
    fi
' _ < "$stress_cohort/jobs.tsv"
# Missing or failed cells make report fail; no partial pass denominator is used.
"$stress_binary" report "$stress_cohort" > "$stress_cohort/summary.json"
xargs -P "$stress_workers" -n 3 bash -c '
    arm=$1
    task=$2
    seed=$3
    parent="$stress_cohort/$arm/$task"
    "$stress_binary" replay "$parent/seed$seed" > "$parent/seed$seed.replay.log" 2>&1
' _ < "$stress_cohort/jobs.tsv"
