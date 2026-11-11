#!/usr/bin/env bash
# Keep every source task/seed and both arms, including failures and their replay.
set -euo pipefail
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stress_binary="$experiment/target/release/convex-stress-minibatch"
stress_cohort=${1:-"$experiment/runs/minibatch-v1"}
stress_workers=${2:-8}
mkdir -- "$stress_cohort"
"$stress_binary" verify > "$stress_cohort/preflight.json" 2> "$stress_cohort/preflight.log"
jq -r '.tasks[].name as $task | ["frank_wolfe","forward_only"][] as $arm | range(0;3) as $seed | [$arm,$task,$seed] | @tsv' \
    "$experiment/../convex_stress/suite.json" > "$stress_cohort/jobs.tsv"
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
"$stress_binary" report "$stress_cohort" > "$stress_cohort/summary.json"
xargs -P "$stress_workers" -n 3 bash -c '
    arm=$1
    task=$2
    seed=$3
    parent="$stress_cohort/$arm/$task"
    "$stress_binary" replay "$parent/seed$seed" > "$parent/seed$seed.replay.log" 2>&1
' _ < "$stress_cohort/jobs.tsv"
