#!/usr/bin/env bash
# Preserve the full planned population after an explicit user interruption.
set -euo pipefail
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cohort=${1:-"$experiment/runs/minibatch-v1"}
old=${2:-"$experiment/../convex_stress/runs/stress-v1/summary.json"}
jq -e '.status == "stopped_by_user" and .complete_cohort == false' \
    "$cohort/interruption.json" > /dev/null
while IFS=$'\t' read -r arm task seed; do
    cell="$cohort/$arm/$task/seed$seed"
    if [[ ! -f "$cell/result.json" ]]; then
        jq -s --slurpfile meta "$cell/metadata.json" \
            '{arm:$meta[0].arm,task:$meta[0].task,tier:$meta[0].tier,seed:$meta[0].seed,last_observation:.[-1]}' \
            "$cell/trace.jsonl"
    fi
done < "$cohort/jobs.tsv" | jq -s '.' > "$cohort/interrupted_cells.json"
jq -s --slurpfile old "$old" --slurpfile interrupted "$cohort/interrupted_cells.json" \
    -f "$experiment/compare.jq" "$cohort"/*/*/seed*/result.json > "$cohort/partial_comparison.json"
