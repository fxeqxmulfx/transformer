#!/usr/bin/env bash
# Run an already frozen binary. Logs/resources belong beside, not inside, root.
set -euo pipefail
if test "$#" -ne 5; then
    printf '%s\n' 'usage: run.sh BINARY WORKSPACE NEW_OUTPUT PREPARED_DATA PRODUCER_COMMIT' >&2
    exit 2
fi
binary=$1
workspace=$2
output=$3
data=$4
producer=$5
test ! -e "$output"
"$binary" prepare "$workspace" "$output" "$data" "$producer"
mapfile -t cases < <(jq -r '.cases[].source.case.name' "$output/data/inventory.json")
test "${#cases[@]}" -eq 6
for case_name in "${cases[@]}"; do
    printf 'stage %s\n' "$case_name"
    "$binary" stage "$workspace" "$output" "$case_name"
done
"$binary" seal "$workspace" "$output"
for case_name in "${cases[@]}"; do
    printf 'score %s\n' "$case_name"
    "$binary" score "$workspace" "$output" "$case_name"
done
for case_name in "${cases[@]}"; do
    printf 'replay %s\n' "$case_name"
    "$binary" replay "$workspace" "$output" "$case_name"
done
