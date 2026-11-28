#!/usr/bin/env bash
# Run each canonical case in manifest order. The binary replays each endpoint.
set -euo pipefail
if (( $# < 2 || $# > 3 )) || [[ ! $1 =~ ^(short|long|all)$ ]] || [[ ! ${3:-single} =~ ^(single|sqrt|retry|adamw|adamw-scaled|equal-moment|uncorrected-adam|normalized-adamw|first-noise|first-noise-perfect)$ ]]; then
    echo 'usage: bash run_page_screen.sh TIER RUN_DIRECTORY [single|sqrt|retry|adamw|adamw-scaled|equal-moment|uncorrected-adam|normalized-adamw|first-noise|first-noise-perfect]' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${ARGAND_BIN:-"$dir/target/release/argand"}
settings=()
case ${3:-single} in
    single) op=page-run ;;
    sqrt|retry) op=page-run-$3 ;;
    adamw) op=adamw-run ;;
    adamw-scaled|equal-moment|uncorrected-adam|normalized-adamw) op=$3-until-perfect; settings=(3) ;;
    first-noise) op=adamw-first-noise-screen ;;
    first-noise-perfect) op=adamw-first-noise-until-perfect ;;
esac
rows=$(jq -er --arg tier "$1" '.seeds as $s | .tasks[] | select($tier == "all" or .tier == $tier) | .name as $name | $s[] | [$name,.] | @tsv' "$dir/../convex_stress/suite.json")
files=()
# Bash reads the training and aggregation commands together before the
# first GPU job, so a later edit of this script cannot interrupt the report.
{
    while IFS=$'\t' read -r name seed; do
        cell="$2/$name/seed$seed"
        noise=()
        if [[ $op == adamw-first-noise-* ]]; then
            noise=("$((371+seed))")
            cell+="-noise${noise[0]}"
        fi
        "$bin" "$op" "$name" "$seed" "${noise[@]}" "$2" 1000 "${settings[@]}"
        files+=("$cell/result.json" "$cell/replay.json")
    done <<< "$rows"
    jq -es --slurpfile suite "$dir/../convex_stress/suite.json" --arg tier "$1" \
        -f "$dir/page_screen_summary.jq" "${files[@]}" > "$2/$1-summary.pending"
    mv -- "$2/$1-summary.pending" "$2/$1-summary.json"
    exit 0
}
