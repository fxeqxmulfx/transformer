#!/usr/bin/env bash
# Run each canonical case in manifest order. The binary replays each endpoint.
set -euo pipefail
if (( $# < 2 || $# > 3 )) || [[ ! $1 =~ ^(short|long|all)$ ]] || [[ ! ${3:-single} =~ ^(single|sqrt)$ ]]; then
    echo 'usage: bash run_page_screen.sh TIER RUN_DIRECTORY [single|sqrt]' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${ARGAND_BIN:-"$dir/target/release/argand"}
op=page-run
if [[ ${3:-single} == sqrt ]]; then op=page-run-sqrt; fi
rows=$(jq -er --arg tier "$1" '.seeds as $s | .tasks[] | select($tier == "all" or .tier == $tier) | .name as $name | $s[] | [$name,.] | @tsv' "$dir/../convex_stress/suite.json")
files=()
while IFS=$'\t' read -r name seed; do
    "$bin" "$op" "$name" "$seed" "$2" 1000
    files+=("$2/$name/seed$seed/result.json" "$2/$name/seed$seed/replay.json")
done <<< "$rows"
jq -es --slurpfile suite "$dir/../convex_stress/suite.json" --arg tier "$1" \
    -f "$dir/page_screen_summary.jq" "${files[@]}" > "$2/$1-summary.pending"
mv -- "$2/$1-summary.pending" "$2/$1-summary.json"
