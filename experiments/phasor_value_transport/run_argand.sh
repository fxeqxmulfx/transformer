#!/usr/bin/env bash
# Train frozen GPT cases with Argand and replay each passing checkpoint.
set -euo pipefail

usage='usage: run_argand.sh TIER REPEAT RUN_DIR [MAX_UPDATES]'
if (( $# < 3 || $# > 4 )); then
    echo "$usage" >&2
    exit 2
fi
tier=$1
shift
if [[ $tier != short && $tier != long && $tier != all ]]; then
    echo 'TIER must be short, long or all' >&2
    exit 2
fi
if [[ $1 != [1-9]* || ${1//[0-9]/} ]]; then
    echo "$usage" >&2
    exit 2
fi
repeat=$1
out=$2
cap=${3:-0}
if [[ $cap != 0 && ( $cap != [1-9]* || ${cap//[0-9]/} ) ]]; then
    echo 'MAX_UPDATES must be a positive integer' >&2
    exit 2
fi

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${PHASOR_BIN:-"$dir/target/release/phasor-value-transport"}
suite="$dir/../convex_stress/suite.json"
rows=$(jq -er --arg tier "$tier" '.seeds as $seeds |
    (.tasks[] | select($tier == "all" or .tier == $tier)) as $task | $seeds[] |
    [$task.name, ., $task.case.max_updates, $task.case.observe] | @tsv' "$suite")
while IFS=$'\t' read -r case_name seed budget every; do
    stop=$budget
    if (( cap > 0 && cap < budget )); then
        stop=$((cap / every * every))
        if (( stop == 0 )); then
            echo "MAX_UPDATES precedes the first positive observation for $case_name" >&2
            exit 2
        fi
    fi
    dest="$out/$case_name/seed$seed"
    mkdir -p "$dest"
    result=$("$bin" argand-run "$case_name" "$seed" "$repeat" "$out" "$stop" 2>> "$dest/observations.jsonl")
    printf '%s\n' "$result"
    if jq -e '.passed' <<< "$result" >/dev/null; then
        "$bin" argand-replay "$case_name" "$seed" "$repeat" "$out" > "$dest/replay.pending"
        mv -- "$dest/replay.pending" "$dest/replay.json"
    fi
done <<< "$rows"
