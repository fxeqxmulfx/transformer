#!/usr/bin/env bash
# Apply the predeclared 200k decision to the twelve terminal 100k replays.
set -euo pipefail
if (( $# != 2 )); then
    echo 'usage: bash double_division_horizon.sh COHORT_100K COHORT_200K' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
primary=$1
out=$2
flock -s "$primary/horizon.lock" true
result="$primary/confirmation.json"
jq -e --slurpfile plan "$primary/protocol.json" '
    .protocol==$plan[0] and .protocol.case=="grok-division-p97" and
    .protocol.updates==100000 and .protocol.n==12' "$result" >/dev/null
cells=()
for ((i=0;i<12;i++)); do cells+=("$primary/draw-$i/replay.json"); done
k=$(jq -es --slurpfile plan "$primary/protocol.json" '
    if length==12 and all(to_entries[];
      .value.step==100000 and .value.identity.fixed_horizon and
      .value.identity.seed==$plan[0].seeds[.key][0] and .value.strict.total==4656)
    then map(select(.strict.all_strict and .splits.hidden.correct==4656)) | length
    else error("missing or incompatible terminal replays") end' "${cells[@]}")
jq -e --argjson k "$k" '.successes==$k' "$result" >/dev/null
if (( k < 12 )); then
    echo "$k/12 terminal successes at 100000 updates; running all twelve seeds to 200000"
    bash "$dir/run_division_horizon.sh" "$primary" "$out" 200000
else
    echo '12/12 terminal successes at 100000; the second cohort is unnecessary'
fi
