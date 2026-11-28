#!/usr/bin/env bash
# Stop each registered seed at the first scheduled strictly perfect hidden score.
set -euo pipefail
if (( $# != 3 )) || [[ ! $3 =~ ^[1-9][0-9]*$ ]]; then
    echo 'usage: bash run_division_to_perfect.sh REGISTERED_COHORT OUTPUT_DIRECTORY WSD_BUDGET' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${ARGAND_BIN:-"$dir/target/release/argand"}
registered=$(realpath -e -- "$1")
mkdir -p -- "$2"
out=$(cd -- "$2" && pwd -P)
cap=$3
case_name=grok-division-p97
exec 9>"$out/cohort.lock"
flock -x 9
jq -e --argjson cap "$cap" '
  select(.case=="grok-division-p97" and .correction=="adamw-wsd" and .n==12 and (.seeds|length)==12) |
  {case,correction,n,seeds,updates:$cap,stop:"first-strict-100-percent",
   success:"all 4656 hidden answers strict at one scheduled evaluation by the WSD budget"}' \
  "$registered/protocol.json" > "$out/protocol.pending"
if [[ -f "$out/protocol.json" ]]; then
    jq -es '.[0]==.[1]' "$out/protocol.json" "$out/protocol.pending" >/dev/null
fi
mv -- "$out/protocol.pending" "$out/protocol.json"
records=()
for ((i=0; i<12; i++)); do
    seed=$(jq -r --argjson i "$i" '.seeds[$i][0]' "$out/protocol.json")
    cell="$out/$case_name/seed$seed"
    kind=terminal
    replay=replay.json
    certificate=observed-certificate.json
    if [[ $(jq -r '.updates' "$registered/protocol.json") == "$cap" ]]; then
        # Use the earliest attempt in wall-clock order. An interruption after
        # the first perfect replay already satisfies this stopping rule.
        first=''
        old="$registered/draw-$i"
        for candidate in "$old"/interrupted/step-* "$old"; do
            [[ -s "$candidate/first-strict-replay.json" ]] || continue
            if [[ -z $first || "$candidate/first-strict-replay.json" -ot "$first/first-strict-replay.json" ]]; then
                first=$candidate
            fi
        done
        if [[ -n $first ]]; then
            cell=$(realpath -e -- "$first")
            kind=first-strict
            replay=first-strict-replay.json
            certificate=first-strict-certificate.json
        fi
    fi
    if [[ $kind == terminal && ! -s "$cell/$replay" ]]; then
        mkdir -p -- "$cell"
        (
            # Another trainer can already own this cell. An incomplete
            # attempt cannot resume its in-memory moments and shuffle state.
            flock -n 8 || exit 0
            [[ ! -s "$cell/result.json" && -s "$cell/observations.jsonl" ]] || exit 0
            mkdir -p "$cell/interrupted"
            archive=$(mktemp -d "$cell/interrupted/attempt.XXXXXXXX")
            for f in "$cell"/weights.f32 "$cell"/*.pending "$cell"/observations.jsonl; do
                [[ ! -f "$f" ]] || mv -- "$f" "$archive/"
            done
        ) 8>>"$cell/cell.lock"
        "$bin" adamw-until-perfect "$case_name" "$seed" "$out" "$cap"
    fi
    if [[ ! -s "$cell/$certificate" ]]; then "$bin" division-certificate "$cell" "$kind" >/dev/null; fi
    record="$out/draw-$i.json"
    jq -en --argjson seed "$seed" --argjson i "$i" --argjson cap "$cap" --arg cell "$cell" --arg kind "$kind" \
      --slurpfile r "$cell/$replay" --slurpfile c "$cell/$certificate" -f "$dir/verify_division_first_hit.jq" > "$record.pending"
    mv -- "$record.pending" "$record"
    records+=("$record")
    jq -es --argjson cap "$cap" '
      (map(select(.success))|length) as $k | {updates:$cap,registered:12,evaluated:length,
       successes:$k,empirical_fraction:(if length==12 then $k/12 else null end),draws:.}' \
      "${records[@]}" > "$out/summary.pending"
    mv -- "$out/summary.pending" "$out/summary.json"
    jq -c '{seed,step,success}' "$record"
done
