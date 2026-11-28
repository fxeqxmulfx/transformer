#!/usr/bin/env bash
# Apply one new WSD budget to every seed from an existing registration.
set -euo pipefail
if (( $# < 3 || $# > 4 )) || [[ ! $3 =~ ^[1-9][0-9]*$ ]]; then
    echo 'usage: bash run_division_horizon.sh SOURCE_COHORT OUTPUT_COHORT UPDATES [EXISTING_FIRST_CELL]' >&2
    exit 2
fi
dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bin=${ARGAND_BIN:-"$dir/target/release/argand"}
source=$1
out=$2
steps=$3
case_name=grok-division-p97
mkdir -p -- "$out"
exec 9>"$out/horizon.lock"
flock -x 9
tmp=$(mktemp "$out/protocol.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT
jq -e --argjson steps "$steps" --arg name "$case_name" '
    if .case==$name and .correction=="adamw-wsd" and (.seeds|length)==.n and .n==12
    then .updates=$steps else error("invalid source cohort") end' "$source/protocol.json" > "$tmp"
if [[ -e "$out/protocol.json" ]]; then
    jq -es '.[0]==.[1]' "$tmp" "$out/protocol.json" >/dev/null
else
    mv -- "$tmp" "$out/protocol.json"
fi
rm -f -- "$tmp"
trap - EXIT
"$bin" adamw-confirm-plan "$case_name" "$out" "$steps"
if (( $# == 4 )); then
    # The ordinary single-run command and the confirmation trainer use the
    # same AdamW identity, file lock, terminal horizon and independent replay.
    first=$(realpath -e -- "$4")
    seed=$(jq -r '.seeds[0][0]' "$out/protocol.json")
    head -n 1 "$first/observations.jsonl" |
        jq -e --argjson seed "$seed" --argjson steps "$steps" --arg name "$case_name" '
        .identity.seed==$seed and .identity.budget==$steps and .identity.fixed_horizon and
        .identity.optimizer=="adamw-wsd" and .identity.case==$name' >/dev/null
    if [[ -e "$out/draw-0" || -L "$out/draw-0" ]]; then
        [[ "$out/draw-0" -ef "$first" ]]
    else
        ln -sr -- "$first" "$out/draw-0"
    fi
fi
# The frozen trainer saves endpoint weights and observations, while AdamW's
# moments and shuffle state live in memory. Keep interrupted prefixes as
# separate attempts before starting the same registered draw from step zero.
for ((i=0; i<12; i++)); do
    cell="$out/draw-$i"
    staging="$cell/interrupted/.moving"
    [[ ! -e "$cell/result.json" && ( -s "$cell/observations.jsonl" || -d "$staging" ) ]] || continue
    (
        flock -n 8 || exit 0
        [[ ! -e "$cell/result.json" ]] || exit 0
        mkdir -p -- "$staging"
        for file in "$cell"/first-strict* "$cell"/weights.f32 "$cell"/*.pending "$cell"/observations.jsonl; do
            [[ -f "$file" ]] || continue
            mv -- "$file" "$staging/"
        done
        last=$(sed '${/}$/!d;}' "$staging/observations.jsonl" | jq -sr 'last.step // 0')
        archive=$(mktemp -d "$cell/interrupted/step-$last.XXXXXXXX")
        mv -T -- "$staging" "$archive"
        echo "Archived the interrupted draw $i through update $last in $archive"
    ) 8>>"$cell/cell.lock"
done
exec "$bin" adamw-confirm "$case_name" "$out" "$steps"
