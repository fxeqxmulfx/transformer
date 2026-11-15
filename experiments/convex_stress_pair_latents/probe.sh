#!/usr/bin/env bash
# All48 original training pools, no heldout evaluation or target feature input.
set -euo pipefail
test "$#" -eq 5
binary=$1
output=$2
data=$3
verification=$4
producer=$5
protocol=$(dirname "$(realpath "$0")")/protocol.json
test ! -e "$output"
test -x "$binary"
test "$(sha256sum "$data/inventory.json" | cut -d' ' -f1)" = "$(jq -r .data348_inventory_sha256 "$protocol")"
test "$(sha256sum "$verification" | cut -d' ' -f1)" = "$(jq -r .data348_verification_sha256 "$protocol")"
test "$(jq '[.cases[] | select(.data != null)] | length' "$data/inventory.json")" -eq 48
mkdir -p "$output/training" "$output/results" "$output/logs" "$output/resources" "$output/exits"
printf '%s\n' "$producer" > "$output/producer.commit"
sha256sum "$binary" > "$output/binary.sha256"
cp "$data/inventory.json" "$output/data348-inventory.json"
cp "$verification" "$output/data348-verified.json"
cp "$protocol" "$output/protocol.json"
while IFS=$'\t' read -r case file expected; do
    [[ "$case" =~ ^[a-z0-9-]+$ ]]
    test "$file" = "data/$case.json"
    test "$(sha256sum "$data/$file" | cut -d' ' -f1)" = "$expected"
    jq '.training' "$data/$file" > "$output/training/$case.json"
    set +e
    /usr/bin/time -v -o "$output/resources/$case.txt" \
        prlimit --as=68719476736:68719476736 -- "$binary" encode \
        "$output/training/$case.json" "$output/results/$case.json" > "$output/logs/$case.log" 2>&1
    result=$?
    set -e
    printf '%s\n' "$result" > "$output/exits/$case.exit"
    if [[ "$result" -ne 0 ]]; then
        jq -n --argjson exit "$result" --rawfile error "$output/logs/$case.log" \
            '{process_exit:$exit,refusal:$error,heldout_pools_read:0,accuracy_measurements:0}' > "$output/results/$case.json"
    fi
done < <(jq -r '.cases[] | select(.data != null) | [.case,.data.file,.data.sha256] | @tsv' "$data/inventory.json")
cd "$output"
sha256sum training/*.json results/*.json protocol.json data348-inventory.json data348-verified.json > manifest.sha256
sha256sum -c manifest.sha256 > manifest-check.log
printf '0\n' > process.exit
