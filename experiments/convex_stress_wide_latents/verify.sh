#!/usr/bin/env bash
# Rehash every complete artifact after the independent rebuild/repeat.
set -euo pipefail
test "$#" -eq 3
original=$(realpath "$1")
repeat=$(realpath "$2")
output=$3
test ! -e "$output"
test "$(cat "$original/process.exit")" = 0
test "$(cat "$repeat/process.exit")" = 0
cmp "$original/producer.commit" "$repeat/producer.commit"
(cd "$original" && sha256sum -c manifest.sha256) > "$original/review-rehash.log"
(cd "$repeat" && sha256sum -c manifest.sha256) > "$repeat/review-rehash.log"
cmp "$original/manifest.sha256" "$repeat/manifest.sha256"
test "$(wc -l < "$original/manifest.sha256")" -eq 102
jq -e '.original_cases==54 and .available_cases==48 and .unavailable_source_cases==6 and (.cases|unique_by(.case)|length)==48 and .heldout_pools_read==0 and .accuracy_measurements==0 and .training_updates==0' "$original/summary.json" > /dev/null
jq -n --arg producer "$(cat "$original/producer.commit")" \
    --arg manifest "$(sha256sum "$original/manifest.sha256" | cut -d' ' -f1)" \
    '{producer_commit:$producer,manifest_sha256:$manifest,artifacts:102,
      independent_repeat_exact:true,heldout_pools_read:0,accuracy_measurements:0}' > "$output"
