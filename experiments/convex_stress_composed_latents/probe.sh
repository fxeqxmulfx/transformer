#!/usr/bin/env bash
# Reuse only the independently verified original training extracts. No heldout JSON is read.
set -euo pipefail
test "$#" -eq 7
binary=$(realpath "$1")
workspace=$(realpath "$2")
storage=$(realpath "$3")
output=$(realpath -m "$4")
predecessor=$(realpath "$5")
verification=$(realpath "$6")
producer=$7
protocol=$(dirname "$(realpath "$0")")/protocol.json
export LC_ALL=C
test ! -e "$output"
test -x "$binary"
[[ "$output" == "$storage/"* ]]
ancestor=$output
while [[ ! -e "$ancestor" ]]; do ancestor=$(dirname "$ancestor"); done
test "$(stat -c %d "$ancestor")" = "$(stat -c %d "$storage")"
[[ "$producer" =~ ^[0-9a-f]{40}$ ]]
test "$(sha256sum "$predecessor/manifest.sha256" | cut -d' ' -f1)" = "$(jq -r .predecessor_manifest_sha256 "$protocol")"
test "$(sha256sum "$verification" | cut -d' ' -f1)" = "$(jq -r .predecessor_verification_sha256 "$protocol")"
test "$(sha256sum "$predecessor/data348-inventory.json" | cut -d' ' -f1)" = "$(jq -r .data348_inventory_sha256 "$protocol")"
test "$(sha256sum "$predecessor/data348-verified.json" | cut -d' ' -f1)" = "$(jq -r .data348_verification_sha256 "$protocol")"
prlimit --as=110000000000:110000000000 -- "$binary" check "$workspace" "$storage"
mkdir -p "$output/training" "$output/results" "$output/logs" "$output/resources" "$output/exits"
trap 'code=$?; printf "%s\n" "$code" > "$output/process.exit"' EXIT
(cd "$predecessor" && sha256sum -c manifest.sha256) > "$output/predecessor-check.log"
printf '%s\n' "$producer" > "$output/producer.commit"
sha256sum "$binary" > "$output/binary.sha256"
cp "$predecessor/data348-inventory.json" "$output/data348-inventory.json"
cp "$predecessor/data348-verified.json" "$output/data348-verified.json"
cp "$predecessor/manifest.sha256" "$output/predecessor-manifest.sha256"
cp "$verification" "$output/predecessor-verified.json"
cp "$protocol" "$output/protocol.json"
mapfile -t cases < <(jq -r '.cases[] | select(.data != null) | .case' "$output/data348-inventory.json")
test "${#cases[@]}" -eq 48
for case in "${cases[@]}"; do
    [[ "$case" =~ ^[a-z0-9-]+$ ]]
    prlimit --as=110000000000:110000000000 -- "$binary" check "$workspace" "$storage"
    test "$(stat -c %s "$predecessor/training/$case.json")" -le 8388608
    cp "$predecessor/training/$case.json" "$output/training/$case.json"
    set +e
    /usr/bin/time -v -o "$output/resources/$case.txt" \
        prlimit --as=110000000000:110000000000 -- "$binary" probe \
        "$output/training/$case.json" "$output/results/$case.json" "$workspace" "$storage" \
        > "$output/logs/$case.log" 2>&1
    code=$?
    set -e
    printf '%s\n' "$code" > "$output/exits/$case.exit"
    # A computation or disk failure stops the cohort and leaves its evidence.
    # Empty support and model-memory refusals are complete, successful audit reports.
    if [[ "$code" -ne 0 ]]; then exit "$code"; fi
done
prlimit --as=110000000000:110000000000 -- "$binary" check "$workspace" "$storage"
jq -n --slurpfile inventory "$output/data348-inventory.json" '[inputs |
 {case:(input_filename|split("/")[-1]|rtrimstr(".json")),
  records:.selection.records, retained:(.selection.retained|length),
  known_rows:.selection.audit.known_rows, unknown_rows:.selection.audit.unknown_rows,
  complete_buffer_forecast:.selection.audit.complete_buffer_forecast,
  refusal:.selection.audit.admission_refusal, operations:.work.used,
  heldout_pools_read, accuracy_measurements, training_updates}] as $cases |
 {original_cases:($inventory[0].cases|length), available_cases:($cases|length),
  unavailable_source_cases:([$inventory[0].cases[]|select(.data==null)]|length),
  heldout_pools_read:([$cases[].heldout_pools_read]|add),
  accuracy_measurements:([$cases[].accuracy_measurements]|add),
  training_updates:([$cases[].training_updates]|add),
  nonempty_support:([$cases[]|select(.retained>0)]|length),
  admitted_models:([$cases[]|select(.refusal==null)]|length),
  complete_family_members:962806, cases:$cases}' "$output"/results/*.json > "$output/summary.json"
cd "$output"
sha256sum training/*.json results/*.json protocol.json data348-inventory.json data348-verified.json \
    predecessor-manifest.sha256 predecessor-verified.json summary.json > manifest.sha256
test "$(wc -l < manifest.sha256)" -eq 102
sha256sum -c manifest.sha256 > manifest-check.log
prlimit --as=110000000000:110000000000 -- "$binary" check "$workspace" "$storage"
