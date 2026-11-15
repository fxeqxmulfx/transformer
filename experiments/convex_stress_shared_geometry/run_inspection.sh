#!/usr/bin/env bash
# Invoke from the immutable freeze. Attempt all six original training roles.
set -euo pipefail
test "$#" -eq 3
binary=$(realpath "$1")
parent=$2
root=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test ! -e "$root"
mkdir -p "$root"
cp "$script_dir/inspection-inputs.json" "$root/inventory.json"
storage=$(realpath "$parent/../../..")
require_hash() {
    local key=$1 file=$2 expected actual
    expected=$(jq -r --arg key "$key" '.[$key]' "$root/inventory.json")
    actual=$(sha256sum "$file")
    test "${actual%% *}" = "$expected"
    printf '%s\n' "$actual" >> "$root/references.sha256"
}
require_hash baseline_report_sha256 "$storage/signed-compare/report-v2.json"
require_hash baseline_complete_summary_sha256 "$storage/signed-compare/original-v2/summary.json"
require_hash shared_head_inspection_summary_sha256 "$storage/shared-head/inspection-v1/summary.json"
require_hash comparison_summary_sha256 "$storage/shared-head-compare/original-v2/summary.json"
require_hash previous_geometry_summary_sha256 "$storage/shared-geometry/inspection-v1/summary.json"
status=0
while IFS=$'\t' read -r file mode hash; do
    for role in prior supplement; do
        name=$mode-$role
        set +e
        /usr/bin/time -v -o "$root/$name.resources" prlimit --as=68719476736:68719476736 \
            -- "$binary" inspect "$parent/$file" "$hash" "$role" "$mode" "$root/$name" \
            > "$root/$name.log" 2>&1
        result=$?
        set -e
        if [[ "$result" -eq 0 ]]; then
            set +e
            prlimit --as=68719476736:68719476736 -- jq -e \
                --slurpfile old "$storage/shared-geometry/inspection-v1/$name/outcome.json" \
                -f "$script_dir/retained_candidates.jq" "$root/$name/outcome.json" \
                > "$root/$name/retained-v1.json" 2>> "$root/$name.log"
            result=$?
            set -e
        fi
        printf '%s\n' "$result" > "$root/$name.exit"
        if [[ "$result" -ne 0 ]]; then status=1; fi
    done
done < <(jq -r '.models[] | [.file,.mode,.sha256] | @tsv' "$root/inventory.json")
printf '%s\n' "$status" > "$root/process.exit"
if [[ "$status" -eq 0 ]]; then
    jq -s '{complete:true,inspections:.,original_heldout_evaluations:0}' \
        "$root"/*/summary.json > "$root/summary.json"
fi
exit "$status"
