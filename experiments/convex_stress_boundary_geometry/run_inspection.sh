#!/usr/bin/env bash
# Run only from a frozen source tree. Preserve all six roles and every exit.
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
require_hash shared_geometry_summary_sha256 "$storage/shared-geometry/inspection-v2/summary.json"
require_hash raw_comparison_summary_sha256 "$storage/shared-geometry-compare/original-v1/summary.json"
cp --reflink=auto "$storage/shared-geometry/inspection-v2/summary.json" "$root/retained331.json"
cp --reflink=auto "$storage/shared-geometry-compare/original-v1/summary.json" "$root/retained332.json"
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
