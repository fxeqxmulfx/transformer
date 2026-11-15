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
