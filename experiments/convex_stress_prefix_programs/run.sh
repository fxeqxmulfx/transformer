#!/usr/bin/env bash
# Reconstruct all16 states before any heldout scoring; preserve every exit.
set -euo pipefail
test "$#" -ge 4
binary=$(realpath "$1")
root=$2
mode=$3
source=$4
test ! -e "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$mode" == original ]]; then
    test "$#" -eq 6
    prlimit --as=68719476736:68719476736 -- "$binary" prepare-run "$root" "$source" "$5" "$6"
elif [[ "$mode" == repeat ]]; then
    test "$#" -eq 4
    prlimit --as=68719476736:68719476736 -- "$binary" repeat-run "$source" "$root"
else
    exit 2
fi
mapfile -t cases < <(jq -r '.cases[]' "$root/inventory.json")
test "${#cases[@]}" -eq 16
status=0
for phase in stage score replay; do
    for case in "${cases[@]}"; do
        set +e
        /usr/bin/time -v -o "$root/$case-$phase.resources" \
            prlimit --as=68719476736:68719476736 -- "$binary" "$phase" "$root" "$case" \
            > "$root/$case-$phase.log" 2>&1
        result=$?
        set -e
        printf '%s\n' "$result" > "$root/$case-$phase.exit"
        if [[ "$result" -ne 0 ]]; then status=1; fi
    done
done
set +e
/usr/bin/time -v -o "$root/summary.resources" prlimit --as=68719476736:68719476736 \
    -- "$binary" summary "$root" "$root/report.json" > "$root/summary.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$root/summary.exit"
if [[ "$result" -ne 0 ]]; then status=1; fi
printf '%s\n' "$status" > "$root/process.exit"
exit "$status"
