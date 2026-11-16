#!/usr/bin/env bash
# Freeze first. Complete every training endpoint before any original heldout score.
set -euo pipefail
test "$#" -ge 5
binary=$(realpath "$1")
workspace=$(realpath "$2")
root=$3
mode=$4
shift 4
test ! -e "$root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$mode" == original ]]; then
    test "$#" -eq 4
    prlimit --as=110000000000:110000000000 -- "$binary" prepare "$workspace" "$root" "$1" "$2" "$3" "$4"
elif [[ "$mode" == repeat ]]; then
    test "$#" -eq 1
    prlimit --as=110000000000:110000000000 -- "$binary" repeat "$workspace" "$1" "$root"
else
    exit 2
fi
mkdir "$root/logs"
mapfile -t cases < <(jq -r '.cases[].name' "$root/inventory.json")
test "${#cases[@]}" -eq 54
for phase in stage seal score replay summary; do
    phase_cases=("${cases[@]}")
    if [[ "$phase" == seal || "$phase" == summary ]]; then phase_cases=(complete); fi
    for case in "${phase_cases[@]}"; do
        args=("$phase" "$workspace" "$root")
        if [[ "$case" != complete ]]; then args+=("$case"); fi
        set +e
        /usr/bin/time -v -o "$root/logs/$case-$phase.resources" \
            prlimit --as=110000000000:110000000000 -- "$binary" "${args[@]}" \
            > "$root/logs/$case-$phase.log" 2>&1
        result=$?
        set -e
        printf '%s\n' "$result" > "$root/logs/$case-$phase.exit"
        if [[ "$result" -ne 0 ]]; then
            printf '%s\n' "$result" > "$root/process.exit"
            exit "$result"
        fi
    done
done
printf '0\n' > "$root/process.exit"
