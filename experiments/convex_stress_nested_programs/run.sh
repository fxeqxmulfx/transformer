#!/usr/bin/env bash
# Retain every process outcome; all six states are sealed before any scoring.
set -euo pipefail
test "$#" -eq 3
binary=$1
baseline=$2
run_root=$3
test ! -e "$run_root"
test -x "$binary"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
producer=$(git rev-parse HEAD)
prlimit --as=68719476736:68719476736 -- "$binary" prepare-run "$run_root" "$producer" "$baseline"
trap 'printf "%s\n" "$?" > "$run_root/process.exit"' EXIT
failed=0
record() {
    local phase=$1 case=$2 status=0
    /usr/bin/time -v -o "$run_root/$phase-$case.resources" \
        prlimit --as=68719476736:68719476736 -- "$binary" "$phase" "$run_root" "$case" \
        > "$run_root/$phase-$case.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$run_root/$phase-$case.exit"
    printf '%s %s exit=%s\n' "$phase" "$case" "$status"
    if test "$status" -ne 0; then failed=$((failed + 1)); fi
}
for phase in stage score replay; do
    for case in suite-mqar suite-lookup suite-dyck suite-blocks suite-dyck-2 suite-dyck-3; do record "$phase" "$case"; done
done
test "$failed" -eq 0
prlimit --as=68719476736:68719476736 -- "$binary" summary "$run_root" "$run_root/report.json"
