#!/usr/bin/env bash
# Generic component runner; original benchmark integration is not available yet.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: run_catalogue.sh CONFIG FRESH_ROOT PRODUCER_COMMIT' >&2; exit 2; }
binary=${QUOTIENT_CATALOGUE_BINARY:-experiments/convex_stress_quotient/cohort/target/release/convex-stress-quotient-cohort}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
test ! -e "$2"
/usr/bin/time -v -o "$2-run.resources" prlimit --as=68719476736:68719476736 -- "$binary" run "$1" "$2" "$3"
/usr/bin/time -v -o "$2-replay.resources" prlimit --as=68719476736:68719476736 -- "$binary" replay "$2" > "$2/replay.json"
