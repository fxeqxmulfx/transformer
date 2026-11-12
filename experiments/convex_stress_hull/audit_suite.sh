#!/usr/bin/env bash
# Audit completed cells serially. Save each exit and measured peak resident size.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/experiments/convex_stress_hull"
cohort=$(realpath "${1:?usage: audit_suite.sh COHORT FRESH_AUDIT_DIRECTORY}")
out=${2:?usage: audit_suite.sh COHORT FRESH_AUDIT_DIRECTORY}
counts=${HULL_COUNT_AUDITOR:-"$source_dir/target/release/audit_counts"}
moments=${HULL_MOMENT_AUDITOR:-"$source_dir/target/release/audit_moments"}
mkdir "$out"
out=$(realpath "$out")
fingerprint=$(cat "$cohort/source_fingerprint.txt")
sha256sum "$counts" "$moments" > "$out/binary-hashes.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
failed=0
for result in "$cohort"/*/result.json; do
  if ! jq -e '.status == "complete"' "$result" > /dev/null; then continue; fi
  directory=$(dirname "$result")
  cell=$(basename "$directory")
  if [ "$(jq -r .candidate "$result")" = counts ]; then auditor=$counts; else auditor=$moments; fi
  if /usr/bin/time -v -o "$out/$cell.resources" bash "$source_dir/limited.sh" 8 \
      "$auditor" "$directory" "$fingerprint" > "$out/$cell.json" 2> "$out/$cell.log"; then
    printf '0\n' > "$out/$cell.exit"
  else
    status=$?
    printf '%s\n' "$status" > "$out/$cell.exit"
    failed=$((failed + 1))
  fi
done
test "$failed" = 0
