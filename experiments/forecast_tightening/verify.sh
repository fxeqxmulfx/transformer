#!/usr/bin/env bash
# Verify the retained cohort, immutable inputs, frozen payloads and query records.
set -euo pipefail
study=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cohort=$(realpath -- "${1:?usage: verify.sh COHORT_DIRECTORY}")
scratch=$(mktemp -d /tmp/forecast-tightening-verify.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
jq -e -f "$study/verify.jq" "$cohort/report.json" > /dev/null
jq -r '.inputs.files|to_entries[]|"\(.value)  \(.key)"' "$cohort/report.json" > "$scratch/inputs.sha256"
sha256sum --check --quiet "$scratch/inputs.sha256"
jq -r '.cases[]|[.arm,.seed,.step,.audit.binary.sha256]|@tsv' "$cohort/report.json" > "$scratch/cases.tsv"
while IFS=$'\t' read -r arm seed step digest; do
  directory="$cohort/$arm-seed$seed-step$step"
  printf '%s  %s\n' "$digest" "$directory/forecast.bin" > "$scratch/forecast.sha256"
  sha256sum --check --quiet "$scratch/forecast.sha256"
  jq -e -n --slurpfile report "$cohort/report.json" --slurpfile row "$directory/result.json" \
    '$report[0].cases|any(.==$row[0])' > /dev/null
  jq -e --slurpfile row "$directory/result.json" \
    '.binary==$row[0].audit.binary and .state_sha256==$row[0].audit.state_sha256 and .matrix_bounds==$row[0].audit.matrix_bounds' \
    "$directory/frozen.json" > /dev/null
  jq -e -f "$study/queries_verify.jq" "$directory/queries.json" > /dev/null
  jq -e --slurpfile row "$directory/result.json" \
    '([.rows[]|select(.[1]==.[2])]|length)==($row[0].audit.checks.subsets.train.actually_correct+$row[0].audit.checks.subsets.validation.actually_correct)' \
    "$directory/queries.json" > /dev/null
  printf 'verified %s seed=%s step=%s\n' "$arm" "$seed" "$step"
done < "$scratch/cases.tsv"
printf '24 cases, 74 inputs, 24 frozen binary hashes and 225816 query records verified\n'
