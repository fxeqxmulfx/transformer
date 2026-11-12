#!/usr/bin/env bash
# Reproduce the independent checks on a complete frozen cohort; never retrain.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
source_dir="$root/experiments/convex_stress_schatten"
cohort=$(realpath "${1:?usage: audit_cohort.sh COHORT PRIOR_COHORT CONTROL_OUTPUT}")
prior=$(realpath "${2:?missing prior coupled-moment cohort}")
controls=${3:?missing fresh output directory for corruption controls}
binary=${SCHATTEN_BIN:-"$source_dir/target/release/convex-stress-schatten"}
auditor=${SCHATTEN_AUDITOR:-"$source_dir/target/release/audit_saved"}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
fingerprint=$(cat "$cohort/source_fingerprint.txt")
test "$fingerprint" = "$("$binary" fingerprint)"
test ! -e "$controls"
mkdir -p "$controls"
controls=$(realpath "$controls")
jq -s --slurpfile p "$source_dir/protocol.json" -f "$source_dir/summarize.jq" \
  "$cohort"/*/result.json > "$controls/summary-replay.json"
cmp "$cohort/summary.json" "$controls/summary-replay.json"
for result in "$cohort"/*/result.json; do
  "$auditor" "${result%/result.json}" "$fingerprint"
done > "$cohort/independent-audits.jsonl"
jq -s '{audited:length,all_pass:all(.independent_audit)}' \
  "$cohort/independent-audits.jsonl" > "$cohort/independent-audit.json"
for task in arithmetic-div arithmetic-mul; do
  for arm in full_batch minibatch forward_only; do
    "$auditor" compare "$prior/$arm--$task--sqrt" \
      "$cohort/$arm--$task--full_rank" "$fingerprint"
  done
done > "$cohort/original-coordinate-comparisons.jsonl"

expect_rejected() {
  local label=$1
  shift
  if "$@" > "$controls/$label.log" 2>&1; then
    echo "ERROR: accepted corrupt $label" >&2
    exit 1
  fi
  printf 'rejected %s\n' "$label"
}

good="$cohort/full_batch--arithmetic-add--rank_one"
cp -a --reflink=auto "$good" "$controls/ce"
jq '.metrics.train.loss += 0.01' "$good/result.json" > "$controls/ce/result.json"
expect_rejected ce-replay "$binary" replay "$controls/ce"
expect_rejected ce-independent "$auditor" "$controls/ce" "$fingerprint"

cp -a --reflink=auto "$good" "$controls/nuclear"
jq '.moment_norms[0].nuclear += 1' "$good/result.json" > "$controls/nuclear/result.json"
expect_rejected nuclear-replay "$binary" replay "$controls/nuclear"
expect_rejected nuclear-independent "$auditor" "$controls/nuclear" "$fingerprint"

cp -a --reflink=auto "$good" "$controls/state"
jq 'map(0)' "$good/diagnostic_state.json" > "$controls/state/diagnostic_state.json"
state_hash=$(sha256sum "$controls/state/diagnostic_state.json")
state_hash=${state_hash%% *}
jq --arg hash "$state_hash" '.diagnostic_state_sha256=$hash' "$good/result.json" \
  > "$controls/state/result.json"
expect_rejected rehashed-state "$binary" replay "$controls/state"

mkdir "$controls/hull"
for result in "$cohort"/*/result.json; do
  cell=${result%/result.json}
  ln -s "$cell" "$controls/hull/${cell##*/}"
done
jq '.choices[0].selected_budget="full_rank"' "$cohort/hull.json" > "$controls/hull/hull.json"
expect_rejected hull-selection "$binary" replay-hull "$controls/hull"

jq -s '.[1:]' "$cohort"/*/result.json > "$controls/missing-cell.json"
expect_rejected missing-inventory jq --slurpfile p "$source_dir/protocol.json" \
  -f "$source_dir/summarize.jq" "$controls/missing-cell.json"
printf 'All independent audits, original-coordinate comparisons and corruption controls passed.\n'
