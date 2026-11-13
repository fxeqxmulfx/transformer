#!/usr/bin/env bash
# Adaptively integrate completed frozen parents; do not run new discovery.
set -euo pipefail
ulimit -c 0
ulimit -v 67108864
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git rev-parse --path-format=absolute --git-common-dir)
root=${root%/.git}
cd "$root"
protocol="$source_dir/protocol.json"
fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ $# == 1 && $1 == preflight || $# == 2 && $1 == compare ]] || fail 'usage: run.sh preflight | compare FRESH_OUTPUT'
mode=$1
hash_check() {
  local actual
  [[ -s "$1" ]] || fail "Complete parent missing: $1"
  actual=$(sha256sum -- "$1")
  [[ ${actual%% *} == "$2" ]] || fail "Frozen parent changed: $1"
}
general="$root/$(jq -r .general "$protocol")"
hash_check "$general/summary.json" "$(jq -r .general_summary_sha256 "$protocol")"
[[ -f "$general/process.exit" && $(cat "$general/process.exit") == 0 ]] || fail 'General comparison is incomplete.'
sha256sum --check --quiet "$general/inputs.sha256" "$general/results.sha256"
while IFS=$'\t' read -r name cohort binary summary_hash binary_hash fingerprint; do
  hash_check "$root/$cohort/summary.json" "$summary_hash"
  hash_check "$root/$binary" "$binary_hash"
  [[ $("$root/$binary" fingerprint) == "$fingerprint" ]] || fail "Producer fingerprint changed: $name"
done < <(jq -r '.families[]|[.name,.cohort,.binary,.summary_sha256,.binary_sha256,.source_fingerprint]|@tsv' "$protocol")
[[ $mode == compare ]] || { printf '%s\n' 'Complete pinned parents are present; compare reruns their native evidence gates.'; exit 0; }
output=$(realpath -m -- "$2")
[[ ! -e "$output" ]] || fail 'Comparison destination already exists.'
mkdir -- "$output"
trap 'printf "%s\n" "$?" > "$output/process.exit"' EXIT
cp -- "$protocol" "$output/protocol.json"
git -C "$source_dir" rev-parse HEAD > "$output/comparison.commit"
sha256sum -- "$source_dir/run.sh" "$source_dir/combine.jq" "$protocol" \
  "$root/experiments/convex_stress/suite.json" "$general/summary.json" > "$output/inputs.sha256"
measure() {
  local label=$1; shift
  local status=0
  /usr/bin/time -v -o "$output/$label.resources" prlimit --as=68719476736:68719476736 -- \
    "$@" > "$output/$label.json" 2> "$output/$label.log" || status=$?
  printf '%s\n' "$status" > "$output/$label.exit"
  return "$status"
}
mapfile -t general_inputs < <(jq -r --arg root "$root" '.general_inputs[]|$root+"/"+.' "$protocol")
measure general-gates bash "$root/$(jq -r .general_harness "$protocol")/run.sh" compare \
  "${general_inputs[@]}" "$output/general"
cmp "$general/summary.json" "$output/general/summary.json"
while IFS=$'\t' read -r name cohort binary; do
  sha256sum -- "$root/$cohort/summary.json" "$root/$binary" >> "$output/inputs.sha256"
  measure "$name" "$root/$binary" summarize "$cohort"
  # Legacy parent summaries retain relative report paths; replay their exact
  # original invocation, then compare all bytes. This does not rewrite parents.
  cmp "$root/$cohort/summary.json" "$output/$name.json"
done < <(jq -r '.families[]|[.name,.cohort,.binary]|@tsv' "$protocol")
measure families jq -n --slurpfile density "$output/density.json" --slurpfile balanced "$output/balanced.json" \
  --slurpfile tagged "$output/tagged.json" --slurpfile fibers "$output/fibers.json" \
  '[{name:"density",summary:$density[0]},{name:"balanced",summary:$balanced[0]},{name:"tagged",summary:$tagged[0]},{name:"fibers",summary:$fibers[0]}]'
measure summary jq -n --slurpfile general "$output/general/summary.json" --slurpfile families "$output/families.json" \
  --slurpfile suite "$root/experiments/convex_stress/suite.json" --slurpfile protocol "$protocol" -f "$source_dir/combine.jq"
sha256sum --check --quiet "$output/inputs.sha256"
sha256sum -- "$output"/*.json > "$output/results.sha256"
printf '%s\n' 'All105 choices retained; completed constructors and every terminal outcome checked.'
