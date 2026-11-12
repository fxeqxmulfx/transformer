#!/usr/bin/env bash
# Compare only complete audited predecessors; never modify their evidence.
set -euo pipefail
ulimit -c 0
ulimit -v 67108864
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(git rev-parse --path-format=absolute --git-common-dir)
project_root=${project_root%/.git}
protocol="$source_dir/protocol.json"
fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ $# == 6 && $1 == preflight || $# == 7 && $1 == compare ]] ||
  fail 'usage: run.sh preflight GENERAL CERTIFIED PAIRED DECODED RECOVERY | compare GENERAL CERTIFIED PAIRED DECODED RECOVERY FRESH_OUTPUT'
mode=$1
general=$(realpath -m -- "$2")
certified=$(realpath -m -- "$3")
paired=$(realpath -m -- "$4")
decoded=$(realpath -m -- "$5")
recovery=$(realpath -m -- "$6")
for path in "$general/summary.json" "$certified/summary.json" "$paired/summary.json" \
    "$decoded/summary.json" "$recovery/view/selected.json" "$recovery/view/prepared.json" \
    "$recovery/comparison.sha256" "$recovery/completed-endpoints.sha256"; do
  [[ -s "$path" ]] || fail "Complete parent required: $path"
done
for path in "$recovery/selection.exit" "$recovery/selection-replay.exit"; do
  [[ -f "$path" && $(cat -- "$path") == 0 ]] || fail "Successful full-suite selection required: $path"
done
while IFS=$'\t' read -r path expected; do
  actual=$(sha256sum -- "$project_root/$path")
  [[ ${actual%% *} == "$expected" ]] || fail "Frozen input changed: $path"
done < <(jq -r '.frozen_inputs|to_entries[]|[.key,.value]|@tsv' "$protocol")
general_binary="$project_root/$(jq -r .general_binary "$protocol")"
certified_binary="$project_root/$(jq -r .certified_binary "$protocol")"
[[ $("$general_binary" fingerprint) == $(jq -r .general_fingerprint "$protocol") ]] || fail 'General producer changed.'
[[ $("$certified_binary" fingerprint) == $(jq -r .certified_fingerprint "$protocol") ]] || fail 'Certified producer changed.'
[[ $mode == compare ]] || { printf '%s\n' 'Required complete files and frozen producers are present; comparison still reruns all evidence gates.'; exit 0; }
output=$(realpath -m -- "$7")
[[ ! -e "$output" ]] || fail 'Comparison destination already exists.'
mkdir -- "$output"
trap 'status=$?; printf "%s\n" "$status" > "$output/process.exit"' EXIT
cp -- "$protocol" "$output/protocol.json"
git -C "$source_dir" rev-parse HEAD > "$output/comparison.commit"
sha256sum -- "$source_dir/run.sh" "$source_dir/combine.jq" "$protocol" \
  "$general/summary.json" "$certified/summary.json" "$paired/summary.json" "$decoded/summary.json" \
  "$recovery/view/selected.json" "$recovery/view/prepared.json" > "$output/inputs.sha256"
measure() {
  local label=$1; shift
  local status=0
  /usr/bin/time -v -o "$output/$label.resources" prlimit --as=68719476736:68719476736 -- \
    "$@" > "$output/$label.json" 2> "$output/$label.log" || status=$?
  printf '%s\n' "$status" > "$output/$label.exit"
  return "$status"
}
measure general "$general_binary" summarize "$general"
cmp -- "$general/summary.json" "$output/general.json"
measure certified "$certified_binary" integrate "$certified" "$paired" "$decoded"
measure recovery bash "$project_root/experiments/convex_stress_hull/recovery/run.sh" verify "$recovery"
jq -e '.selection_replayed == true and .choices == 105' "$output/recovery.json" > /dev/null
measure summary jq -n --slurpfile general "$output/general.json" --slurpfile certified "$output/certified.json" \
  --slurpfile hull "$recovery/view/selected.json" --slurpfile prepared "$recovery/view/prepared.json" \
  --slurpfile suite "$project_root/experiments/convex_stress/suite.json" --slurpfile protocol "$protocol" \
  -f "$source_dir/combine.jq"
sha256sum --check --quiet "$output/inputs.sha256"
sha256sum -- "$output"/*.json > "$output/results.sha256"
printf '%s\n' 'All 27 comparisons checked; complete parents and every refusal retained.'
