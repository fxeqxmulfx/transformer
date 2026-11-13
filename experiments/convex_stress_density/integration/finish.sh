#!/usr/bin/env bash
# Reuse completed immutable native gates after a metadata-reader correction.
set -euo pipefail
ulimit -c 0
ulimit -v 67108864
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ $# == 2 ]] || { echo 'usage: finish.sh VERIFIED_GATE_DIRECTORY FRESH_OUTPUT' >&2; exit 2; }
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(git rev-parse --path-format=absolute --git-common-dir)
root=${root%/.git}
cd "$root"
prior=$(realpath -e -- "$1")
output=$(realpath -m -- "$2")
protocol="$source_dir/protocol.json"
[[ ! -e "$output" ]]
cmp "$protocol" "$prior/protocol.json"
for gate in general-gates density balanced tagged fibers families; do
  [[ $(cat "$prior/$gate.exit") == 0 ]]
done
[[ $(cat "$prior/general/process.exit") == 0 ]]
sha256sum --check --quiet "$prior/inputs.sha256" "$prior/general/inputs.sha256" "$prior/general/results.sha256"
recovery="$root/$(jq -r '.general_inputs[4]' "$protocol")"
sha256sum --check --quiet "$recovery/comparison.sha256" "$recovery/completed-endpoints.sha256"
general_hash=$(sha256sum "$prior/general/summary.json")
[[ ${general_hash%% *} == $(jq -r .general_summary_sha256 "$protocol") ]]
mkdir "$output"
trap 'printf "%s\n" "$?" > "$output/process.exit"' EXIT
cp "$protocol" "$output/protocol.json"
printf '%s\n' "$prior" > "$output/verified-gates.path"
git -C "$source_dir" rev-parse HEAD > "$output/comparison.commit"
sha256sum "$source_dir/finish.sh" "$source_dir/combine.jq" "$protocol" "$prior/inputs.sha256" \
  "$prior/general/summary.json" "$root/experiments/convex_stress/suite.json" > "$output/inputs.sha256"
while IFS=$'\t' read -r name cohort binary summary_hash binary_hash fingerprint; do
  actual=$(sha256sum "$root/$binary")
  [[ ${actual%% *} == "$binary_hash" && $("$root/$binary" fingerprint) == "$fingerprint" ]]
  /usr/bin/time -v -o "$output/$name.resources" prlimit --as=68719476736:68719476736 -- \
    "$root/$binary" summarize "$cohort" > "$output/$name.json" 2> "$output/$name.log"
  actual=$(sha256sum "$output/$name.json")
  [[ ${actual%% *} == "$summary_hash" ]]
  cmp "$prior/$name.json" "$output/$name.json"
  sha256sum "$root/$binary" "$root/$cohort/summary.json" >> "$output/inputs.sha256"
done < <(jq -r '.families[]|[.name,.cohort,.binary,.summary_sha256,.binary_sha256,.source_fingerprint]|@tsv' "$protocol")
jq -n --slurpfile density "$output/density.json" --slurpfile balanced "$output/balanced.json" \
  --slurpfile tagged "$output/tagged.json" --slurpfile fibers "$output/fibers.json" \
  '[{name:"density",summary:$density[0]},{name:"balanced",summary:$balanced[0]},{name:"tagged",summary:$tagged[0]},{name:"fibers",summary:$fibers[0]}]' > "$output/families.json"
/usr/bin/time -v -o "$output/summary.resources" prlimit --as=68719476736:68719476736 -- jq -n \
  --slurpfile general "$prior/general/summary.json" --slurpfile families "$output/families.json" \
  --slurpfile suite "$root/experiments/convex_stress/suite.json" --slurpfile protocol "$protocol" \
  -f "$source_dir/combine.jq" > "$output/summary.json" 2> "$output/summary.log"
sha256sum --check --quiet "$output/inputs.sha256"
sha256sum "$output"/*.json > "$output/results.sha256"
