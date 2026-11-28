#!/usr/bin/env bash
# Replay the Lean kernel checks on all generated FP32 score blocks, then
# check the theorem quantifying over every numerator, denominator and rival.
set -euo pipefail
if (( $#<2 || $#>3 )) || [[ $2 != terminal && $2 != first-strict ]] || [[ ${3:---hidden} != --hidden ]]; then
    echo 'usage: bash check_recorded_division.sh CELL (terminal|first-strict) [--hidden]' >&2; exit 2
fi
cell=$(realpath -e -- "$1")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
kind=$2
scope=all
theorem=all_recorded_quotients
if (( $#==3 )); then
    kind+=-hidden
    scope=hidden
    theorem=hidden_recorded_quotients
    bin=$(realpath -e -- "${ARGAND_BIN:-$repo/experiments/argand/target/release/argand}")
fi
root="$cell/recorded-$kind-lean"
report="$cell/recorded-$kind-division.json"
source="$cell/recorded-$kind-division.lean"
exec 9<"$root/export.lock"
flock -x 9
weights=weights.f32
checkpoint=result.json
if [[ $2 == first-strict ]]; then weights=first-strict.f32; checkpoint=first-strict.json; fi
registry=$(sha256sum "$report" | cut -d' ' -f1)
jq -e --slurpfile checkpoint "$cell/$checkpoint" --arg name "recorded-$kind-division.lean" --arg root "recorded-$kind-lean" --arg scope "$scope" '
  .lean_file==$name and .modules==$root and .recorded_logits==2225568 and
  (.scope//"all")==$scope and (if $scope=="hidden" then
    .certified_inputs==4656 and (.training_mask_lean_sha256|test("^[0-9a-f]{64}$")) else (.certified_inputs//9312)==9312 end) and
  .step==$checkpoint[0].step and .weights_sha256==$checkpoint[0].weights_sha256 and
  [.blocks[].file]==[range(0;97)|"B"+(("0"+tostring)[-2:])+".lean"] and
  all(.blocks[]; .sha256|test("^[0-9a-f]{64}$"))' "$report" >/dev/null
(
    cd -- "$cell"
    jq -r --arg weights "$weights" '"\(.weights_sha256)  \($weights)"' "$report" | sha256sum --check --status
    jq -r '"\(.lean_sha256)  \(.lean_file)",(.modules as $p | .blocks[] | "\(.sha256)  \($p)/RecordedDivisionBlocks/\(.file)")' "$report" |
        sha256sum --check --status
    if [[ $scope == hidden ]]; then
        jq -r '"\(.training_mask_lean_sha256)  \(.modules)/RecordedDivisionBlocks/TrainingMask.lean"' "$report" | sha256sum --check --status
    fi
)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
mkdir -- "$work/RecordedDivisionBlocks"
cd -- "$repo"
export LAKE_ARTIFACT_CACHE=false
if [[ $scope == hidden ]]; then
    "$bin" division-mask-lean > "$work/canonical-mask.lean"
    rg '^def observed : Array Nat' "$root/RecordedDivisionBlocks/TrainingMask.lean" > "$work/recorded-rows"
    rg '^def observed : Array Nat' "$work/canonical-mask.lean" | cmp - "$work/recorded-rows"
    lake build Transformer.Grokking.FiniteRecovery
else
    lake build Transformer.Grokking.FiniteRecovery.Binary32Scores
fi
paths=$(lake env printenv LEAN_PATH)
export LEAN_PATH="$work:$paths"
if [[ $scope == hidden ]]; then
    lean --root="$root" -j 1 -o "$work/RecordedDivisionBlocks/TrainingMask.olean" \
        "$root/RecordedDivisionBlocks/TrainingMask.lean" > "$work/training-mask.log" 2>&1 || { cat "$work/training-mask.log" >&2; exit 1; }
    if rg -q 'sorryAx|warning:|error:' "$work/training-mask.log"; then cat "$work/training-mask.log" >&2; exit 1; fi
fi
jobs=${ARGAND_LEAN_JOBS:-2}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo 'invalid Lean worker count' >&2; exit 2; }
printf 'B%02d\n' {0..96} | xargs -r -P "$jobs" -n 1 bash -euc '
    root=$1; work=$2; block=$3
    lean --root="$root" -j 1 -o "$work/RecordedDivisionBlocks/$block.olean" \
      "$root/RecordedDivisionBlocks/$block.lean" > "$work/$block.log" 2>&1 || { cat "$work/$block.log" >&2; exit 1; }
    if [[ -s "$work/$block.log" ]]; then cat "$work/$block.log" >&2; exit 1; fi
' bash "$root" "$work"
lean -j 1 "$source" > "$work/theorem.log" 2>&1 || { cat "$work/theorem.log" >&2; exit 1; }
if grep -Eq 'sorryAx|warning:|error:' "$work/theorem.log"; then cat "$work/theorem.log" >&2; exit 1; fi
grep -Fq "'RecordedDivision.$theorem' depends on axioms:" "$work/theorem.log"
jq -n --slurpfile r "$report" --arg manifest "$registry" --arg scope "$scope" --arg version "$(lean --version)" --rawfile axioms "$work/theorem.log" '
  {step:$r[0].step,weights_sha256:$r[0].weights_sha256,logits_sha256:$r[0].logits_little_endian_sha256,
   manifest_sha256:$manifest,scope:$scope,division_inputs:($r[0].certified_inputs//9312),
   recorded_inputs:9312,output_tokens:239,checked_blocks:97,lean:$version,theorem:$axioms}' \
  > "$work/kernel.json"
cp -- "$work/kernel.json" "$cell/recorded-$kind-kernel.json.pending"
mv -- "$cell/recorded-$kind-kernel.json.pending" "$cell/recorded-$kind-kernel.json"
cat "$work/kernel.json"
