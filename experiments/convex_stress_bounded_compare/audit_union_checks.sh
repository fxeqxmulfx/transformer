#!/usr/bin/env bash
# Exercise omissions and double counting against copies of retained evidence.
set -euo pipefail
test "$#" -eq 7
audit_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
audit_scratch=$(mktemp -d)
trap 'rm -rf -- "$audit_scratch"' EXIT
audit_inputs=("$@")
bash "$audit_source_dir/audit_union.sh" "${audit_inputs[@]}" > "$audit_scratch/positive.json"
jq -e '.counts.current_two_mode == 98 and .counts.three_mode == 96' \
  "$audit_scratch/positive.json" > /dev/null
printf '%s\n' 'PASS: complete named union'
reject_probe_change() {
  local audit_name=$1 audit_filter=$2 audit_reason=$3
  jq "$audit_filter" "${audit_inputs[6]}" > "$audit_scratch/$audit_name.json"
  if bash "$audit_source_dir/audit_union.sh" "${audit_inputs[@]:0:6}" \
      "$audit_scratch/$audit_name.json" > "$audit_scratch/$audit_name.out" \
      2> "$audit_scratch/$audit_name.err"; then
    printf 'FAIL: accepted %s\n' "$audit_name" >&2
    exit 1
  fi
  rg -q -F "$audit_reason" "$audit_scratch/$audit_name.err"
  printf 'PASS: rejected %s\n' "$audit_name"
}
reject_probe_change missing_mode \
  '(.cases[] | select(.name=="gpt-probe-conditional-amplitudes") | .modes) |= map(select(.mode!="Mini"))' \
  'probe modes, complete splits or pass criteria differ'
reject_probe_change missing_novel_split \
  '(.cases[] | select(.name=="gpt-probe-key-value-binding") | .modes[].splits) |= map(select(.split!="validation/novel"))' \
  'probe modes, complete splits or pass criteria differ'
reject_probe_change overlapping_new_case \
  '(.cases[] | select(.name=="gpt-probe-conditional-amplitudes") | .name) = "arithmetic-max"' \
  'named union does not match the expected verified coverage'
reject_probe_change false_raw99 \
  '(.cases[] | select(.name=="gpt-probe-order-discrimination") | .modes[0].splits[0].score.raw99) = false' \
  'probe modes, complete splits or pass criteria differ'
