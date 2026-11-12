#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
suite="$source_dir/../../convex_stress/suite.json"
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
jq -n --slurpfile protocol "$source_dir/protocol.json" --slurpfile suite "$suite" \
  -f "$source_dir/fixture.jq" > "$scratch/original.json"
combine() {
  for key in general certified hull prepared; do jq ".$key" "$scratch/input.json" > "$scratch/$key.json"; done
  jq -n --slurpfile protocol "$source_dir/protocol.json" --slurpfile suite "$suite" \
    --slurpfile general "$scratch/general.json" --slurpfile certified "$scratch/certified.json" \
    --slurpfile hull "$scratch/hull.json" --slurpfile prepared "$scratch/prepared.json" \
    -f "$source_dir/combine.jq" > "$scratch/result.json" 2> "$scratch/error.log" || {
      cat -- "$scratch/error.log" >&2; return 1;
    }
}
mutation() { jq "$1" "$scratch/original.json" > "$scratch/input.json"; }
reject() {
  mutation "$1"
  if combine 2>/dev/null; then printf 'Accepted invalid fixture: %s\n' "$1" >&2; exit 1; fi
}
mutation '.'
combine
jq -e '.choices_compared==27 and .general_selected_choices==27 and .primary_pass_choices==0
  and (.parents.full_suite_hull.choices|length)==105 and (.parents.general.records|length)==27
  and .full35_result==false and .fresh_transformer_comparison==false
  and all(.choices[]; .selected.decoder_index==0 and (.candidates|length)==3)' "$scratch/result.json" > /dev/null
cp -- "$scratch/result.json" "$scratch/baseline.json"

# Reversing held-out success does not alter the identities selected by CE.
mutation '.general.candidates |= map(.primary_pass=true | .all_splits_pass=true | .metrics.validation.sequence_accuracy=1)'
combine
cmp <(jq '[.choices[]|[.task,.training_mode,.selected.decoder_index,.selected.train_ce]]' "$scratch/baseline.json") \
    <(jq '[.choices[]|[.task,.training_mode,.selected.decoder_index,.selected.train_ce]]' "$scratch/result.json")

# Declaration order breaks an ideal-CE tie even when a later CE is lower.
mutation '.general.candidates |= map(.train_ce=(if .decoder_index==0 then 3.592270159 else 3.5922701555127 end))'
combine
jq -e 'all(.choices[]; .selected.decoder_index==0)' "$scratch/result.json" > /dev/null
mutation '.general.candidates |= map(.train_ce=(if .decoder_index==1 then 3.6 else 3.8 end))'
combine
jq -e 'all(.choices[]; .selected.decoder_index==1)' "$scratch/result.json" > /dev/null
mutation '.general.candidates |= map(.train_ce=4.1)'
combine
jq -e '.general_selected_choices==0 and all(.choices[]; .selected.train_ce==3.9)' "$scratch/result.json" > /dev/null

# Every explicit failed process stays in the comparison and uses its parent.
mutation '.general.candidates=[] | .general.choices |= map(.status="process_failure")
  | .general.records |= map(.status="process_failure" | .planned_decoder_slots=4 | del(.candidates))'
combine
jq -e '.general_selected_choices==0 and (.choices|length)==27
  and all(.choices[]; .general_record.status=="process_failure" and .selected.train_ce==3.9)' "$scratch/result.json" > /dev/null
jq '.hull.choices |= map(if .task=="arithmetic-abs-diff-blind" then .status="no_model" else . end)' \
  "$scratch/input.json" > "$scratch/next.json"
mv -- "$scratch/next.json" "$scratch/input.json"
combine
jq -e '([.choices[]|select(.status=="no_model")]|length)==3
  and all(.choices[]|select(.status=="no_model"); .selected==null)' "$scratch/result.json" > /dev/null

reject '.general.records |= .[1:]'
reject '.general.choices += [.general.choices[0]]'
reject '.general.candidates += [.general.candidates[0]]'
reject '.general.candidates[0].task="unplanned"'
reject '.general.records[0].status="running"'
reject '.general.records[0].candidates |= .[1:]'
reject '.general.records[0].candidates[3].status="running"'
reject '.general.candidates[0].train_ce=null'
reject '.general.candidates[0].train_ce=0'
reject '.general.candidates[0].vocab=98'
reject '.certified.choices |= .[1:]'
reject '.certified.choices += [.certified.choices[0]]'
reject '.hull.choices |= .[1:]'
reject '.hull.choices += [.hull.choices[0]]'
reject '.prepared.tasks |= map(if .task=="arithmetic-abs-diff-blind" then .scale=10 else . end)'
reject '.certified.source_fingerprint="changed"'
reject '.hull.source_fingerprint="changed"'

# Missing parents prevent output creation or any original native replay.
if bash "$source_dir/run.sh" compare "$scratch/missing" "$scratch/missing" "$scratch/missing" \
    "$scratch/missing" "$scratch/missing" "$scratch/output" > "$scratch/gate.log" 2>&1; then
  echo 'Missing-parent preflight succeeded.' >&2; exit 1
fi
test ! -e "$scratch/output"
rg -q 'Complete parent required:' "$scratch/gate.log"
printf '%s\n' '27-choice accounting, CE-only ties/selection, 105-choice retention, failures, 17 mutations and missing-parent gate pass.'
