#!/usr/bin/env bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$source_dir" rev-parse --show-toplevel)
out=$(mktemp -d /tmp/convex-full35-integration.XXXXXX)
jq -n --slurpfile protocol "$source_dir/protocol.json" --slurpfile suite "$repo/experiments/convex_stress/suite.json" \
  -f "$source_dir/fixture.jq" > "$out/fixture.json"
run_fixture() {
  local input=$1
  for part in general families suite protocol; do jq ".$part" "$input" > "$out/$part.json"; done
  jq -n --slurpfile general "$out/general.json" --slurpfile families "$out/families.json" \
    --slurpfile suite "$out/suite.json" --slurpfile protocol "$out/protocol.json" -f "$source_dir/combine.jq"
}
run_fixture "$out/fixture.json" > "$out/result.json"
jq -e '.choices_compared==105 and (.choices|length)==105 and (.by_mode|all(.cases==35))
  and (.choices|map(select(.updated_scope))|length)==27
  and (.choices|map(select(.updated_scope))|all(.selected.family=="fibers" and (.candidates|length)==5))
  and (.choices|map(select(.updated_scope|not))|all(.selected.family=="earlier"))
  and (.constructor_accounting|all(.planned_constructions==9 and .mode_references==27 and .completed_seconds==9 and .charged_once))
  and .full35_result and (.full35_pass|not) and (.fresh_transformer_comparison|not)' "$out/result.json" > /dev/null
# Make all fresh models tie the earlier choice: the earlier choice must stay.
jq '(.families[].summary.records[].result.evaluation.candidates[]|.train_ce,.audit.train_ce,.metrics.train.loss)=3.9
    |(.families[].summary.mode_references[].selected|.train_ce,.audit.train_ce,.metrics.train.loss)=3.9' "$out/fixture.json" > "$out/tie.json"
run_fixture "$out/tie.json" > "$out/tie-result.json"
jq -e 'all(.choices[];.selected.family=="earlier")' "$out/tie-result.json" > /dev/null
# Every late model has better held-out performance, but worse observed CE.
jq '(.families[].summary.records[].result.evaluation.candidates[]|.train_ce,.audit.train_ce,.metrics.train.loss)=4.0
    |(.families[].summary.mode_references[].selected|.train_ce,.audit.train_ce,.metrics.train.loss)=4.0
    |(.families[].summary.records[].result.evaluation.candidates[]|.metrics.validation.sequence_accuracy)=1
    |(.families[].summary.records[].result.evaluation.candidates[]|.primary_pass,.all_splits_pass)=true
    |(.families[].summary.mode_references[].selected|.metrics.validation.sequence_accuracy)=1
    |(.families[].summary.mode_references[].selected|.primary_pass,.all_splits_pass)=true' "$out/fixture.json" > "$out/heldout.json"
run_fixture "$out/heldout.json" > "$out/heldout-result.json"
jq -e 'all(.choices[];.selected.family=="earlier")' "$out/heldout-result.json" > /dev/null
jq '.families[].summary |= (.process_failures=9|.records|=map({task,status:"process_failure",run_exit:1})
    |.mode_references|=map(.status="process_failure"|.selected=null))' "$out/fixture.json" > "$out/failed.json"
run_fixture "$out/failed.json" > "$out/failed-result.json"
jq -e '(.choices|length)==105 and all(.constructor_accounting[];.process_failures==9)
    and all(.constructor_accounting[];.completed_seconds==0)
    and all(.choices[];.selected.family=="earlier")' "$out/failed-result.json" > /dev/null
jq '(.general.choices[]|.status)="no_model"|(.general.choices[]|.selected)=null
    |(.general.parents.full_suite_hull.choices[]|.status)="no_model"' "$out/failed.json" > "$out/no-model.json"
run_fixture "$out/no-model.json" > "$out/no-model-result.json"
jq -e '(.choices|length)==105 and all(.choices[];.status=="no_model" and .selected==null)
    and all(.by_mode[];.no_model==35 and .primary_passes==0 and .all_split_passes==0)' "$out/no-model-result.json" > /dev/null
# Novel subsets can be derived from a sequence parent's raw data fingerprint.
# Preserve its exact replayed metric inventory, including these derived splits.
jq '.suite.tasks[0].name as $task|.suite.tasks[0].selection="validation/novel"
    |(.general.parents.full_suite_hull.choices[]|select(.task==$task)|.metrics["validation/novel"])={loss:4.2,sequence_accuracy:0.5}' "$out/fixture.json" > "$out/novel.json"
run_fixture "$out/novel.json" > "$out/novel-result.json"
jq -e '.choices_compared==105 and (.choices[0].selected.metrics["validation/novel"].sequence_accuracy==0.5)' "$out/novel-result.json" > /dev/null
# Original density audits call this field original_vocabulary; newer phase
# audits call it vocabulary. Both certify the same unchanged CE denominator.
jq '(.families[0:2][] .summary.records[].result.evaluation.candidates[].audit)|=(.original_vocabulary=.vocabulary|del(.vocabulary))
    |(.families[0:2][] .summary.mode_references[].selected.audit)|=(.original_vocabulary=.vocabulary|del(.vocabulary))' "$out/fixture.json" > "$out/legacy.json"
run_fixture "$out/legacy.json" > "$out/legacy-result.json"
jq -e '.choices_compared==105' "$out/legacy-result.json" > /dev/null
mutations=(
  'del(.general.parents.full_suite_hull.choices[0])'
  '.general.parents.full_suite_hull.choices += [.general.parents.full_suite_hull.choices[0]]'
  'del(.general.choices[0])'
  'del(.families[0].summary.records[0])'
  '.families[0].summary.records += [.families[0].summary.records[0]]'
  '.families|=reverse'
  '.families[0].summary.source_fingerprint="changed"'
  '.families[0].summary.producer_commit="changed"'
  '.families[0].summary.records[0].result.metadata.input.data_fingerprints.train="changed"'
  '.families[0].summary.records[0].result.metadata.input.vocab=98'
  '.families[0].summary.records[0].result.evaluation.candidates[0].audit.train_ce=0'
  '.families[0].summary.records[0].result.evaluation.candidates[0].primary_pass=true'
  '.families[0].summary.mode_references[0].selected=null'
  '.families[0].summary.mode_references[0].gradient_training_run=true'
  '.families[0].summary.records[0].result.construction_seconds=-1'
  '.families[0].summary.records[0].result.metadata.training_gradient_evaluations=1'
  '.families[0].summary.records[0].result.evaluation.selected="other"'
  '.families[0].summary.process_failures=1'
  'del(.families[0].summary.records[0].result.evaluation.candidates[0].metrics.validation)'
  '.general.choices[0].selected=null'
  '.families[0].summary.records[0].result.evaluation.status="proposal_ineligible"'
  '.families[0].summary.records[0].result.evaluation.candidates[0].audit.original_vocabulary=98'
)
for index in "${!mutations[@]}"; do
  jq "${mutations[$index]}" "$out/fixture.json" > "$out/mutation.json"
  if run_fixture "$out/mutation.json" > "$out/rejected-$index.json" 2> "$out/rejected-$index.log"; then
    printf 'Mutation %s was accepted\n' "$index" >&2; exit 1
  fi
done
mkdir "$out/gate"
cp "$source_dir/run.sh" "$out/gate/run.sh"
jq '.general="missing-complete-parent"' "$source_dir/protocol.json" > "$out/gate/protocol.json"
if bash "$out/gate/run.sh" compare "$out/should-not-exist" > "$out/gate.log" 2>&1; then exit 1; fi
test ! -e "$out/should-not-exist"
cp "$source_dir/finish.sh" "$out/gate/finish.sh"
mkdir "$out/incomplete-gates"
cp "$out/gate/protocol.json" "$out/incomplete-gates/protocol.json"
printf '1\n' > "$out/incomplete-gates/general-gates.exit"
if bash "$out/gate/finish.sh" "$out/incomplete-gates" "$out/unfinished-output" > "$out/finish-gate.log" 2>&1; then exit 1; fi
test ! -e "$out/unfinished-output"
printf '%s\n' 'Full105 retention, derived novel splits, both audit vocabularies, all35/modes, earlier-first CE, held-out invariance, constructor accounting, all-failure/no-model retention,22 mutations and missing-parent gate pass.'
