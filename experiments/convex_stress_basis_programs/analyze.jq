# Read-only analysis of the c8cc576d report. Counts are split occurrences;
# overlapping novel subsets are not claimed as additional independent samples.
def total($field): map(.[$field]) | add // 0;
def rollup:
  {split_occurrences:length,
   targets:total("targets"), correct:total("correct"),
   examples:total("examples"), exact:total("exact"),
   minimum_raw:(map(.token_accuracy)|min),
   minimum_exact:(map(.sequence_accuracy)|min),
   weighted_teacher_ce:((map(.teacher_forced.loss*.teacher_forced.targets)|add)
                       /(map(.teacher_forced.targets)|add))};
def splits($kind):
  .evaluation.splits | to_entries[] |
  select(if $kind=="heldout" then (.key|startswith("train")|not)
         else .key=="test" or (.key|startswith("test/")) end) | .value.metrics;
. as $report |
[.cases[] as $case | $case.endpoints[] |
  . + {case:$case.case}] as $endpoints |
{
  producer_commit, binary_sha256,
  new_cases, data_seed,
  independent_repeat_required,
  original_training_targets:([.cases[].data.records.train.target_count]|add),
  original_case_splits:([.cases[].endpoints[0].evaluation.splits|length]|add),
  cases:[.cases[]|{case,error,
    original_selection,
    all_six_heldout99:([.endpoints[].all_heldout99]|all),
    minimum_raw:([.endpoints[]|splits("heldout")|.token_accuracy]|min),
    minimum_exact:([.endpoints[]|splits("heldout")|.sequence_accuracy]|min),
    old_all_anchor_checks:.all_anchor_diagnostics}],
  heldout:($endpoints|group_by([.prior,.mode])|map(
    {prior:.[0].prior,mode:.[0].mode}+([.[]|splits("heldout")]|rollup))),
  tests:($endpoints|group_by([.prior,.mode])|map(
    {prior:.[0].prior,mode:.[0].mode}+([.[]|splits("tests")]|rollup))),
  work:($endpoints|group_by([.prior,.mode])|map({prior:.[0].prior,mode:.[0].mode,
    gradient_records:([.[].training.work.gradient_records]|add),
    gradient_batches:([.[].training.work.gradient_batches]|add),
    effective_updates:([.[].training.work.effective_updates]|add),
    maximum_violation:([.[].training.final_normalized_violation]|max)})),
  maximum_product_residual:([$endpoints[].evaluation.splits[]
    .independent_complex_max_residual]|max),
  maximum_forecast_gib:([.cases[].buffer_forecast]|max|./1073741824),
  original_query_positions:([$endpoints[].evaluation.original_query_positions]|add),
  actual_product_queries:([$endpoints[].evaluation.actual_audit_calls]|add),
  training_references:([$endpoints[].evaluation.grouped_training_references]|add),
  maximum_complex_queries_uncovered:([$endpoints[].evaluation.splits[]
    .uncovered_scored_queries]|max),
  noise:[$endpoints[]|select(.case=="basis-parity-noise20")|{prior,mode,
    train:.evaluation.splits.train.metrics,
    clean_train_diagnostic:.evaluation.splits.train_clean.metrics,
    test:.evaluation.splits.test.metrics}],
  provisional_new_count:$report.new_all_modes_all_heldout99_cases,
  provisional_catalog_count:$report.candidate_catalog_all_modes_all_heldout99_cases
}
