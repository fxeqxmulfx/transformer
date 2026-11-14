# Interpret sealed original evidence; never select a model or modify success.
def require($ok; $message): if $ok then . else error($message) end;
def counts:
  group_by(.training_mode) | map({
    training_mode: .[0].training_mode, cases: length,
    primary_passes: (map(select(.primary_pass == true)) | length),
    all_split_passes: (map(select(.all_splits_pass == true)) | length),
    no_model: (map(select(.status != "selected")) | length),
    fresh_metrics: (map(select(.metric_provenance.kind == "fresh_changed_model")) | length),
    inherited_metrics: (map(select(.metric_provenance.kind == "inherited_reviewed_parent")) | length)
  });
def support_counts:
  with_entries(.value |= with_entries(select((.value | type) == "number")));
def route_totals:
  reduce .[] as $m (
    {calls: 0, supported_calls: 0, parent_unavailable_calls: 0,
     parent_route_calls: 0, candidate_without_parent_calls: 0};
    .calls += $m.calls |
    .supported_calls += $m.supported_calls |
    .parent_unavailable_calls += ($m.calls - $m.parent_available_calls) |
    .parent_route_calls += $m.parent_route_calls |
    .candidate_without_parent_calls += $m.candidate_without_parent_calls
  );

. as $summary |
require(.tasks == 35 and (.records | length) == 35 and .choices_per_arm == 105
  and ([.records[].task] | unique | length) == 35;
  "complete original 35-task scope required") |
require((.arms | map(.initializer) | sort) == ["first_small_ce", "minimum_ce"];
  "both initializer arms required") |
.complete_parent.complete_parent.complete_parent.complete_parent as $base |
require(($base.transformer_endpoints | length) == 35 and ($base.depth_outcomes | length) == 336;
  "complete conventional and depth baselines required") |
{
  source_fingerprint, producer_commit, inventory_sha256, tasks, process_failures,
  success_rule, metric_provenance, constructor_accounting,
  conventional_transformer_endpoints: ($base.transformer_endpoints | length),
  depth_outcomes: ($base.depth_outcomes | length),
  failed_cases: [.records[] | select(.status != "complete") |
    {task, status, run_exit, replay_exit}],
  fresh_endpoints: [.records[] | .result.outcomes[]? | {
    task, training_mode, reference, primary_pass, all_splits_pass, train_ce,
    primary_split, primary_accuracy: .metrics[.primary_split].sequence_accuracy,
    metric_provenance, metric_scalar_work_forecast,
    support: (.prediction_masks | support_counts)
  }],
  partial_parent_routes: [.records[] | .result.outcomes[]? |
    select(any(.prediction_masks[]; has("parent_available_calls"))) |
    {task, training_mode, reference, totals: (.prediction_masks | route_totals)}],
  transformations: [.records[] | .result.seal.choices.choices[]? | {
    initializer, task, training_mode, prior_reference, selected, train_ce,
    transformation: (.transformation | {status, reason, coverage, predecessor_reference})
  }],
  changes,
  arms: [.arms[] | . as $arm |
    require((.choices | length) == 105
      and ([.choices[] | [.task, .training_mode]] | unique | length) == 105;
      "missing or repeated task/mode choice") |
    (.choices | counts) as $counts |
    require($counts == (.by_mode | sort_by(.training_mode)); "stored counts disagree") |
    require(.full35_pass == all(.choices[]; .primary_pass == true);
      "stored primary full-suite flag differs") |
    {initializer, by_mode: $counts, full35_pass,
      all_split_diagnostic_pass,
      complete_primary_pass: ($summary.process_failures == 0 and .full35_pass),
      choices: [.choices[] | {
        task, training_mode, status, reference, primary_split, primary_pass,
        all_splits_pass, metric_provenance,
        primary_accuracy: (if .primary_split then .metrics[.primary_split].sequence_accuracy else null end)
      }],
      failures: [.choices[] | select(.primary_pass != true) | {
        task, training_mode, status, primary_split,
        primary_accuracy: (if .primary_split then .metrics[.primary_split].sequence_accuracy else null end)
      }]
    }
  ]
}
