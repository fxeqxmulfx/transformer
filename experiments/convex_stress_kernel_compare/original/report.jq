# Read-only interpretation of the frozen original comparison. It never selects
# a model or changes the frozen all-split diagnostic flags.
def require($condition; $message):
  if $condition then . else error($message) end;
def deltas($old; $new):
  [$new.metrics | to_entries[] | . as $split |
    ["sequence_accuracy", "loss"][] as $metric |
    $old.metrics[$split.key][$metric] as $before |
    $split.value[$metric] as $after |
    select(($before | type) == "number" and ($after | type) == "number") |
    {split: $split.key, metric: $metric, before: $before, after: $after,
      delta: ($after - $before)}];
def counts:
  group_by(.training_mode) | map({
    training_mode: .[0].training_mode, cases: length,
    primary_passes: (map(select(.primary_pass == true)) | length),
    all_split_passes: (map(select(.all_splits_pass == true)) | length),
    no_model: (map(select(.status == "no_model")) | length)
  });

. as $summary |
require(.tasks == 35 and (.records | length) == 35 and .choices_per_arm == 105;
  "complete original task scope required") |
require((.arms | map(.initializer) | sort) == ["first_small_ce", "minimum_ce"];
  "both initializer arms required") |
require((.complete_parent.complete_parent.complete_parent.complete_parent.transformer_endpoints | length) == 35
  and (.complete_parent.complete_parent.complete_parent.complete_parent.depth_outcomes | length) == 336;
  "complete conventional and depth baselines required") |
{
  source_fingerprint, producer_commit, tasks, process_failures,
  failed_cases: [.records[] | select(.status != "complete") |
    {task, status, run_exit, replay_exit, prior_choices_retained}],
  constructor_accounting,
  conventional_transformer_endpoints:
    (.complete_parent.complete_parent.complete_parent.complete_parent.transformer_endpoints | length),
  depth_outcomes: (.complete_parent.complete_parent.complete_parent.complete_parent.depth_outcomes | length),
  evaluated_candidates: [.records[] | .result.evaluation[]? |
    select(.status == "evaluated") | {
      task, training_mode, prime, offer_eligible, primary_pass, all_splits_pass, train_ce,
      primary_split, primary_accuracy: .metrics[.primary_split].sequence_accuracy,
      support: (.known_inputs | with_entries(.value |= {
        calls, known_calls, unknown_calls, unknown_axis_calls, unmatched_active_calls
      }))
    }],
  refused_modes: [.records[] | .result.evaluation[]? | select(.status == "refused") |
    {task, training_mode, prime, reason}],
  arms: [.arms[] | . as $arm |
    require((.choices | length) == 105
      and ([.choices[] | [.task, .training_mode]] | unique | length) == 105;
      "missing or repeated task/mode choice") |
    ($summary.complete_parent.arms[] | select(.initializer == $arm.initializer)) as $parent |
    (.choices | counts) as $counts |
    require($counts == (.by_mode | sort_by(.training_mode)); "stored counts disagree") |
    [.choices[] | . as $new |
      ($parent.choices[] | select(.task == $new.task and .training_mode == $new.training_mode)) as $old |
      select(.reference != $old.reference) |
      deltas($old; $new) as $deltas |
      {task, training_mode, old_reference: $old.reference, new_reference: .reference,
        old_primary_pass: $old.primary_pass, primary_pass,
        gained_primary_pass: ($old.primary_pass != true and .primary_pass == true),
        primary_regression: ($old.primary_pass == true and .primary_pass != true),
        accuracy_decreases: [$deltas[] | select(.metric == "sequence_accuracy" and .delta < -1e-12)],
        loss_increases: [$deltas[] | select(.metric == "loss" and .delta > 1e-12)],
        split_deltas: $deltas}
    ] as $changes |
    {initializer, by_mode: $counts,
      all_primary_pass: all(.choices[]; .primary_pass == true),
      complete_primary_pass: ($summary.process_failures == 0 and all(.choices[]; .primary_pass == true)),
      stored_full35_flag: .full35_pass,
      changed_choices: ($changes | length),
      gained_primary_passes: ([$changes[] | select(.gained_primary_pass)] | length),
      primary_regressions: ([$changes[] | select(.primary_regression)] | length),
      accuracy_decreases: ([$changes[].accuracy_decreases[]] | length),
      loss_increases: ([$changes[].loss_increases[]] | length),
      failures: [.choices[] | select(.primary_pass != true) |
        {task, training_mode, status, primary_split,
          primary_accuracy: (if .primary_split then .metrics[.primary_split].sequence_accuracy else null end)}],
      changes: $changes}
  ]
}
