# Generated summaries remain local; retain failures and every declared query set.
def pool:
  map(select(.queries > 0)) as $rows |
  ($rows | map(.queries) | add // 0) as $n |
  {
    query_steps: $n,
    actual_correct: ($rows | map(.old_new_correct[1]) | add // 0),
    joint_correct: ($rows | map(.joint.certified_correct) | add // 0),
    separate_correct: ($rows | map(.separate.certified_correct) | add // 0),
    joint_loss_decreased: ($rows | map(.joint.certified_loss_decreased) | add // 0),
    joint_loss_increased: ($rows | map(.joint.certified_loss_increased) | add // 0),
    joint_loss_inconclusive: ($rows | map(.joint.loss_inconclusive) | add // 0),
    separate_loss_inconclusive: ($rows | map(.separate.loss_inconclusive) | add // 0),
    mean_joint_margin_interval_width: (if $n > 0 then
      ($rows | map(.queries * .joint.mean_margin_interval_width) | add) / $n else null end),
    mean_separate_margin_interval_width: (if $n > 0 then
      ($rows | map(.queries * .separate.mean_margin_interval_width) | add) / $n else null end)
  };
{
  status, failed, elapsed_seconds, source_sha256,
  trained_cohorts: {
    train: ([.cases[] | select(.step > 0) | .audit.query_control.train.all] | pool),
    validation: ([.cases[] | select(.step > 0) | .audit.query_control.validation.all] | pool),
    validation_difficult: ([.cases[] | select(.step > 0) |
      .audit.query_control.validation.old_bottom_decile] | pool)
  },
  cases: [.cases[] | {
    arm, seed, step, post_budget_probe, status, error,
    new_state_sha256: .audit.new_state_sha256,
    query_control: (.audit.query_control | del(.rows_by_canonical_pair_index,
      .train.old_bottom_decile_indices, .train.old_error_indices,
      .validation.old_bottom_decile_indices, .validation.old_error_indices))
  }],
  checks: {
    cases: (.cases | length),
    bitwise_replays: ([.cases[] | select(.audit.bitwise_train_step_match == true)] | length),
    inputs_unchanged: .input_verification.unchanged,
    input_files: (.input_verification.files | length),
    queries: ([.cases[].audit.query_control.checks.queries // 0] | add),
    competitor_pairs: ([.cases[].audit.query_control.checks.competitor_pairs // 0] | add),
    max_scaled_identity_error: ([.cases[].audit.query_control.checks.max_scaled_identity_error] | max),
    max_scaled_gap_bound_excess: ([.cases[].audit.query_control.checks.max_scaled_gap_bound_excess] | max),
    max_scaled_weighted_loss_error: ([.cases[].audit.query_control.checks.max_scaled_weighted_loss_error] | max),
    false_certificates: ([.cases[].audit.query_control | (.train, .validation) |
      (.all, .old_bottom_decile, .old_error_rows).false_certificates // 0] | add)
  }
}
