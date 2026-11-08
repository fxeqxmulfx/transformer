# Generated results remain local. Pool query-update observations, not independent samples.
def pool:
  map(select(.queries > 0)) as $rows |
  ($rows | map(.queries) | add // 0) as $n |
  {
    query_steps: $n,
    actual_correct: ($rows | map(.old_new_correct[1]) | add // 0),
    forecast_correct: ($rows | map(.forecast.certified_correct) | add // 0),
    retrospective_correct: ($rows | map(.retrospective_joint.certified_correct) | add // 0),
    forecast_loss_decreased: ($rows | map(.forecast.certified_loss_decreased) | add // 0),
    forecast_loss_increased: ($rows | map(.forecast.certified_loss_increased) | add // 0),
    forecast_loss_inconclusive: ($rows | map(.forecast.loss_inconclusive) | add // 0),
    retrospective_loss_inconclusive: ($rows | map(.retrospective_joint.loss_inconclusive) | add // 0),
    mean_forecast_margin_width: (if $n>0 then
      ($rows | map(.queries*.forecast.mean_margin_interval_width) | add)/$n else null end),
    mean_retrospective_margin_width: (if $n>0 then
      ($rows | map(.queries*.retrospective_joint.mean_margin_interval_width) | add)/$n else null end)
  };
{
  status,failed,elapsed_seconds,source_sha256,forecast_file,
  trained_cohorts: {
    train: ([.cases[] | select(.step>0) | .audit.feature_forecast.train.all] | pool),
    validation: ([.cases[] | select(.step>0) | .audit.feature_forecast.validation.all] | pool),
    validation_difficult: ([.cases[] | select(.step>0) | .audit.feature_forecast.validation.old_bottom_decile] | pool)
  },
  cases: [.cases[] | {
    arm,seed,step,post_budget_probe,status,error,
    new_state_sha256:.audit.new_state_sha256,
    mean_real_feature_error: ([.audit.feature_forecast.feature_rows[]?[0]] | if length>0 then add/length else null end),
    mean_real_feature_radius: ([.audit.feature_forecast.feature_rows[]?[1]] | if length>0 then add/length else null end),
    forecast:(.audit.feature_forecast | del(.feature_rows,.query_rows))
  }],
  checks: {
    cases:(.cases|length),
    bitwise_replays:([.cases[]|select(.audit.bitwise_train_step_match==true)]|length),
    inputs_unchanged:.input_verification.unchanged,
    input_files:(.input_verification.files|length),
    queries:([.cases[].audit.feature_forecast.checks.queries // 0]|add),
    competitor_pairs:([.cases[].audit.feature_forecast.checks.competitor_pairs // 0]|add),
    max_tangent_error:([.cases[].audit.feature_forecast.checks.tangent_scaled_error]|max),
    max_error_radius_ratio:([.cases[].audit.feature_forecast.checks.max_error_radius_ratio]|max),
    max_old_fp32_feature_error:([.cases[].audit.feature_forecast.checks.max_old_fp32_feature_error]|max),
    max_new_fp32_feature_error:([.cases[].audit.feature_forecast.checks.max_new_fp32_feature_error]|max),
    max_fp32_logit_error:([.cases[].audit.feature_forecast.checks.max_fp32_logit_error]|max),
    max_scaled_gap_excess:([.cases[].audit.feature_forecast.checks.max_scaled_gap_excess]|max),
    violations:([.cases[].audit.feature_forecast.checks.violations // 0]|add),
    false_certificates:([.cases[].audit.feature_forecast|(.train,.validation)|
      (.all,.old_bottom_decile,.old_error_rows).false_certificates // 0]|add)
  }
}
