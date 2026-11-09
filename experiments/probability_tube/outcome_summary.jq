# Additional complete-inventory summaries for either frozen verifier report.
def count_if(f): map(select(f)) | length;
def extent: [min,max];
{
  status,failed,seconds,total_seconds,
  slots:(.cases|length),
  cases_by_status:(.cases|group_by(.status)|map({status:.[0].status,count:length})),
  metrics:[range(2) as $m | [.cases[]|select(.anchor>0)] as $cases |
    {metric:$cases[0].metrics[$m].metric,moving_slots:($cases|length),
     universal_passes:($cases|count_if(.metrics[$m].universal_coefficient!=null and .metrics[$m].universal_coefficient<=2)),
     old_q_passes_invalid_as_forecast:($cases|count_if(.metrics[$m].old_q_coefficient_invalid_as_forecast!=null and .metrics[$m].old_q_coefficient_invalid_as_forecast<=2)),
     positive_horizons:([$cases[].endpoints[]?|select(.horizon>0)]|length),
     transported_passes:([$cases[].endpoints[]?|select(.horizon>0)]|count_if(.coefficients[$m]!=null and .coefficients[$m]<=2)),
     anchors_with_positive_horizon:($cases|count_if(any(.endpoints[]?;.horizon>0 and .status=="pass" and .coefficients[$m]!=null and .coefficients[$m]<=2))),
     full_increases:([$cases[].endpoints[]?|select(.horizon>0)]|count_if(.full_potentials[$m].increase==true)),
     full_increases_despite_transported_condition:([$cases[].endpoints[]?|select(.horizon>0)]|count_if(.full_potentials[$m].increase==true and .coefficients[$m]!=null and .coefficients[$m]<=2)),
     ordinary_energy_bound_ratio:([$cases[].metrics[$m].ordinary_energy_bound_ratio|select(.!=null)]|extent),
     universal_coefficient:([$cases[].metrics[$m].universal_coefficient|select(.!=null)]|extent),
     rmax:([$cases[].metrics[$m].rmax|select(.!=null)]|extent)}],
  next_step: ([.cases[]|. as $case|.next_step.tracks[]?|
    .+{cohort:$case.cohort,anchor:$case.anchor,seed:$case.seed}]|
    group_by([.metric,.method,.anchor==0])|map({metric:.[0].metric,method:.[0].method,
      initialization:(.[0].anchor==0),slots:length,
      condition_passes:count_if(.sufficient_condition),full_increases:count_if(.actual_increase),
      full_increases_with_condition:count_if(.sufficient_condition and .actual_increase),
      relative_adaptive_credit_mean:(map(.relative_linear[1])|add/length),
      relative_encoder_credit_mean:(map(.relative_linear[2])|add/length),
      relative_quadratic_mean:(map(.relative_quadratic)|add/length),
      closure_maximum:(map(.closure_scaled)|max)})),
  moving_endpoint_probabilities: ([.cases[]|select(.anchor>0)|.endpoints[]?|select(.horizon>0)] |
    {slots:length,forecast_range:(map(.forecast_q)|extent),actual_range:(map(.actual_q)|extent),
     saturated_forecasts:count_if(.forecast_q==1),
     fixed_old_q_excess_maximum:(map(.old_q_excess_invalid_forecast)|max),
     fixed_old_q_violations:count_if(.old_q_excess_invalid_forecast!=null and .old_q_excess_invalid_forecast>1e-8)}),
  numerical_maxima: {partial_gradient_error:([.cases[].next_step.partial_gradient_error|select(.!=null)]|max),
    full_identity_scaled_error:([.cases[].next_step.tracks[]?.closure_scaled]|max)},
  errors:[.cases[]|select(.status!="pass")]
}
