# Complete inventory and raw fixed-reference conditions; no seed selection.
def count_if(f): map(select(f)) | length;
{
  status, failed, seconds, total_seconds,
  slots: (.cases | length),
  endpoints: ([.cases[].endpoints[]?] | length),
  endpoint_statuses: ([.cases[].endpoints[]?] | group_by(.status) | map({status:.[0].status,count:length})),
  maxima: ([.cases[].endpoints[]? | select(.status=="pass" or .status=="fail")] |
    {parameter_scaled_excess: (map(.parameter_scaled_excess)|max),
     feature_scaled_excess: (map(.feature_scaled_excess)|max),
     contrast_scaled_excess: (map(.contrast_scaled_excess)|max),
     probability_excess: (map(.probability_excess)|max),
     feature_execution_error: (map(.execution.feature_error)|max),
     readout_execution_error: (map(.execution.readout_error)|max),
     feature_energy: (map(.execution.maximum_feature_energy)|max)}),
  groups: (.cases | group_by([.cohort,.anchor]) | map(. as $cases |
    {cohort:.[0].cohort, anchor:.[0].anchor, initialization:(.[0].anchor==0), slots:length,
     verified:count_if(.status=="pass"),
     old_q_range: ([.[].old_q | select(.!=null)] | [min,max]),
     metrics: [range(2) as $m | {
       metric: $cases[0].metrics[$m].metric,
       universal_passes: ($cases | count_if(.metrics[$m].universal_coefficient!=null and .metrics[$m].universal_coefficient<=2)),
       fixed_old_q_passes_invalid_as_forecast: ($cases | count_if(.metrics[$m].old_q_coefficient_invalid_as_forecast!=null and .metrics[$m].old_q_coefficient_invalid_as_forecast<=2)),
       horizons: ([$cases[].endpoints[]?] | group_by(.horizon) | map({horizon:.[0].horizon,
         slots:length, verified:count_if(.status=="pass"),
         transported_passes:count_if(.coefficients[$m] != null and .coefficients[$m]<=2),
         full_potential_increases:count_if(.full_potentials[$m].increase==true),
         forecast_q_range:([.[].forecast_q|select(.!=null)]|[min,max]),
         actual_q_range:([.[].actual_q|select(.!=null)]|[min,max])})),
       passing_positive_horizons_by_seed: [$cases[] | {seed,
         horizon:([.endpoints[]? | select(.status=="pass" and .horizon>0 and .coefficients[$m]<=2) | .horizon] | max // 0)}]
     }]})),
  next_step: ([.cases[] | . as $case | .next_step.tracks[]? |
      . + {cohort:$case.cohort,anchor:$case.anchor,seed:$case.seed}] |
      group_by([.metric,.method]) | map({metric:.[0].metric,method:.[0].method,slots:length,
        current_batch_condition_passes:count_if(.sufficient_condition),
        full_increases:count_if(.actual_increase),
        maximum_closure_error:(map(.closure_scaled)|max)})),
  errors: [.cases[] | select(.status!="pass")]
}
