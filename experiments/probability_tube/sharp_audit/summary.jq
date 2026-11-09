# Complete matched comparison, with zero useful horizons retained.
def count_if(f): map(select(f)) | length;
{
  status, failed, seconds, total_seconds,
  slots: (.cases | length),
  endpoints: ([.cases[].endpoints[]?] | length),
  maxima: ([.cases[].endpoints[]?] |
    {feature_excess:(map(.feature_excess)|max),
     feature_scaled_excess:(map(.feature_scaled_excess)|max),
     probability_excess:(map(.probability_excess)|max)}),
  groups: (.cases | group_by([.cohort,.anchor]) | map(. as $cases |
    {cohort:.[0].cohort, anchor:.[0].anchor, initialization:(.[0].anchor==0),
     slots:length, matched:count_if(.status=="pass"),
     metrics: [range(2) as $m | {
       metric:$cases[0].metrics[$m].metric,
       horizons: ([$cases[].endpoints[]?] | group_by(.horizon) | map({
         horizon:.[0].horizon, slots:length, matched:count_if(.status=="pass"),
         original_passes:count_if(.original_coefficients[$m]!=null and .original_coefficients[$m]<=2),
         refined_passes:count_if(.refined_coefficients[$m]!=null and .refined_coefficients[$m]<=2),
         original_q_range:([.[].original_q|select(.!=null)]|[min,max]),
         refined_q_range:([.[].refined_q|select(.!=null)]|[min,max]),
         actual_q_range:([.[].actual_q|select(.!=null)]|[min,max]),
         full_potential_increases:count_if(.full_potentials[$m].increase==true)})),
       passing_positive_horizons_by_seed: [$cases[] | {seed,
         original:([.endpoints[]? | select(.status=="pass" and .horizon>0 and .original_coefficients[$m]!=null and .original_coefficients[$m]<=2)|.horizon]|max//0),
         refined:([.endpoints[]? | select(.status=="pass" and .horizon>0 and .refined_coefficients[$m]!=null and .refined_coefficients[$m]<=2)|.horizon]|max//0)}]
     }]})),
  errors:[.cases[]|select(.status!="pass")]
}
