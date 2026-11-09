# Regenerate summaries only from the complete fixed inventory.
if .status != "pass" or .failed != 0 or (.traces|length) != 24 or (.cases|length) != 192
  or any(.traces[]; .status != "pass") or any(.cases[]; .status != "pass")
then error("incomplete persistence inventory") else . end |
def span: [min,max];
def total(f): map(f)|add;
{
  source_sha256, seconds,
  traces: [.traces|group_by(.cohort)[] |
    {cohort:.[0].cohort, seeds:length, prefixes:total(.audit.prefixes),
     moving_updates:total(.audit.moving_updates),
     minimum_projected_share_range:(map(.audit.minimum_projected_share_bound)|span),
     minimum_global_share_range:(map(.audit.minimum_global_share_bound)|span),
     minimum_actual_share_range:(map(.audit.minimum_actual_share)|span),
     final_active_radius_range:(map(.audit.final_projected_radii[0])|span),
     final_global_radius_range:(map(.audit.final_global_radius)|span),
     previous_worst_case_motion_range:(map(.audit.previous_worst_case_motion_bound)|span),
     actual_displacement_range:(map(.audit.actual_max_displacement)|span),
     max_bound_excess:(map(.audit.max_scaled_bound_excess)|max),
     thresholds: [range(4) as $i |
       {threshold:.[0].audit.thresholds[$i],
        projected_certified_prefixes:total(.audit.certified_prefixes[0][$i]),
        global_certified_prefixes:total(.audit.certified_prefixes[1][$i]),
        projected_full_windows:([.[]|select(.audit.certified_prefixes[0][$i]==4000)]|length),
        global_full_windows:([.[]|select(.audit.certified_prefixes[1][$i]==4000)]|length)}]}],
  forecasts: ([.cases[] as $c | $c.audit.coverage[] |
      . + {cohort:$c.cohort,seed:$c.seed,anchor:$c.anchor,end:$c.end,horizon:$c.horizon}] |
    group_by([.cohort,.horizon,.split,.method]) |
    map({cohort:.[0].cohort,horizon:.[0].horizon,split:.[0].split,method:.[0].method,
      cases:length,seeds:(map(.seed)|unique|length),count:total(.count),correct:total(.correct),
      certified:total(.certified),coverage:(total(.certified)/total(.count)),
      prior_certified:total(.prior_certified),retained:total(.retained),
      case_coverage_range:(map(.coverage)|span)})),
  case_rows: [.cases[]|{cohort,seed,anchor,end,horizon,coverage:.audit.coverage,
      parameter_displacement_norm:.audit.parameter_displacement_norm,
      mean_combined_feature_radius:.audit.mean_combined_feature_radius}],
  trace_rows:.traces
}
