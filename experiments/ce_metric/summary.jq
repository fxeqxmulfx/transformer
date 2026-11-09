# Complete frozen comparison; failed slots remain in the inventory.
def require($ok; $message): if $ok then . else error($message) end;
def average(f): [.[]|f]|if length==0 then null else add/length end;
def span(f): [.[]|f]|if length==0 then null else [min,max] end;
def statistics:
  {cases:length,current_condition_passed:([.[]|select(.sufficient_condition)]|length),
   global_raw_condition_passed:([.[]|select(.global_condition_raw)]|length),
   actual_increases:([.[]|select(.actual_increase)]|length),
   stable_actual_increases:([.[]|select(.actual_increase and .sufficient_condition)]|length),
   reference_increases:([.[]|select(.baseline_change>1e-8*(1+(.before|fabs)))]|length),
   rmax:span(.rmax),ordinary_energy_bound_ratio:span(.ordinary_energy_bound_ratio),
   stability:span(.stability_number),global_stability_raw:span(.global_stability_number_raw),
   actual_change:span(.actual_change),relative_actual_change:span(.relative_actual_change),
   mean_relative_actual:average(.relative_actual_change),
   mean_relative_baseline:average(.relative_baseline_change),
   mean_relative_linear:{decay:average(.relative_linear[0]),adaptive:average(.relative_linear[1]),
     tied_encoder:average(.relative_linear[2]),execution:average(.relative_linear[3])},
   mean_relative_quadratic:average(.relative_quadratic),
   maximum_closure:([.[].closure_scaled]|max)};

. as $report
| ["lr001","lr01","settle"] as $cohorts
| [1,3,5,7,11,13,17,19] as $seeds
| [0,1000,4000,6000,8000,9500] as $anchors
| [$cohorts[] as $c|$seeds[] as $s|$anchors[]|[$c,$s,.]] as $expected
| require(([.cases[]|[.cohort,.seed,.anchor]]|sort)==$expected; "case inventory")
| require(all(.cases[]|select(.status=="pass");
    ([.diagnosis.tracks[]|[.metric,.method]]|sort)==
      [["feature","raw"],["feature","stable_control"],["isotropic","raw"],["isotropic","stable_control"]]); "track inventory")
| [.cases[]|select(.status=="pass")] as $passed
| [$passed[] as $c|$c.diagnosis.tracks[]|.+{cohort:$c.cohort,seed:$c.seed,anchor:$c.anchor,eta:$c.diagnosis.eta}] as $tracks
| {inventory:{cases:($report.cases|length),passed:($passed|length),
     failed:([$report.cases[]|select(.status!="pass")]|length),
     zero_eta:([$passed[]|select(.diagnosis.eta==0)]|length),
     moving:([$passed[]|select(.diagnosis.eta>0)]|length)},
   passed:$report.passed,seconds:$report.seconds,combined_seconds:$report.combined_seconds,
   source_sha256:$report.source_sha256,construction_sha256:$report.construction_sha256,
   failures:[$report.cases[]|select(.status!="pass")],
   checks:{partial_gradient_error:([$passed[].diagnosis.partial_gradient_error]|max),
     maximum_feature_energy:([$passed[].diagnosis.maximum_feature_energy]|max),
     closure:([$tracks[].closure_scaled]|max)},
   tracks:($tracks|group_by([.metric,.method])|map({metric:.[0].metric,method:.[0].method,
     all:statistics,moving:(map(select(.eta>0))|statistics)})),
   by_anchor:($tracks|group_by([.cohort,.anchor,.metric,.method])|map(
     {cohort:.[0].cohort,anchor:.[0].anchor,metric:.[0].metric,method:.[0].method,statistics:statistics})),
   condition_failures:[$tracks[]|select(.sufficient_condition|not)|
     {cohort,seed,anchor,metric,method,stability_number,baseline_change,actual_change}],
   actual_increases:[$tracks[]|select(.actual_increase)|
     {cohort,seed,anchor,metric,method,sufficient_condition,actual_change}],
   late_settling:($tracks|map(select(.cohort=="settle" and .anchor>=4000))|
     group_by([.metric,.method])|map({metric:.[0].metric,method:.[0].method,statistics:statistics})),
   scope:"Checkpoint-fixed weights; current-q stability is diagnostic, global-q condition concerns only the reference. Different potentials are not optimizer interventions."}
