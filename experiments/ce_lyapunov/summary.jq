# Native rendering of the complete frozen inventory; failures remain visible.
def require($ok; $message): if $ok then . else error($message) end;
def average(f): [ .[] | f ] | if length == 0 then null else add/length end;
def span(f): [ .[] | f ] | if length == 0 then null else [min,max] end;
def statistics:
  {cases:length,
   condition_passed:([.[]|select(.sufficient_condition)]|length),
   condition_failed:([.[]|select(.sufficient_condition|not)]|length),
   reference_increases:([.[]|select(.baseline_change >
     1e-8*(1+([(.before|fabs),(.reference_after|fabs)]|max)))]|length),
   actual_increases:([.[]|select(.actual_increase)]|length),
   stable_but_actual_increases:([.[]|select(.sufficient_condition and .actual_increase)]|length),
   effective_step:span(.s), stability_number:span(.stability_number),
   baseline_change:span(.baseline_change), actual_change:span(.actual_change),
   mean_baseline:average(.baseline_change), mean_actual:average(.actual_change),
   mean_linear:{decay:average(.perturbation_linear[0]),
     adaptive:average(.perturbation_linear[1]),
     tied_encoder:average(.perturbation_linear[2]),
     execution:average(.perturbation_linear[3])},
   mean_total_quadratic:average(.perturbation_quadratic),
   maximum_closure:([.[]|.closure_scaled]|max)};

. as $reports
| require((map(.cohort)|sort)==["lr001","lr01","settle"]; "cohort inventory")
| [1,3,5,7,11,13,17,19] as $seeds
| [0,1000,4000,6000,8000,9500] as $anchors
| [$seeds[] as $s | $anchors[] | [$s,.]] as $expected
| require(all(.[]; ([.cases[]|[.seed,.anchor]]|sort)==$expected); "seed/anchor inventory")
| require(all(.[]|.cases[]|select(.status=="pass");
    ([.diagnosis.tracks[].method]|sort)==["raw","stable_control"]); "track inventory")
| [.[] as $r | $r.cases[] | . + {cohort:$r.cohort}] as $cases
| [$cases[]|select(.status=="pass")] as $passed
| [$passed[] as $c | $c.diagnosis.tracks[] |
    . + {cohort:$c.cohort,seed:$c.seed,anchor:$c.anchor,eta:$c.diagnosis.eta}] as $tracks
| {inventory:{cohorts:3,seeds:$seeds,anchors:$anchors,cases:($cases|length),
     passed:($passed|length),failed:([$cases[]|select(.status!="pass")]|length),
     zero_eta:([$passed[]|select(.diagnosis.eta==0)]|length),
     moving:([$passed[]|select(.diagnosis.eta>0)]|length)},
   reports:[$reports[]|{cohort,status,failed,seconds,source_sha256,trainer_source_sha256,
     optimizer_ptx_sha256}],
   failures:[$cases[]|select(.status!="pass")|{cohort,seed,anchor,status,error}],
   numeric_checks:{
     partial_gradient_error:([$passed[].diagnosis.partial_gradient_error]|max),
     maximum_feature_energy:([$passed[].diagnosis.max_feature_energy]|max),
     optimizer_ulps:([$passed[].optimizer.max_ulps]|max),
     optimizer_norm_error:([$passed[].optimizer.norm_relative_error]|max),
     closure:([$tracks[].closure_scaled]|max)},
   by_method:($tracks|group_by(.method)|map({method:.[0].method,all:statistics,
     moving:(map(select(.eta>0))|statistics)})),
   by_anchor:($tracks|group_by([.cohort,.anchor,.method])|map(
     {cohort:.[0].cohort,anchor:.[0].anchor,method:.[0].method,statistics:statistics})),
   coefficient_ranges:($passed|group_by([.cohort,.anchor])|map(
     {cohort:.[0].cohort,anchor:.[0].anchor,q:span(.diagnosis.q),
      L:span(.diagnosis.L),clip:span(.diagnosis.clip),alpha:span(.diagnosis.alpha)})),
   condition_failures:($tracks|map(select(.sufficient_condition|not)|
     {cohort,seed,anchor,method,stability_number,baseline_change,actual_change})),
   actual_increases:($tracks|map(select(.actual_increase)|
     {cohort,seed,anchor,method,sufficient_condition,baseline_change,actual_change})),
   late_settling:($tracks|map(select(.cohort=="settle" and .anchor>=4000))|
     group_by(.method)|map({method:.[0].method,statistics:statistics})),
   scope:"Indexed one-step diagnoses, not independent trials or future certificates; s and q vary by case."}
