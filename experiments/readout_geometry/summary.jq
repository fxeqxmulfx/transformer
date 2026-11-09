# Fixed comparisons, threshold crossings, numerical closure and attribution.
def bridge($family; $rank; $method):
  .audit.bridges[$family].validation[] | select(.rank==$rank and .method==$method);
def first_at($rows; $field; $level):
  [$rows[]|select(.[$field]>=$level)|.step] | if length==0 then null else .[0] end;
. as $report |
[.cases[]|select(.status=="pass")|
 {seed,step,old_accuracy:.audit.old_metrics.validation.accuracy,
  new_accuracy:.audit.new_metrics.validation.accuracy,
  old_loss:.audit.old_metrics.validation.loss,new_loss:.audit.new_metrics.validation.loss,
  hidden:(bridge("hidden";32;"directional_bridge")|.coverage),
  readout:(bridge("readout";32;"directional_bridge")|.coverage),
  old_readout:(bridge("readout";32;"old_geometry")|.coverage),
  direct:(.audit.source.subsets.validation.methods[]|select(.method=="directional")|.coverage),
  readout_ranks:[.audit.bridges.readout.validation[]|select(.method=="directional_bridge")],
  attribution:.audit.attribution.validation,training_attribution:.audit.attribution.train,
  seconds:.audit.seconds}] as $rows |
{status,failed,seconds,source_sha256,
 input_files:(.inputs.files|length),original_hashes:(.replay_hashes|length),
 matched_task196_updates:([.cases[].audit.task196_state_match|select(.==true)]|length),
 instrumented_matches:([.cases[].audit.instrumented_step_matches|select(.==true)]|length),
 false_certificates:([.cases[].audit.bridges[].false_certificates]|add),
 bound_violations:([.cases[].audit.bridges[].violations]|add),
 future_real_violations:([.cases[].audit.future_real_violations]|add),
 max_parameter_closure:([.cases[].audit.attribution.parameter_closure_max_error]|max),
 max_tangent_closure:([.cases[].audit.attribution.tangent_closure_scaled_error]|max),
 max_logit_closure:([.cases[].audit.attribution.logit_closure_scaled_error]|max),
 max_readout_gradient_error:([.cases[].audit.full_train_readout_gradient_error]|max),
 feature_error_max:([.cases[].audit.source|.old_fp_max,.new_fp_max]|max),
 logit_error_max:([.cases[].audit.source.logit_error_max]|max),
 rows:$rows,
 crossings:[7,11|. as $seed|[$rows[]|select(.seed==$seed)] as $s|
   {seed:$seed,thresholds:[0.5,0.9,0.99|. as $t|
    {level:$t,old_accuracy:first_at($s;"old_accuracy";$t),
     hidden:first_at($s;"hidden";$t),readout:first_at($s;"readout";$t),
     old_readout:first_at($s;"old_readout";$t),direct:first_at($s;"direct";$t)}]}],
 mean_seconds:(reduce ($rows[].seconds|to_entries[]) as $e
   ({}; .[$e.key] = ((.[$e.key] // 0) + $e.value/($rows|length)))),
 pooled_validation:["hidden","readout"|. as $family|0,32,128|. as $rank|
   [$report.cases[].audit.bridges[$family].validation[]|
    select(.rank==$rank and .method=="directional_bridge")] as $s|
   {family:$family,rank:$rank,count:($s|map(.count)|add),correct:($s|map(.correct)|add),
    certified:($s|map(.certified)|add)}]}
