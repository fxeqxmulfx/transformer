# Regenerate all probe rows, predeclared threshold crossings and audit totals.
def bridge($rank; $method): .audit.bridge.validation[] | select(.rank==$rank and .method==$method);
def first_at($rows; $field; $threshold):
  [$rows[] | select(.[$field] >= $threshold) | .step] | if length==0 then null else .[0] end;
. as $report |
[.cases[] | select(.status=="pass") |
  {seed,step,old_accuracy:.audit.old_metrics.validation.accuracy,
   new_accuracy:.audit.new_metrics.validation.accuracy,
   old_rank32:(bridge(32;"old_geometry")|.coverage),
   combined_rank32:(bridge(32;"combined_bridge")|.coverage),
   directional_rank32:(bridge(32;"directional_bridge")|.coverage),
   direct:(.audit.source.subsets.validation.methods[]|select(.method=="directional")|.coverage),
   directional_ranks:[.audit.bridge.validation[]|select(.method=="directional_bridge")],
   seconds:.audit.seconds}] as $rows |
{status,failed,seconds,source_sha256,
 input_files:(.inputs.files|length), original_hashes:(.replay_hashes|length),
 matched_task196_updates:([.cases[].audit.task196_state_match|select(.==true)]|length),
 old_geometry_max_violation:([.cases[].audit.old_geometry_max_violation]|max),
 feature_error_max:([.cases[].audit.source|.old_fp_max,.new_fp_max]|max),
 logit_error_max:([.cases[].audit.source.logit_error_max]|max),
 tangent_scaled_error_max:([.cases[].audit.tangent_scaled_error]|max),
 bridge_false_certificates:([.cases[].audit.bridge.false_certificates]|add),
 bridge_violations:([.cases[].audit.bridge.violations]|add),
 rows:$rows,
 crossings:[7,11|. as $seed |
   [$rows[]|select(.seed==$seed)] as $s |
   {seed:$seed, thresholds:[0.5,0.9,0.99|. as $t |
     {level:$t,old_accuracy:first_at($s;"old_accuracy";$t),
      new_accuracy:first_at($s;"new_accuracy";$t),
      old_geometry_rank32:first_at($s;"old_rank32";$t),
      bridge_rank32:first_at($s;"directional_rank32";$t),
      direct:first_at($s;"direct";$t)}]}],
 mean_seconds:($rows | {replay:(map(.seconds.replay)|add/length),
   forecast:(map(.seconds.forecast)|add/length),geometry:(map(.seconds.geometry)|add/length),
   readout:(map(.seconds.readout)|add/length),persistence:(map(.seconds.persistence)|add/length),
   audit:(map(.seconds.audit)|add/length),update:(map(.seconds.update)|add/length)}),
 pooled_validation:[0,32,128|. as $rank |
   [$report.cases[].audit.bridge.validation[]|select(.rank==$rank and .method=="directional_bridge") ] as $s |
   {rank:$rank,count:($s|map(.count)|add),correct:($s|map(.correct)|add),certified:($s|map(.certified)|add)}]}
