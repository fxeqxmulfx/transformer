# Regenerable summaries; repeated states are observations, not independent data.
def aggregate($rows; $split):
  [$rows[] | .audit.checks.subsets[$split]] as $s |
  ($s | map(.count) | add) as $n |
  {observations:$n, actually_correct:($s|map(.actually_correct)|add),
   methods:[range(6) as $m |
     [$s[] | .methods[$m]] as $v |
     {method:$v[0].method,
      certified_correct:($v|map(.certified_correct)|add),
      coverage:(($v|map(.certified_correct)|add)/$n),
      certified_loss_decreased:($v|map(.certified_loss_decreased)|add),
      certified_loss_increased:($v|map(.certified_loss_increased)|add),
      loss_inconclusive:($v|map(.loss_inconclusive)|add),
      certified_margin_improved:($v|map(.certified_margin_improved)|add),
      certified_margin_regressed:($v|map(.certified_margin_regressed)|add),
      false_certificates:($v|map(.false_certificates)|add),
      mean_margin_width:(($v|map(.margin_width.mean*.margin_width.count)|add)/$n),
      mean_loss_width:(($v|map(.loss_width.mean*.loss_width.count)|add)/$n)}]};
def layers($rows):
  [$rows[].audit.checks.stages[]] | group_by(.name) |
  map(. as $s | ($s|map(.actual_error.count)|add) as $n |
    {layer:$s[0].name, observations:$n,
     mean_actual_error:(($s|map(.actual_error.mean*.actual_error.count)|add)/$n),
     mean_injected_defect:(($s|map(.injected_defect.mean*.injected_defect.count)|add)/$n),
     methods:[range(4) as $m |
       [$s[].methods[$m]] as $v |
       {method:$v[0].method,
        mean_radius:(($v|map(.radius.mean*.radius.count)|add)/$n),
        mean_effective_amplification:(($v|map(.effective_amplification.mean*.effective_amplification.count)|add)/$n),
        max_error_over_radius:($v|map(.max_error_over_radius)|max)}]});
def distribution:
  sort | {min:.[0],median:.[length/2|floor],max:.[-1],mean:(add/length),total:add};
. as $r | [.cases[]|select(.step>0)] as $trained |
{status, failed, source_sha256, reference_sha256, elapsed_seconds,
 input_files:(.inputs.files|length),
 cohorts:[{cohort:"all",rows:.cases},{cohort:"trained",rows:$trained}] |
   map(. as $c | {cohort,subsets:["train","validation","difficult_train","difficult_validation"] |
     map(. as $split | {split:$split,counts:aggregate($c.rows;$split)})}),
 trained_validation_by_arm:($trained|group_by(.arm)|map({arm:.[0].arm,counts:aggregate(.;"validation")})),
 trained_validation_by_seed:($trained|group_by(.seed)|map({seed:.[0].seed,counts:aggregate(.;"validation")})),
 case_validation:[.cases[] | {arm,seed,step,status,counts:aggregate([.];"validation")}],
 trained_layers:layers($trained),
 trained_layers_by_arm:($trained|group_by(.arm)|map({arm:.[0].arm,layers:layers(.)})),
 seconds:([.cases[].audit.seconds|to_entries[]]|group_by(.key)|map({phase:.[0].key,distribution:([.[].value]|distribution)})),
 payload_bytes:([.cases[].audit.forecast_payload_bytes]|distribution),
 capacity_bytes:([.cases[].audit.forecast_capacity_bytes]|distribution),
 persisted_binary_bytes:([.cases[].audit.binary.bytes]|distribution),
 matrix_bounds:[.cases[]|{arm,seed,step,bounds:.audit.matrix_bounds}],
 numerical:{old_feature_max:([.cases[].audit.checks.old_fp_max]|max),
   new_feature_max:([.cases[].audit.checks.new_fp_max]|max),
   logit_max:([.cases[].audit.checks.logit_error_max]|max),
   tangent_scaled_max:([.cases[].audit.tangent_scaled_error]|max),
   class_pairs_per_track:([.cases[].audit.checks.class_pairs_per_track]|add),
   layer_violations:([.cases[].audit.checks.layer_violations]|add),
   gap_violations:([range(6) as $m|[$r.cases[].audit.checks.gap_violations[$m]]|add]),
   outcome_violations:([range(6) as $m|[$r.cases[].audit.checks.outcome_violations[$m]]|add])}}
