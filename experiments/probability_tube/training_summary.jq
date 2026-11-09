# Run with jq -s over all three cohort reports. Retain every seed and probe time.
def count_if(f): map(select(f)) | length;
def stats: map(select(.!=null)) |
  {available:length,min:min,mean:(if length>0 then add/length else null end),max:max};
map(
  {cohort,status,seconds,total_seconds,seed_slots:(.seeds|length),
   completed_updates:([.seeds[].completed]|add),
   geometry_probes:([.seeds[].geometry[]?]|length),
   checkpoints:([.seeds[].checkpoints[]?]|length),
   instrumented_matches:([.seeds[].instrumented_matches]|add),
   optimizer_max_ulps:([.seeds[].optimizer_checks[]?.max_ulps]|max),
   false_certificates:([.seeds[].geometry[]?.false_certificates]|add),
   damping_failures:([.seeds[].dynamics.damping_failures]|add),
   maximum_update_closure:([.seeds[].dynamics.max_closure_scaled]|max),
   seeds:[.seeds[]|{seed,status,completed,source_checks:.independent_source_checks.passed,
     geometry_probes:(.geometry|length),checkpoints:(.checkpoints|length),
     final_metrics:.metrics[-1],final_structure:.geometry[-1].structural,
     final_coverage:.geometry[-1].coverage,dynamics,
     observed_hold_minimum:([.geometry[]?|select(.step>=4000)] |
       {accuracy:([.[].metrics.validation.accuracy]|min),
        active_fraction:([.[].structural.active_fraction]|min),
        readout32_coverage:([.[].coverage.validation[]?|select(.family=="readout" and .rank==32)|.coverage]|min)})}],
   probe_groups: ([.seeds[]|. as $seed|.geometry[]?|. + {seed:$seed.seed}] |
     group_by(.step)|map({step:.[0].step,slots:length,passed:count_if(.passed),
       train_accuracy:(map(.metrics.train.accuracy)|stats),
       validation_accuracy:(map(.metrics.validation.accuracy)|stats),
       active_fraction:(map(.structural.active_fraction)|stats),
       unused_fraction:(map(.structural.unused_fraction)|stats),
       tail32:(map(.structural.tail32)|stats),
       active_full_overlap:(map(.structural.active_full_overlap)|stats),
       coverage:([.[]|.coverage.validation[]?]|group_by([.family,.rank])|
         map({family:.[0].family,rank:.[0].rank,slots:length,coverage:(map(.coverage)|stats)}))})),
   errors:[.seeds[]|select(.status!="pass")]
  })
