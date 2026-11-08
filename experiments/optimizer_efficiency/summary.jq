def span: {min:min,max:max};
def ratio($a;$b):
  if ($a|type)=="number" and ($b|type)=="number" and $b>0
  then $a/$b else null end;
def difference($a;$b):
  if ($a|type)=="number" and ($b|type)=="number"
  then $a-$b else null end;
(.benchmark.cases // []) as $cases |
(.training.groups // []) as $groups |
{
  status,error,source_sha256,candidate_ptx_sha256,
  inputs_unchanged:.inputs.unchanged,input_files:(.inputs.files // []|length),
  preflight:{passed:.preflight.passed,
    saved_states:(.preflight.cases // []|length),
    passed_states:([.preflight.cases[]?|select(.passed)]|length),
    warmup_checks:([.preflight.cases[]?.audit.warmup[]?]|length),
    passed_warmup_checks:([.preflight.cases[]?.audit.warmup[]?|select(.passed)]|length),
    edges:.preflight.edges,
    failures:[.preflight.cases[]?|select(.passed!=true)]},
  benchmark:{passed:.benchmark.passed,seconds:.benchmark.seconds,
    cases:($cases|length),passed_cases:([$cases[]|select(.passed)]|length),
    all_samples:([$cases[].samples[]?]|length),
    measured_samples:([$cases[].samples[]?|select(.warmup==false)]|length),
    paired_state_checks:([$cases[].benchmark.paired_state_checks // 0]|add),
    exclusive_cases:([$cases[]|select(.gpu_before.clean_ownership and .gpu_after.clean_ownership)]|length),
    empty_event_brackets:.benchmark.empty_event_brackets,
    rows:[$cases[]|{arm,seed,step,passed,error,summary:.benchmark.summary,memory:.benchmark.memory}],
    arms:($cases|group_by(.arm)|map({arm:.[0].arm,cases:length,
      passed_cases:([.[]|select(.passed)]|length),
      update_wall_ratio:([.[].benchmark.summary.update.wall_seconds.paired_speed_ratio.median // empty]|span),
      complete_wall_ratio:([.[].benchmark.summary.complete.wall_seconds.paired_speed_ratio.median // empty]|span),
      update_event_ratio:([.[].benchmark.summary.update.gpu_seconds.paired_speed_ratio.median // empty]|span),
      complete_event_ratio:([.[].benchmark.summary.complete.gpu_seconds.paired_speed_ratio.median // empty]|span),
      memory:([.[].benchmark.memory // empty]|unique)}))},
  training:{completed_runs:.training.completed_runs,
    valid_completed_cohort:.training.valid_completed_cohort,seconds:.training.seconds,
    observations:([$groups[].runs[].history[]?]|length),
    paired_checks:([$groups[].matches[]?]|length),
    passed_paired_checks:([$groups[].matches[]?|select(.passed)]|length),
    exclusive_groups:([$groups[]|select(.clean_device_timing)]|length),
    runs:[$groups[]|. as $g|.runs[]|{arm:$g.arm,seed:$g.seed,method,status,error,
      completed_steps,setup_seconds,training_seconds,evaluation_seconds,active_seconds,
      allocated_payload_bytes,moment_bytes,transient_source_moments_during_setup,cost,summary,checkpoint}],
    pairs:[$groups[]|. as $g|
      (.runs|map(select(.method=="source-adamw"))|first) as $s|
      (.runs|map(select(.method=="compact-adamw"))|first) as $c|
      {arm,seed,source_status:$s.status,candidate_status:$c.status,
        all_observations_match:((.matches|length)==41 and all(.matches[];.passed==true)),
        endpoint:$s.summary.endpoint.validation,
        first99_step:$s.summary.first99.step,first995_step:$s.summary.first995.step,
        matched_endpoint_step:$s.summary.matched_ce_endpoint.step,
        training_speed_ratio:ratio($s.training_seconds;$c.training_seconds),
        active_speed_ratio:ratio($s.active_seconds;$c.active_seconds),
        matched_quality_active_ratio:ratio($s.summary.matched_ce_endpoint.active_seconds;$c.summary.matched_ce_endpoint.active_seconds),
        first99_active_ratio:ratio($s.summary.first99.active_seconds;$c.summary.first99.active_seconds),
        first995_active_ratio:ratio($s.summary.first995.active_seconds;$c.summary.first995.active_seconds),
        allocated_bytes_saved:difference($s.allocated_payload_bytes;$c.allocated_payload_bytes),
        moments_fraction_saved:difference(1;ratio($c.moment_bytes;$s.moment_bytes))}]
  },
  ownership:{
    benchmark_queries:([$cases[]|.gpu_before,.gpu_after]|length),
    training_queries:([$groups[].gpu_snapshots[]?]|length),
    snapshots:([$cases[]|.gpu_before,.gpu_after]+[$groups[].gpu_snapshots[]?])}
}
