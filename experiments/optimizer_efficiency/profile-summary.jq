def span: {min:min,max:max};
{
  status,source_sha256,inputs_unchanged:.inputs.unchanged,input_files:(.inputs.files|length),
  cases:(.profile.cases|length),passed_cases:([.profile.cases[]|select(.passed)]|length),seconds:.profile.seconds,
  measured_samples:([.profile.cases[].profile.samples[]?|select(.warmup==false)]|length),
  all_samples:([.profile.cases[].profile.samples[]?]|length),
  exclusive_cases:([.profile.cases[]|select(.gpu_before.clean_ownership and .gpu_after.clean_ownership)]|length),
  rows:[.profile.cases[]|{arm,seed,step,passed,error,summary:.profile.summary,
    allocated_payload_bytes:.profile.allocated_payload_bytes,moment_bytes:.profile.moment_bytes}],
  arms:([.profile.cases[]|select(.passed)]|group_by(.arm)|map({arm:.[0].arm,cases:length,
    update_wall_us:([.[].profile.summary.update.wall_seconds.median*1e6]|span),
    complete_wall_us:([.[].profile.summary.complete.wall_seconds.median*1e6]|span),
    update_event_us:([.[].profile.summary.update.gpu_seconds.median*1e6]|span),
    complete_event_us:([.[].profile.summary.complete.gpu_seconds.median*1e6]|span),
    wall_ratio:([.[].profile.summary.update_over_complete.wall_seconds]|span),
    event_ratio:([.[].profile.summary.update_over_complete.gpu_seconds]|span),
    moment_bytes:([.[].profile.moment_bytes]|unique)})),
  empty_event_brackets:.profile.empty_event_brackets
}
