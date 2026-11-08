# Regenerate all-case evidence; authored interpretation belongs in README.md.
def range: {min:min,max:max};
{
  status,source_sha256,pair_kernel_ptx_sha256,auxiliary_norm_ptx_sha256,
  input_files:(.input_verification.files|length),inputs_unchanged:.input_verification.unchanged,
  preflight:{passed:.preflight.passed,cases:(.preflight.cases|length),
    directional_checks:([.preflight.cases[].audit.directions[]]|length),
    maximum_directional_error:([.preflight.cases[].audit.directions[].absolute_error]|max),
    update_checks:([.preflight.cases[].audit.gradient_and_update_checks[]]|length),
    ce_bitwise_matches:([.preflight.cases[].audit.source_update_bitwise_match]|all)},
  completed_runs:.training.completed_runs,
  valid_completed_cohort:.training.valid_completed_cohort,
  cohort_wall_seconds:.training.cohort_wall_seconds,
  clean_device_groups:([.training.groups[]|select(.clean_device_timing)]|length),
  source_replay_checks:([.training.groups[].runs[]|select(.method=="ce-bp")|.history[]]|length),
  source_replay_matches:([.training.groups[].runs[]|select(.method=="ce-bp")|.history[].source_replay_match]|all),
  cases:[.training.groups[]|.arm as $arm|.seed as $seed|.clean_device_timing as $clean|
    .runs[]|{arm:$arm,seed:$seed,method,status,completed_steps,clean_device_timing:$clean,
      endpoint_train:.history[-1].train,endpoint_validation:.history[-1].validation,
      setup_seconds,training_seconds,evaluation_seconds,active_seconds,training_seconds_per_update,
      allocated_payload_bytes,auxiliary_payload_bytes,summary,checkpoint}],
  methods:([.training.groups[].runs[]]|group_by(.method)|map({method:.[0].method,runs:length,
    endpoint_accuracy:([.[].history[-1].validation.accuracy]|range),
    endpoint_ce:([.[].history[-1].validation.loss]|range),
    first99_reached:([.[]|select(.summary.first99!=null)]|length),
    first995_reached:([.[]|select(.summary.first995!=null)]|length),
    joint_match_reached:([.[]|select(.summary.matched_ce_endpoint!=null)]|length),
    active_seconds:([.[].active_seconds]|range)}))
}
