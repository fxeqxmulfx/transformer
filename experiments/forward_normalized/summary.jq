# All runs and failures remain visible; never select a seed or radius.
def span: {min:min,max:max};
[.training.groups[]|. as $group|.runs[]|
  .+{arm:$group.arm,seed:$group.seed,clean_device_timing:$group.clean_device_timing}] as $runs|
{
  status,source_sha256,pilot_source_sha256,normalized_ptx_sha256,
  input_files:(.input_verification.files|length),inputs_unchanged:.input_verification.unchanged,
  legacy_preflight:{passed:.legacy_preflight.passed,
    directions:([.legacy_preflight.cases[].audit.directions[]?]|length),
    max_error:([.legacy_preflight.cases[].audit.directions[]?.absolute_error]|max),
    source_updates:([.legacy_preflight.cases[].audit.source_update_bitwise_match]|all)},
  preflight:{passed:.preflight.passed,cases:(.preflight.cases|length),edge_cases:(.preflight.edges.cases|length),
    directions:([.preflight.cases[].audit.directions[]?]|length),
    max_directional_error:([.preflight.cases[].audit.directions[]?.error]|max),
    max_parameter_error:([.preflight.cases[].audit.update.max_parameter_error,.preflight.edges.cases[].update.max_parameter_error]|max),
    E_B_source_match:([.preflight.cases[].audit.E_B_bitwise_source_match]|all),
    A_moments_unchanged:([.preflight.cases[].audit.A_moments_unchanged]|all)},
  completed_runs:.training.completed_runs,valid_completed_cohort:.training.valid_completed_cohort,
  cohort_wall_seconds:.training.cohort_wall_seconds,
  clean_device_groups:([.training.groups[]|select(.clean_device_timing)]|length),
  initial_metrics_match:([.training.groups[].initial_metrics_match]|all),
  scheduled_checks:([$runs[].history[]?]|length),scheduled_checks_passed:([$runs[].history[]?.state_valid]|all),
  chunks:([$runs[].chunks[]?]|length),
  zero_norm_updates:([$runs[]|select(.method=="local-normalized-a")|.normalization_stats[3]]|add),
  cost:(reduce ($runs[].cost//{}) as $row ({};reduce ($row|keys_unsorted[]) as $key (.;.[$key]=(.[$key]//0)+$row[$key]))),
  cases:[$runs[]|{arm,seed,method,status,completed_steps,clean_device_timing,
    endpoint_train:.history[-1].train,endpoint_validation:.history[-1].validation,
    setup_seconds,training_seconds,evaluation_seconds,active_seconds,training_seconds_per_update,
    allocated_payload_bytes,auxiliary_payload_bytes,norm_payload_bytes,normalization_stats,summary,checkpoint}],
  methods:($runs|group_by(.method)|map({method:.[0].method,runs:length,
    completed:([.[]|select(.status=="complete")]|length),
    endpoint_accuracy:([.[].history[-1].validation.accuracy|numbers]|span),
    endpoint_ce:([.[].history[-1].validation.loss|numbers]|span),
    first99_reached:([.[]|select(.summary.first99!=null)]|length),
    first995_reached:([.[]|select(.summary.first995!=null)]|length),
    joint_match_reached:([.[]|select(.summary.matched_ce_endpoint!=null)]|length),
    active_seconds:([.[].active_seconds|numbers]|span)})),
  paired:[.training.groups[]|. as $group|(.runs|map(select(.method=="local-pair"))|.[0]) as $local|
    (.runs|map(select(.method=="local-normalized-a"))|.[0]) as $candidate|
    {arm:$group.arm,seed:$group.seed,local_status:$local.status,candidate_status:$candidate.status,
      validation_correct_change:(if $candidate.status=="complete" and $local.status=="complete" then
        $candidate.history[-1].validation.correct-$local.history[-1].validation.correct else null end),
      validation_ce_change:(if $candidate.status=="complete" and $local.status=="complete" then
        $candidate.history[-1].validation.loss-$local.history[-1].validation.loss else null end)}]
}
