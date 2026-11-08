# Preserve every predeclared direction/scale; do not select a per-state winner.
def counts: group_by(.)|map({status:.[0],count:length});
def span: {min:min,max:max};
[.cohort.cases[]|. as $case|.diagnostic.probes[]|
  .+{arm:$case.arm,seed:$case.seed,origin:$case.origin,case_passed:$case.passed}] as $rows|
{
  status,source_sha256,pilot_source_sha256,alignment_source_sha256,
  input_files:(.input_verification.files|length),inputs_unchanged:.input_verification.unchanged,
  cases:(.cohort.cases|length),passed_cases:([.cohort.cases[]|select(.passed)]|length),
  probes:($rows|length),seconds:.cohort.seconds,cost:.cohort.cost,
  finite_difference_checks:([.cohort.cases[].diagnostic.finite_differences[]]|length),
  max_finite_difference_error:([.cohort.cases[].diagnostic.finite_differences[].error]|max),
  coordinate_pairings:([.cohort.cases[].diagnostic.coordinate_pairing_checks[].groups[]]|length),
  max_pairing_error:([.cohort.cases[].diagnostic.coordinate_pairing_checks[].groups[].error]|max),
  max_norm_match_error:([.cohort.cases[].diagnostic.directions.norm_match_error]|max),
  baseline_replays:([.cohort.cases[].diagnostic.baseline_replay]|all),
  local_step_replays:([.cohort.cases[].diagnostic.source_step_replay]|all),
  adamw_A_replays:([.cohort.cases[].diagnostic.adamw_full_scale_replay]|all),
  ties_and_fixed_E_B:([$rows[].valid_ties_and_fixed_E_B]|all),
  zero_directions:([.cohort.cases[].diagnostic.directions|select(.zero_adamw_direction or .zero_gradient)]|length),
  results:($rows|group_by([.origin,.direction,.scale])|map({origin:.[0].origin,direction:.[0].direction,scale:.[0].scale,cases:length,
    local:([.[].local_objective.status]|counts),
    scopes:(["batch","train","validation"] as $names|[$names[] as $scope|["ce","pair"][] as $objective|
        {scope:$scope,objective:$objective,statuses:([.[].terminal[$scope][$objective].status]|counts),
          loss_change:([.[].terminal[$scope][$objective].change]|span),
          correct_change:([.[].terminal[$scope].correct_change]|span)}])})),
  rows:$rows
}
