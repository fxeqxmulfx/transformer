# Author conclusions in README; keep this regenerated report local.
def span: {min:min,max:max};
def classes: group_by(.)|map({classification:.[0],count:length});
{
  status,source_sha256,pilot_source_sha256,
  input_files:(.input_verification.files|length),inputs_unchanged:.input_verification.unchanged,
  cases:(.cohort.cases|length),passed_cases:([.cohort.cases[]|select(.passed)]|length),
  cost:.cohort.cost,seconds:.cohort.seconds,
  finite_difference_checks:([.cohort.cases[].diagnostic.finite_differences[]]|length),
  max_finite_difference_error:([.cohort.cases[].diagnostic.finite_differences[].error]|max),
  coordinate_pairing_checks:([.cohort.cases[].diagnostic.coordinate_pairing_checks[].groups[]]|length),
  max_pairing_error:([.cohort.cases[].diagnostic.coordinate_pairing_checks[].groups[].error]|max),
  source_replays:([.cohort.cases[].diagnostic.source_replay]|all),
  local_bitwise_matches:([.cohort.cases[].diagnostic.local_update_bitwise_match]|all),
  max_input_E_reconstruction_error:([.cohort.cases[].diagnostic.signal_checks.input_E_reconstruction_max_error]|max),
  states:[.cohort.cases[]|.diagnostic as $d|
    {arm,seed,origin,passed,before:$d.before,
      local_gradient_alignment:$d.gradient_alignment["local-pair"],
      local_update_alignment:$d.updates[3].alignment,
      local_loss_change:$d.updates[3].change,local_block_probes:$d.local_block_probes,
      embedding_ablations:[$d.updates[]|select(.rule=="local-pair" or .rule=="local-no-vocabulary" or .rule=="local-no-readout")|
        {rule,change,alignment:.alignment.validation.ce.E}],
      local_cross_group_interaction:$d.local_cross_group_interaction}],
  by_origin:([.cohort.cases[]]|group_by(.origin)|map({origin:.[0].origin,cases:length,
    local_scopes:(["batch","train","validation"] as $scopes|
      [$scopes[] as $scope|["ce","pair"][] as $objective|["E","A","B","all"][] as $group|
        {scope:$scope,objective:$objective,group:$group,
          gradient_classes:([.[].diagnostic.gradient_alignment["local-pair"][$scope][$objective][$group].classification]|classes),
          update_classes:([.[].diagnostic.updates[3].alignment[$scope][$objective][$group].classification]|classes),
          update_cosine:([.[].diagnostic.updates[3].alignment[$scope][$objective][$group].cosine]|span),
          finite_loss_change:(if $group=="all" then [.[].diagnostic.updates[3].change[$scope][$objective]]
            else [.[].diagnostic.local_block_probes[]|select(.group==$group)|.change[$scope][$objective]] end|span)}])}))
}
