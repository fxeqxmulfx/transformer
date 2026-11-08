# Keep every case/rank/split/interval. Generated summaries remain local.
def adaptive:
  .cases[].diagnostic.groups[].groups[].methods[].adaptive | select(. != null);
def residuals: .cases[].diagnostic.groups[].groups[].methods[] | .train, .validation;
def drift: .intervals[].diagnostic.groups[].groups[].methods[] | .train, .validation;
def distribution:
  {min:min,max:max,mean:(add/length)};
def forcing_summary:
  . as $group |
  [16,32,64][] as $rank |
  [$group.groups[].methods[] | select(.rank==$rank) | .adaptive | select(.!=null)] as $a |
  {rank:$rank, groups:($a|length),
   energies: (if ($a|length)==0 then null else {
      actual:($a|map(.projected_saved_direction_energy)|distribution),
      scaled_gradient:($a|map(.scaled_projected_gradient_energy)|distribution),
      commutator:($a|map(.gradient_commutator_energy)|distribution),
      discrepancy:($a|map(.moment_discrepancy_energy)|distribution),
      cross:($a|map(.cross_terms)|distribution),
      three_term_bound:($a|map(.three_term_energy_bound)|distribution),
      objective_bound:($a|map(.objective_energy_bound)|distribution),
      dmax:($a|map(.dmax)|distribution),
      objective_to_actual:($a|map(.objective_energy_bound/.projected_saved_direction_energy)|distribution)
   } end)};
def signs:
  [.intervals[]|select(.diagnostic.training_ce_decreased)] as $d |
  [range(0;2) as $g | range(0;3) as $r | ["train","validation"][] as $s |
    {grouping:$d[0].diagnostic.groups[$g].grouping,
     rank:$d[0].diagnostic.groups[$g].aggregate[$r].rank,split:$s,
     ce_descent_intervals:($d|length),
     absolute_increases:([$d[].diagnostic.groups[$g].aggregate[$r][$s]|select(.absolute_increased)]|length),
     square_increases:([$d[].diagnostic.groups[$g].aggregate[$r][$s]|select(.square_increased)]|length)}];
{
  complete,seconds,source_sha256,protocol_sha256,
  audits:{
    cases:(.cases|length),intervals:(.intervals|length),
    unique_cases:([.cases[]|[.arm,.seed,.step]]|unique|length),
    unique_intervals:([.intervals[]|[.arm,.seed,.from_step,.to_step]]|unique|length),
    initial_unavailable:([.cases[]|select(.diagnostic.adaptive_status=="not_applicable_initial_step")]|length),
    adaptive_checks:([adaptive]|length),
    stored_counts_match:all(.cases[];.stored_counts_match),
    all_adaptive_pass:all(adaptive;.identity_holds and .energy_identity_holds and
      .three_term_bound_holds and .loss_term_bound_holds and .objective_bound_holds),
    max_adaptive_identity_error:([adaptive|.identity_max_error]|max),
    max_adaptive_energy_identity_error:([adaptive|.energy_identity_error]|max),
    max_projection_defect:([residuals|.max_projection_defect]|max),
    max_cauchy_excess:([residuals|.max_cauchy_excess]|max),
    max_interaction_change_error:([drift|.max_identity_error]|max),
    max_interaction_change_bound_excess:([drift|.max_change_bound_excess]|max),
    max_source_oracle_error:([.cases[].scalar_oracle_max_error]|max),
    max_normalization_error:([.cases[].normalization_max_error]|max),
    ce_descent_intervals:([.intervals[]|select(.diagnostic.training_ce_decreased)]|length)
  },
  signs_under_ce_descent:signs,
  cases:[.cases[]|{arm,seed,step,diagnostic:{
    train:.diagnostic.train,validation:.diagnostic.validation,
    adaptive_status:.diagnostic.adaptive_status,
    readout_gradient_energy:.diagnostic.readout_gradient_energy,
    groups:[.diagnostic.groups[]|{grouping,aggregate,forcing:[forcing_summary]}]}}],
  intervals:[.intervals[]|{arm,seed,from_step,to_step,diagnostic:{
    training_ce_old:.diagnostic.training_ce_old,training_ce_new:.diagnostic.training_ce_new,
    training_ce_decreased:.diagnostic.training_ce_decreased,
    groups:[.diagnostic.groups[]|{grouping,aggregate}]}}]
}
