# Regenerate a compact local report; all fixed cases, ranks, splits and intervals remain.
def methods: .cases[].diagnostic.groups[].groups[].methods[];
def changes: .intervals[].diagnostic.groups[].groups[].methods[] | .train, .validation;
def signs:
  [.intervals[] | select(.diagnostic.training_ce_decreased)] as $d |
  [range(0;2) as $g | range(0;3) as $r | ["train","validation"][] as $s |
   {grouping:$d[0].diagnostic.groups[$g].grouping,
    rank:$d[0].diagnostic.groups[$g].aggregate[$r].rank, split:$s,
    ce_descent_intervals:($d|length),
    state_increases:([$d[].diagnostic.groups[$g].aggregate[$r][$s] | select(.state.increased)]|length),
    readout_increases:([$d[].diagnostic.groups[$g].aggregate[$r][$s] | select(.readout.increased)]|length),
    norm_product_increases:([$d[].diagnostic.groups[$g].aggregate[$r][$s] | select(.norm_product.increased)]|length)}];
{
  complete, seconds, source_sha256, protocol_sha256,
  audits: {
    cases:(.cases|length), intervals:(.intervals|length),
    unique_cases:([.cases[]|[.arm,.seed,.step]]|unique|length),
    unique_intervals:([.intervals[]|[.arm,.seed,.from_step,.to_step]]|unique|length),
    stored_counts_match:all(.cases[]; .stored_counts_match),
    gradient_bound_checks:([methods]|length),
    gradient_bounds_hold:all(methods; .bound_holds),
    sample_moments_hold:all(methods; .sample_moment_holds),
    max_sample_moment_error:([.cases[].diagnostic.max_sample_moment_error]|max),
    max_projector_defect:([.cases[].diagnostic.max_projector_defect]|max),
    max_gradient_bound_excess:([.cases[].diagnostic.max_gradient_bound_excess]|max),
    max_vector_identity_error:([changes|.max_vector_identity_error]|max),
    max_energy_identity_error:([changes|.state.energy_identity_error,.readout.energy_identity_error]|max),
    max_scalar_oracle_error:([.cases[].scalar_oracle_max_error]|max),
    max_normalization_error:([.cases[].normalization_max_error]|max),
    ce_descent_intervals:([.intervals[]|select(.diagnostic.training_ce_decreased)]|length)
  },
  signs_under_ce_descent:signs,
  cases:[.cases[]|{arm,seed,step,diagnostic:{
    train:.diagnostic.train,validation:.diagnostic.validation,
    training_state_mean_energy:.diagnostic.training_state_mean_energy,
    readout_gradient_energy:.diagnostic.readout_gradient_energy,
    groups:[.diagnostic.groups[]|{grouping,aggregate}]}}],
  intervals:[.intervals[]|{arm,seed,from_step,to_step,diagnostic:{
    training_ce_old:.diagnostic.training_ce_old,training_ce_new:.diagnostic.training_ce_new,
    training_ce_decreased:.diagnostic.training_ce_decreased,
    groups:[.diagnostic.groups[]|{grouping,aggregate}]}}]
}
