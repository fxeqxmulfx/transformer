# Aggregate the full fixed cohort without choosing seeds after observation.
if .status!="pass" or .failed!=0 or (.cases|length)!=24
then error("complete motion audit required") else . end |
{status,seconds,source_sha256,
 updates:([.cases[].updates]|add),coordinate_checks:([.cases[].coordinate_step_checks]|add),
 max_relative_direction_excess:([.cases[].max_relative_direction_excess]|max),
 cohorts:(.cases|group_by(.cohort)|map({cohort:.[0].cohort,seeds:length,
   rate_sum:([.[].hold.learning_rate_sum]|[min,max]),
   analytic_frobenius_budget:([.[].hold.analytic_frobenius_motion_budget]|[min,max]),
   actual_max_displacement:([.[].hold.actual_max_frobenius_displacement]|[min,max]),
   actual_path_length:([.[].hold.actual_frobenius_path_length]|[min,max]),
   initial_active_norm:([.[].hold.initial_active_norm]|[min,max]),
   nonzero_hold_steps:([.[].hold.moving_updates]|add),
   bound_below_active_norm:(map(select(.hold.budget_below_initial_active_norm))|length),
   minimum_active_share:([.[].hold.minimum_active_share]|[min,max])}))}
