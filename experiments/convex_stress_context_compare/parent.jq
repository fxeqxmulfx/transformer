# Keep every selected metric and all conventional/depth endpoints, without
# copying recursively duplicated intermediate training histories.
if .tasks != 35 or .choices_per_arm != 105 or .process_failures != 0
  or (.records|length) != 35 or (.arms|length) != 2
then error("incomplete reviewed parent") else . end |
.complete_parent.complete_parent.complete_parent.complete_parent as $base |
if ($base.transformer_endpoints|length) != 35 or ($base.depth_outcomes|length) != 336
then error("incomplete original baselines") else . end |
{tasks, choices_per_arm, process_failures, success_rule, arms, constructor_accounting,
 records: [.records[]|{task,status,run_exit,replay_exit}],
 baselines: {transformer_endpoints:$base.transformer_endpoints, depth_outcomes:$base.depth_outcomes}}
