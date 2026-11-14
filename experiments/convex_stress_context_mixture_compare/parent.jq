# Keep the reviewed task290 selections, original metrics, costs and all baselines.
if .tasks != 35 or .choices_per_arm != 105 or .process_failures != 0
  or (.records|length) != 35 or (.arms|length) != 2
  or (.baselines.transformer_endpoints|length) != 35
  or (.baselines.depth_outcomes|length) != 336
then error("incomplete reviewed mixture parent") else . end |
{tasks, choices_per_arm, process_failures, success_rule, arms, constructor_accounting, baselines,
 prior_context_costs,prior_confidence_costs,
 records: [.records[]|{task,status,eligible_models,selection_seal_sha256}],
 prior_decoder_costs: {original_run:"immutable parent request directory; all resource files in the pinned completed manifest",
   model_forecasts:[.records[].outcomes[]?|{task,training_mode,model_info,metric_scalar_work_forecast}]}}
