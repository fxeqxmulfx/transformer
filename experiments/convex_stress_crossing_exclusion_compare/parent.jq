# Keep complete reviewed task310 outcomes and every earlier reference/cost.
if .tasks != 35 or .choices_per_arm != 105 or .process_failures != 0
  or (.records|length) != 35 or (.arms|length) != 2
  or (.baselines.transformer_endpoints|length) != 35
  or (.baselines.depth_outcomes|length) != 336
then error("incomplete reviewed convex-exclusion parent") else . end |
{tasks, choices_per_arm, process_failures, success_rule, arms, constructor_accounting, baselines,
 prior_context_costs,prior_confidence_costs,prior_decoder_costs,prior_mixture_costs,prior_gated_costs,
 prior_local_reference,previous_refused_attempt,prior_region_reference,prior_region_costs,
 records: [.records[]|{task,status,eligible_models,selection_seal_sha256}],
 prior_exclusion_reference:{tasks,choices_per_arm,process_failures,source_fingerprint,producer_commit,
   inventory_sha256,parent_summary_sha256,success_rule,changes,changes_vs_guarded_reference,
   changes_vs_region_reference,primary_raw_totals,records,arms,raw_label_identification_certified,
   candidate_pool_closure_proved,target_convexity_proved,
   conventional_transformer_endpoints:(.baselines.transformer_endpoints|length),
   depth_outcomes:(.baselines.depth_outcomes|length)},
 prior_exclusion_costs:{original_run:"immutable task310 request directory and pinned completed manifest",
   new_optimizer_updates,new_gradient_batches,new_local_row_updates,
   model_forecasts:[.records[].outcomes[]?|{task,training_mode,model_info,metric_scalar_work_forecast}]}}
