# Preserve every task301 choice/metric and all model costs, including its
# complete task298 tradeoff and previous refused attempt. All artifacts are pinned.
if .tasks != 35 or .choices_per_arm != 105 or .process_failures != 0
  or (.records|length) != 35 or (.arms|length) != 2
  or (.baselines.transformer_endpoints|length) != 35
  or (.baselines.depth_outcomes|length) != 336
then error("incomplete reviewed local reference") else . end |
{tasks,choices_per_arm,process_failures,source_fingerprint,producer_commit,
 inventory_sha256,parent_summary_sha256,success_rule,changes,
 new_optimizer_updates,new_gradient_batches,new_local_row_updates,
 constructor_accounting,prior_context_costs,prior_confidence_costs,
 prior_decoder_costs,prior_mixture_costs,prior_gated_costs,prior_local_reference,
 previous_refused_attempt,raw_label_identification_certified,
 conventional_transformer_endpoints:(.baselines.transformer_endpoints|length),
 depth_outcomes:(.baselines.depth_outcomes|length),
 records,
 arms:[.arms[]|{initializer,by_mode,full35_pass,complete_primary_pass,
   all_split_diagnostic_pass,choices:[.choices[]|{task,training_mode,status,
     reference,parent_reference,primary_split,primary_pass,all_splits_pass,
     metrics,train_ce,model_sha256,metric_provenance}]}]}
