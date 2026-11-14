if .tasks != 35 or (.arms|length) != 2 or (.records|length) != 35
then error("incomplete context mixture comparison") else . end |
{tasks,process_failures,source_fingerprint,producer_commit,inventory_sha256,parent_summary_sha256,
 new_optimizer_updates,new_gradient_batches,new_local_row_updates,raw_label_identification_certified,constructor_accounting,prior_context_costs,
 prior_confidence_costs,prior_decoder_costs,prior_mixture_costs,prior_gated_costs,changes,
 prior_local_reference,previous_refused_attempt,
 conventional_transformer_endpoints:(.baselines.transformer_endpoints|length),
 depth_outcomes:(.baselines.depth_outcomes|length),
 arms:[.arms[]|{initializer,by_mode,full35_pass,all_split_diagnostic_pass}],
 fresh_endpoints:[.records[].outcomes[]?|{task,training_mode,primary_pass,all_splits_pass,
   raw_accuracy:(.metrics[.primary_split].native_token_accuracy // .metrics[.primary_split].token_accuracy),
   primary_accuracy:.metrics[.primary_split].sequence_accuracy,
   validation_ce:.metrics[.primary_split].loss,train_ce,model_info,metric_scalar_work_forecast}]}
