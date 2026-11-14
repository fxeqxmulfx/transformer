if .tasks != 35 or (.arms|length) != 2 or (.records|length) != 35
then error("incomplete exclusion comparison") else . end |
{tasks,process_failures,source_fingerprint,producer_commit,inventory_sha256,parent_summary_sha256,
 new_optimizer_updates,new_gradient_batches,new_local_row_updates,raw_label_identification_certified,
 candidate_pool_closure_proved,target_convexity_proved,primary_raw_totals,
 constructor_accounting,prior_context_costs,prior_confidence_costs,prior_decoder_costs,
 prior_mixture_costs,prior_gated_costs,prior_region_costs,prior_region_reference,changes,
 prior_exclusion_costs,prior_exclusion_reference,changes_vs_exclusion_reference,
 prior_local_reference,previous_refused_attempt,changes_vs_guarded_reference,changes_vs_region_reference,
 conventional_transformer_endpoints:(.baselines.transformer_endpoints|length),
 depth_outcomes:(.baselines.depth_outcomes|length),
 arms:[.arms[]|{initializer,by_mode,full35_pass,all_split_diagnostic_pass}],
 fresh_endpoints:[.records[].outcomes[]?|{task,training_mode,primary_pass,all_splits_pass,
   raw_accuracy:(.metrics[.primary_split].native_token_accuracy // .metrics[.primary_split].token_accuracy),
   primary_accuracy:.metrics[.primary_split].sequence_accuracy,
   validation_ce:.metrics[.primary_split].loss,train_ce,model_info,metric_scalar_work_forecast,
   raw_transition_audit,candidate_pool_closure_proved,target_convexity_verified}]}
