# Closed cohort integrity. The original task #188 counts are fixed references.
(.status=="pass" and .failed==0 and .inputs.unchanged==true) and
(.inputs.files|length==74) and (.cases|length==24) and
([.cases[]|[.arm,.seed,.step]]|unique|length==24) and
([.cases[].seed]|unique==[7,11]) and
([.cases[].step]|unique==[0,5000,10000]) and
([.cases[].arm]|unique==["complex-diagonal","complex-product-plain","complex-product-radial","real-swiglu"]) and
all(.cases[];
 .status=="pass" and .post_budget_probe==(.step==10000) and
 .audit.passed==true and .audit.source_state_match==true and
 .audit.independent_source_checks.passed==true and
 .audit.binary.frozen_before_updated_forward==true and
 .audit.binary.rows==9409 and (.audit.binary.tracks|length==6) and
 .audit.checks.queries==9409 and .audit.checks.class_pairs_per_track==903264 and
 .audit.checks.layer_violations==0 and .audit.checks.feature_allowance_violations==0 and
 .audit.checks.logit_allowance_violations==0 and
 .audit.checks.false_certificates_in_reported_subsets==0 and
 (.audit.checks.gap_violations|length==6 and all(.[];.==0)) and
 (.audit.checks.outcome_violations|length==6 and all(.[];.==0)) and
 .audit.checks.subsets.train.count==4704 and .audit.checks.subsets.validation.count==4705 and
 .audit.checks.subsets.difficult_train.count==471 and .audit.checks.subsets.difficult_validation.count==471 and
 all(.audit.checks.subsets[].methods[];
   .false_certificates==0 and .margin_width.min>=0 and .loss_width.min>=0) and
 all(.audit.checks.stages[];
   (.actual_error.max|type=="number") and
   all(.methods[];.violations==0 and (.radius.min|type=="number") and .radius.min>=0)) and
 .audit.forecast_capacity_bytes>=.audit.forecast_payload_bytes and
 .audit.seconds.prepare>=.audit.seconds.gram+.audit.seconds.propagation+.audit.seconds.readout) and
([.cases[]|select(.step>0)|.audit.checks.subsets.validation.count]|add==75280) and
([.cases[]|select(.step>0)|.audit.checks.subsets.validation.actually_correct]|add==74952) and
([.cases[]|select(.step>0)|.audit.checks.subsets.validation.methods[0].certified_correct]|add==68040)
