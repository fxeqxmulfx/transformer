# Read small per-case metrics, not the large inherited reference tree.
# Usage: jq -s -f analyze.jq RUN/*/metrics.json
[.[] | .results.endpoints[]] as $all |
[
  ["Coordinates", "Categorical"][] as $prior |
  ["Forward", "Full", "Mini"][] as $mode |
  [$all[] | select(.prior == $prior and .mode == $mode)] as $attempted |
  [$attempted[] | select(.error == null and .measurement != null)] as $models |
  [$models[] | .measurement.splits | to_entries[] |
    select(.key == "test" or (.key | startswith("test/"))) | .value] as $tests |
  [$models[] | .measurement.splits | to_entries[] |
    select((.key | startswith("train")) | not) | .value] as $heldout |
  ($tests | map(.metrics.targets) | add) as $targets |
  {
    prior: $prior, mode: $mode,
    attempted_cases: ($attempted | length),
    measured_cases: ($models | length),
    refusals: ($attempted | map(select(.error != null)) | length),
    all_heldout99_cases: ($models | map(select(.all_heldout99)) | length),
    minimum_heldout_raw: ($heldout | map(.metrics.token_accuracy) | min),
    minimum_heldout_exact: ($heldout | map(.metrics.sequence_accuracy) | min),
    test_targets: $targets,
    test_correct: ($tests | map(.metrics.correct) | add),
    test_examples: ($tests | map(.metrics.examples) | add),
    test_exact: ($tests | map(.metrics.exact) | add),
    weighted_test_ce: (($tests | map(.metrics.teacher_forced.loss * .metrics.targets) | add) / $targets),
    unseen_test_targets: ($tests | map(.unseen_label_targets) | add),
    unseen_test_correct: ($tests | map(.unseen_label_correct) | add),
    uncovered_test_queries: ($tests | map(.uncovered_scored_queries) | add),
    maximum_independent_product_residual: ($models | map(.measurement.splits[] | .independent_complex_max_residual) | max),
    maximum_projection_violation: ($models | map(.training.maximum_rechecked_projection_violation) | max),
    gradient_batches: ($models | map(.training.work.gradient_batches) | add),
    gradient_records: ($models | map(.training.work.gradient_records) | add),
    bitwise_parameter_updates: ($models | map(.training.work.effective_updates) | add),
    largest_actual_parameter_step: ($models | map(.training.largest_actual_parameter_step) | max),
    largest_complete_parameter_charge: ($models | map(.training.complete_charge) | max),
    largest_actual_column_charge: ($models | map(.training.actual_max_column_charge) | max)
  }
]
