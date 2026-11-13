# Read-only accounting of the complete frozen task252 training cohort.
def histogram:
  group_by(.) | map({value: .[0], count: length});
def stages: .records[].result.stages[].retained;
def endpoints: stages | .endpoints[];
{
  source_fingerprint,
  binary_sha256,
  inventory_sha256,
  scope: {
    original_tasks: (.tasks | length),
    planned_cases: (.planned | length),
    closed_cases: ([.records[] | select(.status == "complete" and
      .run_exit == 0 and .replay_exit == 0)] | length),
    heldout_evaluation,
    parent_comparison,
    raw_label_identification,
    target_readout_compatibility_proved
  },
  constructor_accounting,
  discovery_reasons: ([stages | .discovery.reason] | histogram),
  complete_kernel_dimensions: ([stages | .discovery |
    select(.basis_audit.complete == true) | .dimension] | histogram),
  initialization_refusals: ([stages | .initialization_refusal |
    select(. != null)] | histogram),
  endpoint_refusals: ([endpoints | .refusal | select(. != null)] | histogram),
  discovery_or_head_errors: ([stages |
    select(.discovery_error != null or .head_error != null)] | length),
  observed_errors: ([endpoints | select(.audit != null) | .audit.train_errors] |
    {endpoints: length, maximum: max, total: add}),
  actual_observed_ce: ([endpoints | select(.audit != null) | .audit.train_ce] |
    {minimum: min, maximum: max}),
  per_case: [.records[] | {
    task,
    available_full_mini_forward: [range(0; 3) as $m |
      [.result.stages[].retained.endpoints[$m] | select(.training != null)] | length],
    forward_primes: [.result.stages[] |
      select(.retained.endpoints[2].training != null) | .prime]
  }]
}
