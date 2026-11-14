# Original shared-role training evidence only. No held-out target is read.
def histogram: group_by(.) | map({value: .[0], count: length});
if .phase != "training_only" or .heldout_evaluation != false
   or .parent_comparison != false or .raw_label_identification != false
   or .target_readout_compatibility_proved != false
   or .shared_role_target_proved != false
   or .original_record_multiplicity_retained != true
   or (.tasks|length) != 35 or (.planned|length) != 23
   or (.records|map(.task)) != .planned
then error("not a complete original shared-role training summary") else . end
| [.records[].result.stages[]?.retained] as $stages
| [$stages[].endpoints[]? | select(.training != null)] as $endpoints
| {
    phase, source_fingerprint, binary_sha256, request, inventory_sha256,
    original_tasks: (.tasks|length), planned_cases: (.planned|length),
    accounting: .constructor_accounting,
    discovery_reasons: ([$stages[].discovery.reason] | histogram),
    restricted_dimensions: ([$stages[].discovery.restricted.dimension] | histogram),
    discovery_errors: ([$stages[].discovery_error] | histogram),
    head_errors: ([$stages[].head_error] | histogram),
    raw_record_ce: ([$endpoints[].audit.record_ce] | {minimum:min, maximum:max}),
    raw_record_errors: ([$endpoints[].audit.record_errors] | histogram),
    available: [.records[] | .task as $task | .result.stages[]?
      | select(any(.retained.endpoints[]?; .training != null))
      | {task:$task, prime,
         restricted_dimension:.retained.discovery.restricted.dimension,
         aligned_tokens:.retained.discovery.restricted.aligned_tokens,
         matrix_shape:.retained.discovery.restricted.matrix_shape,
         original_records:.retained.record_binding.original_records,
         states:.retained.initialization.bounds.states,
         unknown_axis_cells:.retained.initialization.unknown_axis_cells,
         unmatched_active_cells:.retained.initialization.unmatched_active_cells,
         modes:[.retained.endpoints[] | {available:(.training!=null), refusal,
             record_ce:.audit.record_ce, record_errors:.audit.record_errors,
             steps:.training.steps, gradient_batches:.training.work.gradient_batches}]}],
    failures:[.records[]|select(.status!="complete")|{task,status,run_exit,replay_exit}],
    heldout_evaluation:false, target_readout_compatibility_proved:false,
    shared_role_target_proved:false, raw_label_identification:false
  }
