. as $report
| [.cases[] | select(has("audit"))] as $checked
| {
    status, failed, elapsed_seconds, source_sha256, loss_kernel_ptx_sha256,
    cases: (.cases | length),
    errors: [.cases[] | select(.status != "pass") | {arm, seed, step, status, error}],
    input_files: ((.input_verification.files // {}) | length),
    inputs_unchanged: .input_verification.unchanged,
    kernel_edges,
    checks: {
      examples: ([$checked[].audit.kernel.rows] | add // 0),
      gradient_coordinates: ([$checked[].audit.kernel.gradient_coordinates] | add // 0),
      violations: ([$checked[].audit.kernel.violations] | add // 0),
      max_loss_error: ([$checked[].audit.kernel.max_loss_absolute_error] | max),
      max_gradient_error: ([$checked[].audit.kernel.max_gradient_absolute_error] | max),
      max_readout_gradient_error: ([$checked[].audit.partial_readout.max_gradient_error] | max)
    },
    local_step: {
      frozen_decreases: ([$checked[] | select(.audit.partial_readout.frozen_features_change < 0)] | length),
      tied_decreases: ([$checked[] | select(.audit.partial_readout.actual_tied_change < 0)] | length),
      tied_increases: ([$checked[] | select(.audit.partial_readout.actual_tied_change > 0)] | length),
      tied_unchanged: ([$checked[] | select(.audit.partial_readout.actual_tied_change == 0)] | length)
    },
    outcomes: [$report.cases[] | {arm, seed, step, status, error,
      validation: .audit.splits.validation,
      partial_readout: .audit.partial_readout}]
  }
