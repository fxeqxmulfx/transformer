# Derived summaries stay local; this generator and the authored interpretation are tracked.
def maximum(xs): [xs] | max;
{
  status, failed, elapsed_seconds, source_sha256,
  cases: [.cases[] | {
    arm, seed, step, post_budget_probe, status, error,
    old: .audit.source_metrics_old, new: .audit.source_metrics_new,
    direction: .audit.direction,
    orbit32: (.audit.geometry[]? | select(.grouping == "orbit") |
      .aggregate[] | select(.rank == 32))
  }],
  checks: {
    cases: (.cases | length),
    bitwise_replays: ([.cases[] | select(.audit.bitwise_train_step_match == true)] | length),
    derivative_checks: ([.cases[].audit.gradient_validation.full_embedding_directions[]?] | length),
    max_derivative_error: maximum(.cases[].audit.gradient_validation.full_embedding_directions[]?.error),
    max_readout_derivative_error: maximum(.cases[].audit.gradient_validation.readout_max_error),
    max_full_training_readout_error: maximum(.cases[].audit.full_training_readout_max_error),
    inputs_unchanged: .input_verification.unchanged,
    input_files: (.input_verification.files | length),
    bases: ([.cases[].audit.geometry[]?.groups[].methods[]] | length),
    max_forcing_closure: maximum(.cases[].audit.geometry[]?.aggregate[].forcing.closure_scaled_error),
    max_energy_closure: maximum(.cases[].audit.geometry[]?.aggregate[].forcing.energy_scaled_error),
    max_interaction_closure: maximum(.cases[].audit.geometry[]?.aggregate[] |
      (.train.interaction.closure_scaled_error, .validation.interaction.closure_scaled_error)),
    max_margin_closure: maximum(.cases[].audit.geometry[]?.aggregate[] |
      (.train.margins.identity_scaled_error, .validation.margins.identity_scaled_error)),
    max_execution_allowance: maximum(.cases[].audit.geometry[]?.aggregate[] |
      (.train.margins.max_execution_allowance, .validation.margins.max_execution_allowance)),
    invalid_certificates: ([.cases[].audit.geometry[]?.aggregate[] |
      (.train.margins.invalid_certificates, .validation.margins.invalid_certificates)] | add)
  }
}
