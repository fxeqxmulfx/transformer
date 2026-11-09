# Feature-weighted CE momentum control

Can feature-dependent adaptive scaling be absorbed in a checkpoint-fixed
potential, leaving less unexplained full-model forcing than the isotropic
reference? The [frozen protocol](PROTOCOL.md) fixes the theorem and all 144
comparisons before measuring them. This study is in progress; reliable
discovery and full-model future control remain open.

The standalone native crate builds with
`cargo build --offline --locked --release --manifest-path experiments/ce_metric/Cargo.toml`.
The executable first takes `NEW_OUTPUT ORIGINAL_LR001 ORIGINAL_LR01 ORIGINAL_SETTLE`
to freeze every metric. A separate invocation takes
`verify OUTPUT CE_LR001 CE_LR01 CE_SETTLE` and prints the complete JSON audit.
The original sources are the cohorts used by #206; the CE inputs are its
`experiments/ce_lyapunov/runs/{lr001-01,lr01-01,settle-01}` evidence directories.
Neither command runs training or changes #206's pinned native source.
