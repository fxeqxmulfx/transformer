# Frozen query-weighted comparison

The [protocol](PROTOCOL.md), frozen at `a564a02c`, retains both previous
probability/feature forecasts, both fixed metrics, all 144 anchors and 1800
horizons. Compare the uniform, per-query trace and transported covariance
coefficients on the unchanged recreated trajectories. No outcomes for this
new coefficient have been evaluated yet.

The constructor opens identities, named old checkpoints and sealed old-only
forecast payloads. It reconstructs the exact source batches and old GPU features,
retains their per-query weights, and seals its coefficient maxima before future
evidence is available. The verifier reconstructs those predictions, checks the
authenticated prior input ledgers, then checks executed features, real CE,
source GPU gradients and full saved potentials at every declared endpoint.
All numerical and conditional failures remain in the generated reports.

The core math has five independent tests in `../gram`. Three further harness
tests verify the complete inventory, the recovered source's real epoch order
and 96-example tail, and the explicit GPU-to-real execution conversions. These
tests never open a recreated checkpoint. Shared algebra tests also run.

Run from the repository root with the exact source checkouts under
`.lake/reproduction/` as documented in the parent README:

```sh
cargo test --offline --locked --release --manifest-path experiments/probability_tube/gram/Cargo.toml
cargo test --offline --locked --release --manifest-path experiments/probability_tube/gram_audit/Cargo.toml
cargo clippy --offline --locked --release --all-targets --manifest-path experiments/probability_tube/gram_audit/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/gram_audit/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/probability_tube/gram_audit/Cargo.toml
cp experiments/probability_tube/gram_audit/target/release/rotate90-complex-kernite .lake/probability-gram-audit-bin
.lake/probability-gram-audit-bin construct \
  experiments/probability_tube/runs/gram-forecast-01 \
  experiments/probability_tube/runs/forecast-01 \
  experiments/probability_tube/runs/sharp-forecast-01
.lake/probability-gram-audit-bin verify experiments/probability_tube/runs/gram-forecast-01
jq -f experiments/probability_tube/gram_audit/summary.jq \
  experiments/probability_tube/runs/gram-forecast-01/verification-01.json \
  > experiments/probability_tube/runs/gram-forecast-01/summary.json
```

Freeze the checked implementation before running `construct`. The cumulative
7200-second budget includes construction, all verification attempts and ledger
checks. Construction refuses overwrite; verification records separate attempts.
Each report includes the source ledger, trainer/PTX/executable fingerprints,
complete case inventory and elapsed time. Preserve generated outputs under
the root's ignored runs directory and logs under `.kanban/artifacts/84/`.
The earlier two audits remain immutable. Useful partial-CE horizons, if any,
still do not control the full adaptive optimizer or prove geometry discovery.
