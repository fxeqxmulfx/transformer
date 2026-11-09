# Native probability forecast audit

This implements the parent experiment's checkpoint-only construction and
separate future-data verification. The [implementation protocol](PROTOCOL.md)
fixes the conservative contrast envelope and numerical allowances. It uses
the exact recovered trainer and the source paths documented by
[`reproduce`](../reproduce/README.md). One executable handles all three
cohorts; it does not train or update the source checkpoints.

The constructor reads only `identity.json` and the requested anchor state.
Its input ledger makes those reads reviewable. It executes that old state on
all 4704 training inputs, caches intermediate norms with FP64 batch products,
and applies the proved encoder/Adam formulas. At each horizon it takes the
largest parameter caps over all prefixes. The output retains the per-input
probability ceilings, all parameter and feature caps, both fixed metrics,
and the q=1 and invalid fixed-q controls. It records the ordinary-energy
conditioning cost separately from the partial-CE coefficient.

The verifier checks the construction seal, reconstructs every payload from
old-only inputs, then opens later states. It checks all 1800 endpoints from
144 anchors, including 144 zero-horizon comparisons. It retains full
potential changes and the original signed next-step accounting. The native
checks do not establish the numerical allowances for every intervening CUDA
operation; a whole-window claim remains conditional on those allowances.

Build and freeze this implementation before measuring the probability
outcomes. Run from the repository root after the three cohort attempts
finish. Use a fresh output directory and retain failed attempts and their
time; do not restart the 1800-second budget after a failure.

```sh
export CARGO_TARGET_DIR="$PWD/.lake/probability-audit-target"
cargo test --offline --locked --release --manifest-path experiments/probability_tube/audit/Cargo.toml
cargo clippy --offline --locked --release --all-targets --manifest-path experiments/probability_tube/audit/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/audit/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/probability_tube/audit/Cargo.toml
cp "$CARGO_TARGET_DIR/release/rotate90-complex-kernite" .lake/probability-audit-bin
audit_out="$PWD/experiments/probability_tube/runs/forecast-01"
.lake/probability-audit-bin construct "$audit_out" \
  experiments/probability_tube/runs/lr001/attempt-01 \
  experiments/probability_tube/runs/lr01/attempt-01 \
  experiments/probability_tube/runs/settle/attempt-01
.lake/probability-audit-bin verify "$audit_out"
jq -f experiments/probability_tube/audit/summary.jq \
  "$audit_out/verification-01.json" > "$audit_out/summary.json"
```

The report preserves every horizon on missing states, caps or failures.
Verification attempts are separate numbered files and charge all earlier
verification durations plus construction time. The source fingerprint
covers the formulas, shared native modules, protocols and Cargo lockfiles.
Generated reports and logs stay ignored; authored findings belong in the
parent experiment README.

Seven native tests check the source forward oracle, batched versus scalar
encoder bounds, guarded full-parameter FP32 updates, declared endpoint
inventory and the independent CE/metric identities. The probability library
also has its own eight tests. These are numerical and implementation checks;
the corresponding Lean theorems concern real arithmetic.
