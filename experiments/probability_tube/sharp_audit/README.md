# Matched probability-bound comparison

This continuation applies the proved PA1–PA2 pair-amplitude and Gram bounds
to the first audit's complete, frozen checkpoint inventory. The
[protocol](PROTOCOL.md) is fixed before reading any recreated probability
outcomes. It shares the original constructor's restricted file interface,
the guarded parameter caps, all numerical allowances and the future-state
verifier. Only the feature-radius calculation changes. Its final radius
intersects the original bound, so the comparison checks both usefulness and
that it never makes a forecast worse.

Run from the repository root after all three training cohorts finish. Freeze
both constructions before either verification. Retain all failed attempts;
the independent 1800-second budget includes construction, all verifications
and all comparisons. Compilation is excluded. The original audit keeps its
own unchanged budget.

```sh
export CARGO_TARGET_DIR="$PWD/.lake/probability-sharp-audit-target"
cargo test --offline --locked --release --manifest-path experiments/probability_tube/sharp_audit/Cargo.toml
cargo clippy --offline --locked --release --all-targets --manifest-path experiments/probability_tube/sharp_audit/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/sharp_audit/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/probability_tube/sharp_audit/Cargo.toml
cp "$CARGO_TARGET_DIR/release/rotate90-complex-kernite" .lake/probability-sharp-audit-bin
sharp_out="$PWD/experiments/probability_tube/runs/sharp-forecast-01"
original_out="$PWD/experiments/probability_tube/runs/forecast-01"
.lake/probability-sharp-audit-bin construct "$sharp_out" \
  experiments/probability_tube/runs/lr001/attempt-01 \
  experiments/probability_tube/runs/lr01/attempt-01 \
  experiments/probability_tube/runs/settle/attempt-01
# Run the original verifier after both constructions are sealed.
.lake/probability-sharp-audit-bin verify "$sharp_out"
.lake/probability-sharp-audit-bin compare "$original_out" "$sharp_out"
jq -f experiments/probability_tube/audit/summary.jq \
  "$sharp_out/verification-01.json" > "$sharp_out/summary.json"
jq -f experiments/probability_tube/sharp_audit/summary.jq \
  "$sharp_out/comparison-01.json" > "$sharp_out/comparison-summary.json"
```

Eight native tests include the batched/scalar refined bound, independent
source forward, guarded full-parameter updates and comparison failures for
changed metrics, caps, case order, outcomes or a single worsened query.
Source fingerprints include the shared verifier and the unchanged FT2 Gram
implementation. The comparison retains all 144 cases and 1800 endpoints,
with initialization reported separately and zero useful horizons retained.
Passing a partial-CE condition does not prove full optimizer nonincrease.
Generated evidence stays local; findings belong in the parent README.
