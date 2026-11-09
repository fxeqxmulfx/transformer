# AdamW motion and support-complete certificates

The question is whether the main-model AdamW recurrence supplies a useful
finite-horizon persistence guarantee, and whether restoring the direction
between active and unused class means tightens prediction certificates.
The [fixed protocol](PROTOCOL.md) retains every state of all three complete
eight-seed cohorts. No new training or outcome-based selection is planned.

| Part | Changed or derived | Controls |
| --- | --- | --- |
| Optimizer proof | Exact .9/.95 moment invariant and finite learning-rate budget | Arbitrary gradients; explicit floating-point remainder |
| Persistence audit | Analytic bound and actual motion during updates 4001–8000 | All eight seeds in all three cohorts |
| Certificate basis | Active rows centered at the unused-class mean | Same folds, all rivals and numerical allowances; ranks 32 and 33 separate |
| Certificate envelope | Maximum of three valid lower bounds | Extra basis preparation explicitly charged |

Work in progress. No discovery or convergence theorem is claimed.

## Reproduce the certificate audit

Use a fresh output directory from the repository root. Existing cohorts
remain immutable. The independent verifier also checks that the envelope
is exactly the maximum of its three constituent lower bounds.

```sh
export CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --bin geometry-dynamics
"$CARGO_TARGET_DIR/release/geometry-dynamics" experiments/geometry_dynamics/runs/support-01 \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01 \
  experiments/geometry_gap/runs/active-01
"$CARGO_TARGET_DIR/release/geometry-dynamics" verify experiments/geometry_dynamics/runs/support-01
```
