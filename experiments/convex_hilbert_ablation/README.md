# Does the new block use input-dependent information?

The `c620019a` pilot improves original novel/test accuracy with either norm,
but remains below 26%. Before attributing the gains to useful shared content,
this diagnostic checks whether a constant class-score adjustment explains them.
It is declared after those results, covers all eight endpoints and performs
no training, tuning or endpoint selection.

| Diagnostic | Change to the saved model |
| --- | --- |
| intact | None; require every original raw-logit hash and metric to match |
| zero-state | Zero the new block's initial state, retaining its mass and all parameters |
| training-mean-state | Replace the new state by its input-only training mean and every new transition by identity |
| new-block-only | Put parent weight in its structural zero component, retaining all stored components and the new block's original mass |

The mean uses actual complete complex states on every original supervised
training prefix, including duplicates and masks. It never reads target values
and never uses held-out prefixes. Every ablation preserves all original labels,
classes, example lists and scoring rules. Zero-state retains seven eighths of
the parent logits; it measures removal of the new contribution at the same
allocation. Mean-state retains a constant per-class contribution through the
learned readouts. New-block-only uses mass two, not a rescaled standalone model.
These are diagnostics, not newly trained endpoints or additions to a stress
success tally.

The actual constructors check the existing CM107/108 complete geometry. Zero
and identity transitions stay in the bounded family; the computed mean must
pass its state-norm check. This makes no new convergence, identification or
floating-rounding claim. Three native controls distinguish actual input
dependence from constant bias, preserve imaginary coordinates and duplicate
weights, verify that labels do not affect the mean, and keep every stored old
component in the standalone ablation.

`inputs.json` pins metadata, receipts, endpoints and saved scores from all eight
original runs. The diagnostic checks the producing fingerprint and binary,
then reuses the original model and original metrics implementation. Its intact
results must reproduce the saved reports exactly before any ablation is
interpreted. The complete diagnostic must repeat byte for byte in a fresh
directory; changed producer metadata and endpoint copies must be refused.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/hilbert-content
export CARGO_BUILD_JOBS=2 OPENBLAS_NUM_THREADS=1
cargo test --offline --locked --manifest-path experiments/convex_hilbert_ablation/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_hilbert_ablation/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_hilbert_ablation/Cargo.toml --check
cargo build --release --offline --locked --manifest-path experiments/convex_hilbert_ablation/Cargo.toml
```

Freeze the producing commit archive, binary, protocol, input pins, plan,
wrappers and SHA256 manifest before running this declared diagnostic.

```sh
bash experiments/convex_hilbert_ablation/run.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/hilbert-ablation-v1 \
 /run/media/misha/storage/transformer-runs/hilbert-content/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/hilbert-ablation-v1
bash experiments/convex_hilbert_ablation/review.sh \
 /home/misha/lean_projects/transformer /run/media/misha/storage \
 /run/media/misha/storage/transformer-runs/freezes/hilbert-ablation-v1 \
 /run/media/misha/storage/transformer-runs/hilbert-content/original-v1 \
 /run/media/misha/storage/transformer-runs/diagnostics/hilbert-ablation-v1 \
 /run/media/misha/storage/transformer-runs/reviews/task398-v1
```

Use one native worker under a 32 GiB address-space cap, the 110 GB aggregate
RAM cap and both disk guards. Stop on low disk space. Preserve every original
artifact and the transformer baseline; keep generated reports outside the
worktree. Generalization and the requested 99% target remain unresolved.
