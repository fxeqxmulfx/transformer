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

Producer `1978c688` completes the declared 32 model evaluations without an
update. Every intact raw-logit hash and metric matches its original `c620019a`
report. Replacing the new state by its training mean reduces novel and test
accuracy in every cell and raises CE on both held-out splits in every cell.
The complete model therefore benefits from its current input-dependent new
states beyond their average class-score contribution.

| Geometry | Source/mode | Intact novel | Zero-state novel | Mean-state novel | New block alone, novel |
| --- | --- | ---: | ---: | ---: | ---: |
| old | cautious Mini | 23.8281% | 16.9922% | 16.9922% | 13.6719% |
| old | cautious forward | 21.6797% | 18.7500% | 19.7266% | 10.1563% |
| old | refined Mini | 24.0234% | 17.7734% | 18.3594% | 9.1797% |
| old | refined forward | 25.3906% | 18.7500% | 19.3359% | 9.9609% |
| hilbert | cautious Mini | 24.4141% | 16.9922% | 18.1641% | 11.5234% |
| hilbert | cautious forward | 22.8516% | 18.7500% | 19.5313% | 8.3984% |
| hilbert | refined Mini | 23.2422% | 17.7734% | 18.3594% | 9.5703% |
| hilbert | refined forward | 23.8281% | 18.7500% | 18.5547% | 9.3750% |

The intact-minus-mean advantage is 1.9531–6.8359 percentage points on novel
examples and 4.1016–6.8359 points on test examples. Mean-state test accuracy is
17.1875–20.1172%, versus 22.2656–25.3906% intact. The new block alone reaches
only 8.3984–13.6719% novel and 9.9609–13.8672% test accuracy. Every original
success and 99% flag remains false; raw and exact accuracy agree on this
single-target task. These numbers add no newly trained stress-test successes.

The mean-state control is the average contribution of the current new block,
not a separately optimized class-bias baseline. Its failure does not rule out
every possible bias calibration, establish query-specific binding, or show
that wider states are necessary. The isolated block's limited accuracy also
does not distinguish insufficient representation from insufficient learning.
The experiment establishes useful dependence on the input under this ablation;
the next learning step still needs to improve the shared retrieval structure.

All training means use 2,048 original supervised occurrences. Their state norm
is 0.1520–0.2141 for the old geometry and 0.0744–0.1125 for Hilbert; all are
feasible without projection. Mean squared Euclidean deviation is positive in
every cell, 0.6821–0.7765 and 0.1573–0.1937 respectively. The families use
different state budgets, so these ranges alone do not rank their information
content. Maximum numerical norm excess across all ablations is 2.23e-16.

The three native controls, warning-free Clippy, formatting and shell checks
pass. A fresh review repeats every intact replay and all 32 ablation reports
byte for byte; both altered producer metadata and altered endpoint copies are
refused, and restored copies verify. All RAM/disk admission checks pass.
No original endpoint, score, label or transformer file changes. The frozen
diagnostic fingerprint is
`4ca7853c4323e23fcb84d7547a38ee252a5cf2870b73b41d0bde0ef0a215b5f6`;
source, binary, input pins and review evidence are at the locations above.
