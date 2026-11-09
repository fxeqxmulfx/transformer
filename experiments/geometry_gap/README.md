# Early settling and active-answer certificates

The user asks to close the discovery/persistence and certificate gaps after
the eight-seed .01 study. The [frozen protocol](PROTOCOL.md) separates two
changes: reduce the large learning rate before the late measurement window,
and fit the rank-limited readout basis to observed answer classes while still
bounding every rival class. All other training and numerical controls remain.

| Intervention | Changed | Fixed comparison |
| --- | --- | --- |
| Early settling | .01 to .0001 during updates 1000–4000, then hold until 8000 and cool to 1e-8 | Same eight seeds, 10000 updates and all-class certificates against .001 and .01 |
| Active-answer basis | Readout rows used to choose reconstruction directions | Same saved states, training folds, rank budgets and all 97 rival margins |

Implementation and outcomes are pending. Generated evidence is ignored and
preserved outside temporary worktrees. This experiment does not assert a
global training or convergence theorem.

From the repository root, reproduce the schedule arm with native tools and
a CUDA-enabled execution context. Each training run needs a fresh output:

```sh
export CARGO_TARGET_DIR="$PWD/.lake/geometry-gap-target"
cargo test --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle
cargo fmt --manifest-path experiments/class_geometry_wsd/Cargo.toml --check
cargo clippy --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --all-targets -- -D warnings
cargo build --offline --locked --release --manifest-path experiments/class_geometry_wsd/Cargo.toml --features settle --bins
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" preflight
"$CARGO_TARGET_DIR/release/rotate90-complex-kernite" \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  experiments/class_geometry_lr/runs/lr01-01/report.json \
  experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/verify" experiments/geometry_gap/runs/settle-01 \
  experiments/class_geometry_lr/runs/lr01-01/report.json
"$CARGO_TARGET_DIR/release/analyze" experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/compare" experiments/class_geometry_wsd/runs/audit-02 \
  experiments/geometry_gap/runs/settle-01
```

Once the new cohort is complete, compare the two certificate bases on all
744 saved states, then independently reread and verify the emitted bounds:

```sh
"$CARGO_TARGET_DIR/release/active-answers" experiments/geometry_gap/runs/active-01 \
  experiments/class_geometry_wsd/runs/audit-02 \
  experiments/class_geometry_lr/runs/lr01-01 \
  experiments/geometry_gap/runs/settle-01
"$CARGO_TARGET_DIR/release/active-answers" verify experiments/geometry_gap/runs/active-01
```

The original all-class margins must reproduce byte for byte before each
active-basis measurement. Bounds and fitting metadata are saved before query
outcomes are read for scoring. The second pass checks the emitted files
against direct FP64 and recorded GPU margins, without calling the basis
constructor. Its runtime is charged to the same 1800-second postprocessing
cap. The signed-transport bound and OG3 in
[`review/flat_spectrum.md`](../../review/flat_spectrum.md) already apply to
an arbitrary computed basis with its residual and defect terms retained.
