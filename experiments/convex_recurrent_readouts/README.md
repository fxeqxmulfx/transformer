# Revisit complete readouts on learned recurrent states

Can refitting every learned component's class readouts across the training
stream improve original associative recall? The frozen cautious and refined
rotary models fit only 8.5–9.6% of training answers. Their complete states remain
nonconstant with large norms, while each old readout was fitted only on its
component's proposal batch. This experiment tests that optimizer limitation.

CM100 at `16276e1c` proves 22 statements before this implementation: complete
complex row-domain convexity, actual state preservation, linear complete logits,
ordinary full-class and indexed minibatch CE convexity, concrete radial row
feasibility, accepted interpolation and full-family signed CE-query feasibility.
The source is `review/flat_spectrum.md`, CM100 and Lean
`LearnedRecurrent/Rotary/Readout`. It supplies no fit or generalization guarantee.

| Variant | Starting model | Added training |
|---|---|---|
| Cautious / Mini and forward | Matching mode's immutable cautious seed1 endpoint | 512 original minibatches, 8 epochs |
| Refined / Mini and forward | Matching mode's immutable frequency-refined seed1 endpoint | Identical new batch stream and update settings |

All 129 components, 100 original objects/classes, full content matrices, initial
states and learned frequencies remain present. They were learned in the prior
stage; this controlled conditional step revisits all class readout coefficients.
Every raw object occurrence contributes through actual recurrent execution.
Temporary batch-state values are recomputed from the retained learned content;
there is no supplied feature map, answer program, magnitude pruning or rank cut.
The external contract is still an immutable-object sequence to one next object.

For each original minibatch, reserve the same finite-query centre in both modes.
Mini uses complete ordinary-CE residuals. Forward uses two actual CE evaluations
per original class and record, without a CE derivative or recurrent backward
pass. Contract those scalars with every conjugated complex state and its actual
mixture weight, then average over every original target record. Move every row
by the frozen learning rate 4, apply CM100's complete radial budget rule, and
search the full feasible interpolation interval [0,1] with 64 CE iterations.
Every accepted readout and every original-class logit is checked against actual
model execution. The radial rule is not claimed to be Euclidean projection.

All four endpoints seal before new held-out scoring. Report the original success
criterion and separate raw/exact 99% flags on every original split, including
failures. Independently replay every raw logit. All training CE evaluations are
counted: class-bias queries, line search, centre, scalar validation and actual
accepted-model check. Preserve the original failed producers and transformer
baseline. A gain in training fit alone does not resolve generalization.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/recurrent-readouts
export CARGO_BUILD_JOBS=2
cargo test --offline --locked --manifest-path experiments/convex_recurrent_readouts/Cargo.toml
cargo clippy --offline --locked --manifest-path experiments/convex_recurrent_readouts/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path experiments/convex_recurrent_readouts/Cargo.toml --check
cargo build --offline --locked --release --manifest-path experiments/convex_recurrent_readouts/Cargo.toml --bin stress
"$CARGO_TARGET_DIR/release/stress" plan
```

Freeze the executable, protocol, source archive, commit and hashes before:

```sh
bash experiments/convex_recurrent_readouts/run.sh WORKSPACE STORAGE SOURCE_RUNS FREEZE FRESH_COHORT
```

`SOURCE_RUNS` is the parent containing the immutable `cautious-v1` and
`refined-v1` rotary cohorts. All generated output belongs outside the worktree.
Two workers each have a 32 GiB address-space cap, with aggregate RAM admission
below 110 GB and continuous native guards. Stop below 2 GiB workspace space or
16 GiB storage space. Every source checkpoint, original dataset identity and
complete learned-content hash is verified.

All four native controls pass, covering both training modes, imaginary order
information, complete class/coordinate updates, unchanged learned content,
actual feasible signed queries and refusal to score an incomplete or modified
four-endpoint cohort. Clippy, formatting and shell checks pass as well.
