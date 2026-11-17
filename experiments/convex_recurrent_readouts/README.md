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

The first cohort completes at producer `6cc8d452`, freeze
`recurrent-readouts-v1`, source fingerprint
`0b2a966f9f9caea8716d2bf475be283b1d5f3789299a5cbd8f220278b273f7ac`, binary SHA256
`f2b623685e0bed60cb5c96d41d7a6834d4be7e5fe336db0d78cbb2cf757d2174`.
All twelve train/score/replay commands exit0 and all four complete raw-logit and
metric replays match. Evidence is in
`/run/media/misha/storage/transformer-runs/recurrent-readouts/original-v1`.

| Source / mode | Train correct /2048 | Novel validation correct /512 | Test correct /512 | Train CE | Novel CE |
|---|---:|---:|---:|---:|---:|
| Cautious / Mini |1455 (71.045%)|70 (13.672%)|68 (13.281%)|2.06941|3.17215|
| Cautious / forward |1459 (71.240%)|54 (10.547%)|65 (12.695%)|2.05897|3.19051|
| Refined / Mini |1437 (70.166%)|63 (12.305%)|48 (9.375%)|2.09426|3.22125|
| Refined / forward |1393 (68.018%)|66 (12.891%)|75 (14.648%)|2.10767|3.20411|

Both original success and raw/exact99 flags remain false for every cell.
Raw and exact accuracies coincide because each example has one answer. Training
fit improves substantially from the retained sources' 8.5–9.6%, and novel
accuracy improves from their 2.0–4.7%. This demonstrates better fitting with the
same learned state functions; it does not show that those functions identify
the right unseen answers. There is still a large train/held-out gap. Only one
training seed and the original data seed are measured here.

Every complete content hash is unchanged. Each run retains129 components and
processes16384 original records over8 additional epochs; all four batch traces
match. All512 accepted interpolation coefficients are1, and all actual-logit
interpolation discrepancies are0. Maximum operator-budget excess is2.22e-16.
Each Mini run uses1163264 CE evaluations; each forward run4440064, including
3276800 signed class-bias queries. All runs include1114112 line-search and49152
centre/validation/accepted-model calls. Concurrent training times are16.62/18.85s
for cautious and16.27/19.10s for refined, Mini/forward respectively. These are
additional refit costs, not the cost of learning the source content or matched
transformer budgets. The next question is whether further fitting saturates or
whether revisiting shared content is needed;99% remains unresolved.
