# Frequency programs with supplied class geometry

Task339 tests whether a training-selected input-program family and an explicitly
supplied class-coordinate prior can identify missing raw count labels. The
target cohort is the seven reviewed337 frequency datasets, keeping all16
reviewed334 controls and the separate35-task original comparison. No frequency
model result has been measured yet. The complete CM56 decoder, ordinary
cross-entropy training component and finite input-program search are implemented.

Lean first: `b5f1c9d70eb3db33861e28cae6d1512f84c0fedf`,
`ConvexDepth/ClassGeometry/{Basic,Head,Witness}.lean`,
`scripts/UnseenClassGeometryAxioms.lean`, and CM56 in
`review/flat_spectrum.md`. Two distinct observed feature anchors identify an
unknown affine decoder in supplied injective class coordinates. The actual
four-point complex head has eight realified coordinates per raw output class,
with shared scaled slope, intercept and margin. Its complete sufficient costs
are `(4,4,2)`. Flatness bounds normalized features; it does not supply the class
geometry or select an input program.

The native decoder fits only the two first distinct observed feature anchors
and checks every training record, including duplicate input values with
distinct IDs. Prediction multiplies the saved complete coefficient tensor by
the signed normalized complex features. A separate complex multiplication
checks that product. Every raw class is retained, including unobserved ones.
The fitted slope/intercept freeze before CE; a positive shared margin varies
on the convex observed-anchor ray. Full and minibatch steps use ordinary
full-vocabulary CE and its actual derivative, project the margin to the charged
interval, and retain every step and original record ID. The last minibatch
uses its true size. Forward-only makes no CE updates.

This is a real-number theorem with a separately checked floating-point port.
Residual, feature, gradient and coefficient checks are numerical evidence;
there is no formal refinement of floating-point arithmetic or convergence
claim. The library refuses missing anchors, contradictory observations,
unresolved coordinates, nonfinite values and excessive work/memory forecasts.
Serialization is checked against complete reconstruction and training replay.

The fixed input-program bank has6240 configurations per inferred delimiter.
It combines positions/distinct groups/extrema, source/frequency/reverse order,
frequency/equal-frequency-group/prefix-count/position statistics, quantity/copy
emissions and structural prefix/padding/tail options. Raw labels and task names
are absent from execution. An independent interpreter recounts source tokens
and renders the whole latent sequence. Tests compare every configuration on
repeated-token and tied-frequency fixtures, including first-occurrence order.

Discovery saves every candidate, the first conflicting record pair and every
compatible decoder. Structural labels and categorical quantity dictionaries
are learned from training only. The coordinate arm additionally fits an affine
quantity decoder from observed quantity/label anchors. Both arms retain the
explicit identity-copy prior. Their selectors independently take the first
training-compatible candidate in the frozen order. No claim is made that this
selection uniquely identifies an unknown program or all its unseen outputs.

The categorical quantity control uses observed quantity labels and an explicit
raw0 default on unknown quantities. It shares the complete scoring head and
copy prior, but does not extrapolate the quantity decoder. Unknown/defaulted
events remain flagged. A synthetic unseen-quantity test separates the two
arms in all three training modes, while all observed records remain correct.
Thirteen native checks pass, including complete candidate-search serialization,
all six saved training states, independent raw scoring, own-output generation,
memory refusals before fitting, seal mutations and failure denominators.
These are implementation checks, not new benchmark measurements.

The runner seals all seven cases and six endpoints per case before heldout
scoring. It regenerates and checks the reviewed data, preserves every original
mask, and generates from its own outputs through EOS or the original limit.
Every teacher-forced and generated prefix retains the complete raw logit vector
and independent complex-product residual. Unseen labels, unknown decoder
queries, exact sequence accuracy, raw accuracy and ordinary teacher-forced CE
are reported separately. The catalog's original success threshold for these
tasks is 95% exact sequences; Mode's final-token accuracy is diagnostic only.
The user's 99% raw and exact targets are additional reported criteria.

The fixed protocol supplies `e(c)=(c-100)/128` for all raw classes. It uses margin
131072 initially, coefficient radius8388608, and a shared learning rate1e10:
forward-only makes zero updates, full batch takes64 steps, and minibatch makes
eight epochs with batch127 and a true short tail. These values are frozen before
production. Selection sees training records only; the heldout shapes enter
resource forecasts, never fitting or candidate ranking. The complete case
forecast charges three search/training reconstructions and both full scoring
passes for all six endpoints, with a600 trillion scalar-work ceiling and56GiB
buffer ceiling. Full raw vectors stream to disk; their subsequent hashing
allocation is included in the buffer bound. All phases have a64GiB address cap.

The complete16-case334 report is retained as inherited control evidence, and
the reviewed35-case336 summary is copied separately. These are not retrained
by this experiment. Refusals stay in the seven-case denominator. Neither the
conditional Lean theorem nor a training-compatible selected program certifies
the unknown target's realizability; every report keeps that distinction explicit.

```sh
export CARGO_TARGET_DIR=/run/media/misha/storage/transformer-runs/build/frequency-programs
export CARGO_BUILD_JOBS=2 LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
lake env lean -j 2 scripts/UnseenClassGeometryAxioms.lean
cargo fmt --manifest-path experiments/convex_stress_frequency_programs/Cargo.toml --check
prlimit --as=68719476736:68719476736 -- cargo test --offline --locked --manifest-path experiments/convex_stress_frequency_programs/Cargo.toml -- --test-threads=2
prlimit --as=68719476736:68719476736 -- cargo clippy --offline --locked --manifest-path experiments/convex_stress_frequency_programs/Cargo.toml --all-targets -- -D warnings
prlimit --as=68719476736:68719476736 -- cargo build --release --offline --locked --manifest-path experiments/convex_stress_frequency_programs/Cargo.toml
```

Component tests cover an unseen raw class under the CM56 witness, nonzero
imaginary coefficients, all-class independent products, transported categorical
relabeling, effective full/minibatch updates, a repeated-input short batch,
finite-difference CE derivatives, saved-state mutations and forecast refusals.
Generated evidence belongs on storage, not in Git. The process cap is64GiB;
component forecasts reserve56GiB at most and must be composed with the runner's
data, selection, history, evaluation and replay costs before any full run.

After committing the producer and passing these checks, freeze and run with
fresh destination directories. The shell driver records every phase's exit and
resource usage, continues to retain failures, and summarizes only complete
replays. Do not edit a running frozen binary or driver.

```sh
store=/run/media/misha/storage/transformer-runs
freeze="$store/freezes/frequency-programs-v1"
binary="$CARGO_TARGET_DIR/release/convex-stress-frequency-programs"
experiments/convex_stress_frequency_programs/freeze.sh "$freeze" "$binary"
mkdir -p "$store/frequency-programs"
"$freeze/source/experiments/convex_stress_frequency_programs/run.sh" \
  "$freeze/convex-stress-frequency-programs" "$store/frequency-programs/original-v1" \
  original "$(cat "$freeze/producer.commit")" \
  "$store/frequency-data/original-v1.json" "$store/partitioned-compare/original-v1/summary.json"
# Rebuild independently from the committed source, check binary equality,
# then invoke the frozen driver. Repeat copies only pinned input references.
"$freeze/source/experiments/convex_stress_frequency_programs/run.sh" \
  "$freeze/convex-stress-frequency-programs" "$store/frequency-programs/review-v1" \
  repeat "$store/frequency-programs/original-v1"
```
