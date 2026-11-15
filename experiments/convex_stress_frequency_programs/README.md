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
Nine native checks pass, including complete candidate-search serialization and
replay. These are implementation checks, not new benchmark measurements.

Before production, the generic program bank, training-only selection, declared
coordinate/copy priors and categorical-decoder control must be frozen. The
frequency runner must seal all modes before any heldout scoring, preserve all
original masks and final-answer rules, use own-output generation through EOS,
retain failures and compare a fresh complete repeat. Inherited controls and
the original35 results must remain explicit. Component checks alone do not
establish any new stress-test pass.

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
