# Frequency programs with supplied class geometry

Task339 tests whether a training-selected input-program family and an explicitly
supplied class-coordinate prior can identify missing raw count labels. The
target cohort is the seven reviewed337 frequency datasets, keeping all16
reviewed334 controls and the separate35-task original comparison. The complete
CM56 decoder, ordinary cross-entropy training component, finite input-program
search and frozen benchmark runner are implemented.

Producer `34ea5801d748f487c6c83b930b43047fc3b577eb` completes the original run
with no process failures. The coordinate arm gets 100% raw and exact accuracy
on every one of the 11 splits of all seven cases, in all three modes. Every
case's internal reconstruction and complete-logit replay is exact. A fresh run
with an independently rebuilt, byte-identical binary reproduces every one of
the 123 JSON/JSONL artifacts exactly, including full raw logits. Both drivers
exit0. The complete report SHA256 is
`4501ba0f11e8e90da1f548b3f71fc619823ce2062e3f2039b0d57ce3be7d5e0d`;
the relative artifact manifest SHA256 is
`f8fa1558162949f0677e494bf8eba7f36f509dfe9413631146d4c9be1fd744f6`.
The freeze preserves 5427 source inputs, all local Lean modules and the binary.
Evidence lives under storage's `freezes/frequency-programs-v1` and
`frequency-programs/{original-v1,review-v1}`, outside Git.

Across the short, length32 and length64 test sets, each mode gets all 32,696
target tokens and 1,344 complete sequences correct, including all 7,410 target
occurrences whose raw labels were absent from training outputs. These totals
exclude training and validation. Ordinary full-vocabulary CE is measured on
the original teacher-forced prefixes; accuracy uses own-output generation.

| Mode | Test raw accuracy | Test exact accuracy | Token-weighted test CE |
| --- | --- | --- | --- |
| Forward-only | 100% | 100% | 0.006194156 |
| Full batch, 64 steps | 100% | 100% | 0.000703828 |
| Minibatch, 8 epochs, batch127 | 100% | 100% | 0.000452416 |

The categorical control meets both 99% primary criteria and the catalog's
original success criterion on all seven cases. Its all-heldout result is 1/7,
and its short-test both99 result is 4/7, in every mode. The two arms independently
select the same program on each of these seven datasets, so the measured
difference here comes from quantity-decoder completion. Both arms retain the
explicit input-identity copy prior. These are additional controls, not a fresh
matched-budget transformer comparison.

| Case | Length64 targets | Coordinate raw/exact | Categorical raw | Categorical exact |
| --- | ---: | --- | ---: | ---: |
| histogram-bos | 4224 | 100% /100% | 1432/4224 | 2/64 |
| histogram-no-bos | 4160 | 100% /100% | 1368/4160 | 2/64 |
| histogram2 | 4224 | 100% /100% | 4224/4224 | 64/64 |
| mode-none | 128 | 100% /100% | 64/128 | 0/64 |
| mode-counts | 566 | 100% /100% | 387/566 | 0/64 |
| mode-itemized | 566 | 100% /100% | 387/566 | 0/64 |
| most_freq | 4224 | 100% /100% | 4224/4224 | 64/64 |

This table holds in all three modes. Histogram2's categorical failures occur
on other splits. Mode-none illustrates an unseen latent quantity even when
its output label was observed: the selected program copies the first symbol
in descending-frequency order, then maps its count to EOS. Its affine fit
learns a constant EOS decoder from distinct observed counts. The categorical
decoder has no entry for a new count and emits its declared raw0 default.
Mode-counts and mode-itemized retain 100% final-mode diagnostics while failing
exact generation. The original 95% exact success rule is never replaced by
that diagnostic.

The complete inherited 16-case report remains byte-identical under canonical
serialization. Together with these seven new cases it reports 23/23 additional
cases meeting both 99% goals and all-heldout criteria in every mode. The
separate reviewed original35 result is preserved unchanged. Thus 58 of the 165
catalog cases have model measurements; 107 remain unmeasured. The 16 controls
and original35 are inherited, not fresh evaluations of this new head.

The original driver took 9:35.63 and peaked at 8,606,936 KiB RSS. The repeat took
9:25.60 and peaked at 8,608,460 KiB; both report zero swaps.
The process address limit is 64GiB; the largest complete case buffer forecast
is 13,609,811,456 bytes. Every full batch makes 64 effective CE updates, and
every minibatch endpoint makes effective updates with its true tail size.
The largest independent complex-product residual is 5.821e-11. These checks
do not establish floating-point refinement or feature-family identification.

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

The trained quantity decoder is a fixed preprocessing map for this CE stage.
The head uses its decoded center as the scalar feature; the observed centers
fit the identity affine readout. Thus CM56 applies first to quantity completion
under its affine prior and then to the actual head with this fixed feature.
Discrete program selection and decoder fitting are outside the convex CE
optimization. The experiment does not claim a jointly convex search over
programs, supplied class geometry and head parameters.

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

Compare every saved JSON/JSONL artifact after both drivers finish. Hashes are
computed from the files themselves, including complete raw vectors and retained
references; resource logs are intentionally excluded because timings vary.

```sh
for run in original-v1 review-v1; do
  (
    cd "$store/frequency-programs/$run"
    rg --files --hidden --no-ignore -g '*.json' -g '*.jsonl' -0 . |
      LC_ALL=C sort -z | xargs -0 sha256sum
  ) > "$store/frequency-programs/$run.sha256"
done
cmp "$store/frequency-programs/original-v1.sha256" "$store/frequency-programs/review-v1.sha256"
```

Regenerate the test-only aggregate CE and raw/sequence counts without parsing
the large inherited report:

```sh
jq -s '
  [.[] | .results.endpoints[] | select(.prior == "Coordinates") |
    .mode as $mode | .measurement.splits | to_entries[] |
    select(.key == "test" or (.key | startswith("test/"))) |
    {mode:$mode, targets:.value.metrics.targets, correct:.value.metrics.correct,
     exact:.value.metrics.exact, loss:.value.metrics.teacher_forced.loss,
     unseen:.value.unseen_label_targets, unseen_correct:.value.unseen_label_correct}] |
  group_by(.mode) | map({mode:.[0].mode,
    targets:(map(.targets)|add), correct:(map(.correct)|add), exact:(map(.exact)|add),
    unseen:(map(.unseen)|add), unseen_correct:(map(.unseen_correct)|add),
    weighted_ce:((map(.loss*.targets)|add)/(map(.targets)|add))})
' "$store/frequency-programs/original-v1"/*/metrics.json
```
