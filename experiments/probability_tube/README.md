# Prospective probability control

Can a checkpoint's small unused-class probabilities remain small through
actual training motion, long enough to make the CE momentum condition useful?
The [frozen protocol](PROTOCOL.md) fixes the comparison before measurements.
The overall goal still includes full optimizer/encoder control, useful geometry
and prediction; the probability bound alone does not establish those results.

## Formal result

For a particular class u, let p be its actual checkpoint softmax probability.
If every relative logit `z_u-z_c` increases by at most D, then

```
p_future <= p*exp(D) / (1-p+p*exp(D)).
```

This ceiling is sharp, stays in [0,1], and increases with the initial probability
ceiling. It permits signed D. Separate budgets for each rival give a stronger
bound with all rivals retained. Contrast increments telescope over any finite
prefix. A nonzero Lean witness attains the bound: p moves from 1/5 to 1/3
when its contrast increases by log 2.

For an old readout contrast of norm at most A, old feature norm at most X,
readout-motion radius R, feature-motion radius S and execution-contrast change
at most e, the deterioration budget is `R*X+A*S+R*S+e`. The mixed term cannot
be dropped. These are explicit bounds on parameter/feature motion, not an
assumption that future probabilities are small. Feature motion must cover the
full tied encoder and final normalization.

Seventeen new Lean theorems compose the transported ceiling with the existing
weighted actual-partial-CE momentum theorem. This proves the reference step's
nonincrease when `19*s*q_future*max_weight*feature_energy_cap <= 2`.
Full AdamW/encoder perturbations remain separate, and no useful numerical
horizon or full-hold guarantee follows from this reference theorem. The
recreated comparison below measures zero useful positive horizons.

The derivation is [PM1–PM4](../../review/flat_spectrum.md#prospective-probability-control).
The native library evaluates the same probability and motion bounds with
stable log odds. Separate frozen constructors and future-state verifiers
evaluate those bounds on the recreated training evidence.

## Full encoder motion

The next 27 Lean theorems derive feature motion from parameter caps through
both shared input rows, three moving real matrices, the calibrated complex
product, the residual and all four RMSNorms. Matrix/input and gate/value
mixed terms remain. Each RMS radius intersects the old-denominator bound,
local secant bound and range diameter. Exact source-correspondence theorems
identify the resulting features and tied logits with the existing
complex-linear Rotate90 model. The final feature energy is bounded by the
real width, supplying the ordinary cap needed in the CE condition.

The full encoder radius feeds the all-rival probability ceiling. Feature
displacement is no longer a separate unexplained premise. Warm-Adam cones
give parameter displacement about the checkpoint for arbitrary future
gradients. The native constructor applies them to every embedding row and
each complete matrix, including repeated entries in the tied expanded
parameter buffer. It takes saved parameters/moments and a known schedule;
it cannot read future gradients or states. Old execution residuals and
changes of execution contrasts are separate inputs to the probability bound.

See [EM1–EM5](../../review/flat_spectrum.md#full-encoder-motion),
[encoder.rs](encoder.rs), [adam.rs](adam.rs) and
[the Lean module](../../src/Transformer/Rotate90/FeatureMotion.lean).
The constructor uses conservative Frobenius operator caps. Its arithmetic is
FP64, not a formal outward-rounded certificate. Synthetic tests preserve the
source parameter ties and check every intermediate layer, all class
contrasts, probabilities, original dimensions and 24 arbitrary-gradient
Adam prefixes. Small perturbations produce bounds below the final feature
diameter in those tests. This is not a useful-horizon finding on trained
checkpoints. The recreated audit checks numerical allowances and full motion
at all declared endpoints; it does not prove a universal CUDA error bound.

The independent [pairwise refinement](sharpen/README.md) adds 14 Lean
theorems and a native library. It replaces full-vector product amplification
by old pair amplitudes and permits proved Gram operator caps. The final
radius retains the original bound as a comparison. Its four synthetic tests
include original dimensions; no useful training horizon is inferred from
them. The first frozen audit continues to use its original formula.

## Recreating the evidence

The historical ignored run directories and the task #207 artifacts are absent
in this checkout. The user authorized recreation on 2026-11-09. This machine
reports a GTX 1050 with 2 GiB, rather than the previously documented RTX 3050;
new runs must therefore carry separate execution provenance.

The pinned Kernite revision `a1b6bc2c956e1a5f35960e26b209e1b6c213096c`
has been recovered into the root's ignored `.lake/reproduction/kgen` checkout.
The user-provided [grokking source](https://github.com/fxeqxmulfx/grokking/tree/b70811e0e0e7ebdc4a8a2e48044c180332f7c40d/experiments/rotate90_complex)
now exposes the original trainer revision
`b70811e0e0e7ebdc4a8a2e48044c180332f7c40d` in the history of `9022f18`.
It is recovered into `.lake/reproduction/grokking/experiments/rotate90_complex`
at that exact revision. The concatenated source digest matches
`14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da` exactly,
including the original initialization and minibatch ordering.
The user-provided [diff-gpt source](https://github.com/fxeqxmulfx/diff-gpt/tree/6bb9d03a7184cb0a43bfaf240977639213e2cbeb)
has been recovered into `.lake/reproduction/diff-gpt`. All four GPT manifest,
optimizer, optimizer-kernel and schedule digests required by the old runner
match exactly. The selected cohorts keep the original Rotate90 architecture
and use the main-model AdamW/WSD adapter; the full attention architecture
from `diff-gpt` is not substituted. The source prerequisite is now satisfied.
The [source-checked native reproduction runner](reproduce/README.md) imports
those exact sources and preserves the original probe/update checks while
adding every declared forecast checkpoint. It records separate hardware,
source, executable and PTX provenance and all cumulative attempt clocks.
No replacement baseline report or historical result has been manufactured.

The separate [native forecast audit](audit/README.md) reads only anchor states
and input identities during construction. It freezes per-input ceilings for
all declared horizons before checking future checkpoints. It retains both
fixed metrics, the invalid fixed-q comparison, numerical excesses and full
potential changes. The [matched refined audit](sharp_audit/README.md) changes
only the encoder radius, preserving the parameter caps, metrics, execution
allowances and future-state verifier. Its implementation was frozen at
`d3893996` before inspecting recreated probability outcomes.

The pointer to `ae801acc` supplied the full trainer revision, but that commit
contains only inference adapters that import it externally. The subsequent
refresh of `grokking` supplied the actual trainer bytes. The successful source
preflight is preserved in `.kanban/artifacts/80/source-preflight.log`.

## Recreated training

All three original-source cohorts finish all eight seeds and 10000 updates
per seed on the GTX 1050. The complete inventory contains 240000 updates,
744 geometry probes and 1680 unique full checkpoints. Every geometry check
passes, with zero false certificates and zero partial-CE damping failures.
All 720 sampled optimizer updates match the independent FP32 reference with
zero ULP differences, and ordinary/instrumented update buffers match exactly.
Maximum scaled update-energy closure error is 1.78e-13. These are new runs
with their own provenance; they do not restore the missing historical files.

Each row retains all eight seeds. Accuracy, rank-32 readout certificate
coverage, active-contrast share and top-32 readout tail are endpoint means.

| Cohort | Charged seconds / 7200 | Validation accuracy | Rank-32 coverage | Active share | Top-32 tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| .001 WSD | 1324.36 | 99.567% | 97.970% | 73.054% | 23.389% |
| .01 WSD | 1407.28 | 99.631% | 98.015% | 66.524% | 25.531% |
| Early settling | 1360.21 | 99.575% | 98.324% | 71.993% | 22.205% |

The settling cohort retains 99.51–99.62% minimum validation accuracy across
each seed's probes from step 4000 onward. Its corresponding minimum rank-32
coverage is only 97.87–98.49%. Thus stable observed prediction and relative
energy concentration still leave a compressed-certificate gap. The table
does not establish why training discovers an aligned representation, nor a
guarantee between the sampled geometry probes.

The [native summary generator](training_summary.jq) retains every seed,
probe time, endpoint and failed slot. Regenerate it from the root:

```sh
jq -s -f experiments/probability_tube/training_summary.jq \
  experiments/probability_tube/runs/lr001/attempt-01/report.json \
  experiments/probability_tube/runs/lr01/attempt-01/report.json \
  experiments/probability_tube/runs/settle/attempt-01/report.json \
  > experiments/probability_tube/runs/training-summary.json
```

## Frozen probability findings

Both constructors seal all 144 anchors and 1800 declared endpoints. Their
input ledgers contain only each case's identity and named anchor state.
Both use all 4704 training inputs and every unused-class rival. Initialization
has zero learning rate and is reported separately; it supplies no useful
moving-training stability result.

Across the 120 moving anchors and 1368 positive-horizon endpoints, the
original Frobenius/full-vector forecast saturates at q=1 everywhere. The
pairwise/Gram refinement reduces 16 ceilings below one, all in late
settling, but **neither method supplies even a one-update raw CE stability
horizon in either metric**. Zero remains in every seed's useful-horizon
inventory. The universal q=1 condition also fails at every moving anchor.

For comparison, holding the checkpoint's measured maximum probability fixed
would pass 47 isotropic and 13 feature-metric conditions. Those are explicitly
invalid forecasts because they omit subsequent motion. Counts below are
isotropic / feature and retain all eight seeds in every cell.

| Cohort | Anchor 1000 | 4000 | 6000 | 8000 | 9500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| .001 WSD | 6 / 0 | 8 / 1 | 6 / 1 | 7 / 3 | 4 / 0 |
| .01 WSD | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 0 |
| Early settling | 0 / 0 | 2 / 0 | 1 / 0 | 3 / 0 | 8 / 8 |

The refined one-update maximum feature radius still reaches the full
diameter `2*sqrt(128)` at 112 of 120 moving anchors. At settling anchor 9500,
it drops from 22.627 to 2.400–3.645, but the resulting uniform probability
ceiling is 6.702e-5–1.249e-2. The corresponding isotropic coefficients are
158.60–20323.93 and feature coefficients 200.87–34022.28, against the required
maximum of two. A smaller valid radius therefore does not yet supply the
needed stability condition. These radii describe the enclosure, not observed
feature displacement.

The parameter tube permits arbitrary future gradient directions. Propagating
that freedom through the full moving encoder remains too conservative here,
even after removing avoidable product and matrix amplification. This finding
does not establish impossibility of useful future control. The next step
needs sharper control of the actual future gradients or their directional
effects, alongside a less conservative CE coefficient. Full adaptive/encoder
perturbations, discovery, alignment and the certificate-coverage gap remain
open parts of the goal.

The completed original future-state audit passes all 1800 endpoints. Actual
unused-class maxima exceed the fixed old maximum by more than 1e-8 at 396 of
the 1368 moving positive-horizon endpoints; the largest excess is 0.90323.
Observed maxima span 7.66e-8–0.98618. Thus a small measured checkpoint value
cannot be carried forward without a motion argument.

The recreated next-update decomposition retains the earlier obstruction.
On 120 moving cases, the actual-minibatch raw CE condition passes 58/44
isotropic/feature cases. Full raw potentials nevertheless rise in 31/30,
including 13/11 that meet that reference condition. The stable diagnostic
reference meets its condition everywhere but has 36/34 full increases.
These diagnostic conditions use the executed minibatch, unlike the uniform
all-input forecasts. Different potentials do not describe different training
runs.

Across all positive horizons from moving anchors, the full fixed potentials
rise at 483/489 endpoints. None of the 360 later-settling endpoints from
anchors 4000 onward increases either potential. Those observations overlap
in time, and the lr01/settle prefixes coincide through anchor 1000; they are
not independent trials or a guarantee between endpoints. The fixed feature
metric's ordinary-energy penalty is 2.053–13.412 times old unused energy,
versus 1.899–1.906 for the isotropic metric. The sharper probability forecast
does not alter any executed update, potential or geometric certificate.

The original construction plus verification takes 462.71 seconds against
its 1800-second cap, without missing or failed slots. Maximum scaled
parameter, feature and contrast excesses are zero, and maximum probability
excess is 1.12e-16. Feature/readout execution errors are at most
4.44e-5/2.09e-5, feature energy at most 128.000041, partial-gradient scaled
error at most 3.48e-8 and full-identity scaled error at most 1.14e-14. Every
declared allowance is unchanged. The finite checks support the numerical
implementation; whole-window statements still require the stated execution
allowances throughout that window.

The refined verifier also passes every endpoint. Its matched comparison
passes all 144 cases and 1800 endpoints with identical old probabilities,
parameter caps, fixed metrics, executed endpoint probabilities, full
potentials and signed next-update decompositions. Maximum per-query feature
and probability excess over the original forecast is zero. Construction,
verification and comparison together take 637.62 seconds against the separate
1800-second cap; no failed or capped slot is dropped. Both measured methods
therefore validate while retaining zero useful positive CE horizons.

The constructor/verifier commands are in the two audit READMEs. Additional
complete-inventory summaries are generated natively, without loading all
per-query arrays into one process:

```sh
set -o pipefail
for forecast_name in forecast-01 sharp-forecast-01; do
  forecast_root="experiments/probability_tube/runs/$forecast_name"
  jq -c -f experiments/probability_tube/forecast_summary.jq \
    "$forecast_root"/*-seed*-anchor*.json | jq -s . \
    > "$forecast_root/forecast-summary.json"
  jq -f experiments/probability_tube/audit/summary.jq \
    "$forecast_root/verification-01.json" > "$forecast_root/summary.json"
  jq -f experiments/probability_tube/outcome_summary.jq \
    "$forecast_root/verification-01.json" > "$forecast_root/outcome-summary.json"
done
jq -f experiments/probability_tube/sharp_audit/summary.jq \
  experiments/probability_tube/runs/sharp-forecast-01/comparison-01.json \
  > experiments/probability_tube/runs/sharp-forecast-01/comparison-summary.json
```

All generated training, frozen forecasts, verifications, comparisons and
summaries remain under the root's ignored `experiments/probability_tube/runs/`.
The measured executables are `.lake/probability-reproduction-bins/`,
`.lake/probability-audit-bin` and `.lake/probability-sharp-audit-bin`.
The first and refined audit source fingerprints are respectively
`b237a39dddb8fc1f905849c390c8cbc7f4f5cb3cdd1d962a2c174a6e08fa1181`
and `e1a3a4adab7f179aa5e45b552f8b4fefc9f00cc8c3cef05ab81ead0b702d9d68`.
Logs and review evidence are preserved in `.kanban/artifacts/81/` and
`.kanban/artifacts/83/`. No generated report is committed.

## Query-weighted CE continuation

The first two audits replace all query probabilities by their maximum and
the feature operator by a trace. The next 27 Lean theorems retain the
per-query ceilings in the weighted covariance. They prove actual partial-CE
coercivity, finite feature-motion transport, arbitrary-momentum control over
varying batches, and an explicit conversion from executed GPU probabilities
to the real CE reference. This preserves the fixed metric and source batch
sizes, including the 96-example tail of each epoch.

The [native evaluator](gram/README.md) passes five independent tests, with
full-width feature tables at both source batch sizes and a finite-difference
check of the actual CE derivative under nonzero feature motion. The new
coefficient has not yet been measured on the recreated runs. Its
[comparison protocol](gram_audit/PROTOCOL.md) fixes both source forecast
families, three coefficients, all anchors/horizons/metrics and an independent
7200-second cumulative budget. The constructor and verifier must be complete,
checked and frozen before measurements. The previous zero-horizon findings
remain unchanged.

The new formal layer is committed at `fc7a3a3a`. Its full build passes 13369
jobs, and `scripts/CEGramAxioms.lean` checks all 27 results. Global audits
retain 36 existing `sorry`, zero dependencies on them and zero extra axioms.
The native and Lean logs are preserved in `.kanban/artifacts/84/`.

## Checks

The full Lean build passes 13362 jobs. The original probability bridge has
a 17-theorem axiom audit; the encoder continuation adds a separate 27-theorem
audit and four native tests, bringing the native total to eight. Global
checks retain the existing 36 `sorry`, zero results resting on them and zero
extra axioms. The generated INDEX remains free of vacuous and placeholder
declarations. The pairwise continuation adds its own 14-theorem axiom audit
and four native tests. Mathematical checks and finite execution checks are
reported separately.

```sh
cargo test --offline --locked --manifest-path experiments/probability_tube/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/probability_tube/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/Cargo.toml --check
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ProbabilityMotionAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/EncoderMotionAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SharpEncoderAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --locked --release --manifest-path scripts/native_index/Cargo.toml -- --check
rg -n 'native_decide|^axiom |linter\..* false|autoImplicit' src
```

The final scan should find nothing (rg exit 1). Generated logs are preserved
at the root in `.kanban/artifacts/79/` and `.kanban/artifacts/80/`. Native tests compare the formulas
against direct softmax, an attained ceiling, signed cumulative motion and
nonzero mixed terms. They are numerical checks, not a floating-point proof.

The read-only native source preflight checks the trainer's concatenated source
digest, Kernite's exact revision and clean sources, and the four GPT digests.
Supply the recovered trainer's directory as the first argument:

```sh
cargo run --offline --locked --manifest-path experiments/probability_tube/Cargo.toml \
  --bin check-rotate90-sources -- "$PWD/.lake/reproduction/grokking/experiments/rotate90_complex" \
  "$PWD/.lake/reproduction/kgen" \
  "$PWD/.lake/reproduction/diff-gpt/research/main_model"
```

The current preflight verifies the original trainer, Kernite and all four
GPT inputs successfully. The completed training evidence is preserved under
the root's ignored `experiments/probability_tube/runs/`; its source and
execution logs are in `.kanban/artifacts/81/`.
