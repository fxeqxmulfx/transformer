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
horizon or full-hold guarantee has yet been measured.

The derivation is [PM1–PM4](../../review/flat_spectrum.md#prospective-probability-control).
The native library evaluates the same probability and motion bounds with
stable log odds. It is preparation for the frozen constructor/verifier, not
a completed training audit.

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
checkpoints. CUDA update defects, execution allowances and all 144 training
checkpoint slots still need to be checked against the recreated evidence.

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

The pointer to `ae801acc` supplied the full trainer revision, but that commit
contains only inference adapters that import it externally. The subsequent
refresh of `grokking` supplied the actual trainer bytes. The successful source
preflight is preserved in `.kanban/artifacts/80/source-preflight.log`.

## Checks

The full Lean build passes 13358 jobs. The original probability bridge has
a 17-theorem axiom audit; the encoder continuation adds a separate 27-theorem
audit and four native tests, bringing the native total to eight. Global
checks retain the existing 36 `sorry`, zero results resting on them and zero
extra axioms. The generated INDEX remains free of vacuous and placeholder
declarations. These checks do not supply the missing training runs.

```sh
cargo test --offline --locked --manifest-path experiments/probability_tube/Cargo.toml
cargo clippy --offline --locked --all-targets --manifest-path experiments/probability_tube/Cargo.toml -- -D warnings
cargo fmt --manifest-path experiments/probability_tube/Cargo.toml --check
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ProbabilityMotionAxioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/EncoderMotionAxioms.lean
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
GPT inputs successfully. Training and checkpoint evidence still need to be
recreated; the source check does not supply those measurements.
