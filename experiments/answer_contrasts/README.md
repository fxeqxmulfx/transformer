# Combined answer contrasts on frozen cubic checkpoints

Do the saved classifiers preserve the answer more accurately than the full sum,
and does retaining cancellation make a useful class-contrast bound? The fixed
diagnostic supports both, with a remaining limitation: the worst-case L1 and
weight-norm bounds still certify no held-out inputs. No model was retrained.

The [protocol](PROTOCOL.md) was fixed before measurement. All 24 original
checkpoints, all four arms, both seeds and all three steps remain in
[report.json](report.json). The earlier [spectral diagnostic](../flat_approx/README.md)
and its report are unchanged. Both use the previously examined validation split.

## Endpoint results

Each row has 4705 held-out inputs at step 10000. Indicator probes use the same
raw residual states, training-only ridge fit and fixed affine mass correction.
The sum probe has 97 coordinates; the answer probe pools the three equivalent
nonzero sums, giving 33 coordinates including zero. All 97 output labels still
compete in decoder comparisons.

“Combined” adds measured source-alignment error to the magnitude of the recovery
contrast **after summation**. “Coordinate” instead sums absolute contributions
before comparing with the same competitor's ideal gap. A positive budget requires
all 96 wrong-class comparisons to pass. These columns measure retrospective
sufficient conditions with zero assumed execution error, not unseen-data guarantees.

| Arm | Seed | Source accuracy | Sum probe accuracy | Answer probe accuracy | Combined positive | Coordinate positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 99.43% | 31.35% | 99.55% | 97.92% | 13.75% |
| Real SwiGLU | 11 | 99.17% | 29.93% | 99.34% | 95.30% | 29.61% |
| Complex diagonal | 7 | 99.83% | 37.92% | 99.57% | 98.72% | 87.12% |
| Complex diagonal | 11 | 99.74% | 40.60% | 99.57% | 98.30% | 94.98% |
| Complex radial product | 7 | 99.68% | 40.32% | 99.62% | 99.13% | 12.86% |
| Complex radial product | 11 | 98.79% | 43.04% | 99.45% | 98.30% | 60.26% |
| Complex plain product | 7 | 99.66% | 45.40% | 99.53% | 99.30% | 46.10% |
| Complex plain product | 11 | 99.68% | 46.97% | 99.66% | 99.43% | 91.67% |

The full-sum decoder's combined budget is also strong at the plain-product
endpoints: the same counts, 4672 and 4678 positive queries. Pooling does not improve
every diagnostic. Its clearest effect is which representation distinctions
the probes need to recover. These linear-probe results do not show that all
information about the exact sum is absent from the model.

## What is proved and what remains missing

The [Lean extension](../../review/flat_spectrum.md#combined-answer-contrasts)
proves that pooling preserves mass, contracts absolute error, and commutes with
linear prediction and the chosen mass correction. Fiber-balanced errors and
common score offsets disappear. For a zero-total-mass error `e` and contrast
coefficients lying in `[lo,hi]`, the error is at most
`(hi-lo)/2 * sum |e|`. A two-answer witness attains this bound. The actual-source
theorem adds alignment and numerical errors while retaining each competitor's
own gap and the actual final RMS denominator. It has a nonzero-error witness.

The independent 33-coordinate ridge probe agrees with pooling the full probe
to `5.385e-15`. This is the expected linearity identity, not evidence of two
independent discoveries. At the plain-product endpoints, mean L1 error drops
from 3.59/3.46 to 2.45/2.18; the mean per-query reduction is about 32%/37%.
Answer classification is accurate despite these substantial signed-coordinate
errors. The errors are not claimed to be exactly invisible.

The half-range L1 bound has **zero positive held-out budgets in all 24 cases**.
Its constant is sharp in general, but it loses too much directional information
for these errors. Replacing measured alignment with the Cauchy-Schwarz bound
from the actual affine weight residual also gives zero positive budgets, even
when retaining the measured combined recovery error. The plain-product mean
alignment is about 1.78/1.82, versus weight-norm bounds of 18.70/16.04.

Thus the combined-error measurement identifies a useful condition, but it is
not a predictive learning explanation. A next theorem must control the relevant
error directions and alignment from properties established without evaluating
held-out target errors. A predictive replacement for the failed nuclear-norm
bound remains open. Complex multiplication and the original checkpoints remain
unchanged.

## Verification and reproduction

All 24 checkpoint identities exactly match the earlier report. Every stored
GPU train/validation correctness count reproduces. Sampled batched/scalar FP64
logits differ by at most `2.265e-14`; affine source-alignment expansion differs
by at most `2.487e-14`. Every measured combined contrast satisfies both the
coordinate and range inequalities within `1e-8`.
These calculations are not certified floating-point bounds.

Six native tests cover known GEMM/SVD/regression results, all source arms
against the frozen scalar oracle, split/mask reconstruction, commuting ridge
fits and pooling, cancellation and a sharp range-bound example. Formatting and
strict Clippy pass. The complete diagnostic took 9.52 seconds, excluding
compilation, within its five-minute cap; OpenBLAS used four CPU threads.
These are author checks, not independent review.

```sh
cargo test --offline --release --manifest-path experiments/answer_contrasts/Cargo.toml
cargo fmt --manifest-path experiments/answer_contrasts/Cargo.toml --check
cargo clippy --offline --release --manifest-path experiments/answer_contrasts/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/answer_contrasts/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw /tmp/answer-contrast-report.json
```

The report hashes the fixed protocol and all compiled source inputs, including
the shared source evaluator, split code, OpenBLAS bindings and frozen oracle
under `flat_approx`. It records both decoder choices and every failed case.
