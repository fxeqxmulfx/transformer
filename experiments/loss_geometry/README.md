# Geometry controlled by cross-entropy

Small training cross-entropy controls probability error and within-answer
probability scatter. It does not imply contraction of raw class contrasts.
The [Lean extension](../../review/flat_spectrum.md#loss-controlled-geometry)
proves the objective bound and an actual shared-parameter gradient-descent
counterexample to raw contraction. The fixed checkpoint diagnostic finds the
same distinction in two saved training intervals.

The [protocol](PROTOCOL.md) was fixed before measurement. All 24 original
states are retained in [report.json](report.json): four arms, seeds 7/11,
steps 0/5000/10000. No training was run. Earlier diagnostics and checkpoint
files are unchanged. Both sum grouping and cubic-answer orbit grouping are
reported, with training and validation means computed independently.

## What follows from the objective

For softmax probabilities `p`, target `t` and the actual row loss `ell`,
Lean proves

```
1-p_t <= ell = -log(p_t)
||p-e_t||^2 <= 2*ell.
```

For a same-answer group with mean probability vector `pbar`, least-squares
optimality of the mean then gives

```
mean ||p-pbar||^2 <= mean ||p-e_t||^2 <= 2*mean CE.
```

This applies to arbitrary finite logits, including the actual normalized
tied readout. No raw-state contraction assumption is needed. The loss controls
an upper bound on probability scatter, not monotone scatter at every step.
The constant two is convenient; no sharpness claim is made.

The counterexample has two same-label binary examples with logits `(theta,0)`
and `(2*theta,0)`, using a single shared trainable scalar. Their actual mean
CE has negative derivative everywhere. Every positive gradient step starting
at `theta>=0` lowers CE and increases the raw centered contrast energy
`theta^2/2`. The derivative and finite-step statements are proved. This is a
small shared-parameter classifier, not a model of a particular transformer step.

## Endpoint measurements

These are step-10000 results using cubic-answer orbits. Probability scatter is
the mean squared distance to each split's own group mean, summed over all 97
probabilities. Raw contrast scatter averages squared target-versus-competitor
deviations over inputs and the 96 wrong classes. These measures have different
units; their absolute magnitudes are not an architecture-quality ranking.

| Arm | Seed | Training CE | Training raw contrast scatter | Training probability scatter | Validation probability scatter |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 0.01734 | 2.718 | 8.965e-4 | 0.01129 |
| Real SwiGLU | 11 | 0.02200 | 2.497 | 2.717e-3 | 0.01667 |
| Complex diagonal | 7 | 0.001183 | 1.613 | 2.084e-7 | 0.003362 |
| Complex diagonal | 11 | 0.001226 | 1.962 | 2.791e-7 | 0.003768 |
| Complex radial product | 7 | 0.006213 | 2.047 | 3.356e-4 | 0.005307 |
| Complex radial product | 11 | 0.01490 | 2.106 | 5.898e-3 | 0.01723 |
| Complex plain product | 7 | 0.002190 | 1.509 | 3.428e-5 | 0.005069 |
| Complex plain product | 11 | 0.001562 | 1.408 | 1.173e-6 | 0.004630 |

All row and group objective bounds hold numerically. The report also retains
raw state scatter, final-normalized state scatter, normalized-logit contrast
scatter, missing target probability and squared one-hot error. Its centering
identity separates within-group scatter from the group mean's target error;
low scatter alone can describe a consistently wrong prediction.

Among the 16 adjacent training checkpoint intervals, CE falls in 12. In two
of those, raw contrast scatter rises: both diagonal seeds from 5000 to 10000.
Seed 7's raw scatter rises by 0.05323 while CE falls by 0.0002991; seed 11's
raw scatter rises by 0.02230 while CE falls by 0.00005259. Probability scatter
falls in both. These counts agree under sum and orbit grouping.

On validation, CE falls in 10 of the 16 intervals, with raw contrast scatter
rising in one: plain product seed 7 from 0 to 5000. These are endpoint
comparisons of actual training trajectories, not measurements of a single
gradient step or evidence that a specific optimizer term caused the change.
The other intervals, including increasing-loss intervals, remain in the report.

## Why this is not yet a generalization bound

The theorem uses loss on the **same rows** as the probability scatter. At both
plain-product endpoints, validation scatter exceeds twice the training loss:

| Seed | Twice training CE | Validation probability scatter |
| --- | ---: | ---: |
| 7 | 0.004380 | 0.005069 |
| 11 | 0.003123 | 0.004630 |

Thus substituting training loss into the validation inequality already fails
on these saved models. The valid validation bound uses validation CE, which
is a retrospective measurement. No independent statistical test is claimed.

This result identifies a geometry controlled by the actual training objective.
It does not explain how the optimizer reaches small loss or how that control
transfers to unseen inputs. The earlier
[training-state transport bound](../contrast_transport/README.md) still needs
control of its raw contrast terms, or a new transfer bound must work directly
with prediction probabilities. Complex multiplication is unchanged.

## Verification and reproduction

All 24 checkpoint identities exactly match the previous cohort. Every stored
train/validation correctness count reproduces; the maximum sampled scalar
oracle logit difference is `2.265e-14`. Maximum probability mass error is
`2.665e-15`, and maximum centering-decomposition error is `1.776e-15`.
The recorded maximum violations of missing-probability, one-hot-error and
within-group-scatter bounds are zero. The fixed audit tolerance is `1e-10`.
These are FP64 diagnostics, not verified floating-point bounds.

Six native tests cover known GEMM/SVD values, the frozen scalar source and
splits, exact small probability rows with shift invariance, a nonconstant
centering identity, and the actual-gradient counterexample with independent
finite differences. Formatting and strict Clippy pass. The complete diagnostic
took 4.23 seconds, excluding compilation, with four OpenBLAS threads and a
five-minute cap. These are author checks.

```sh
cargo test --offline --release --manifest-path experiments/loss_geometry/Cargo.toml
cargo fmt --manifest-path experiments/loss_geometry/Cargo.toml --check
cargo clippy --offline --release --manifest-path experiments/loss_geometry/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/loss_geometry/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw /tmp/loss-geometry-report.json
```

The report hashes the fixed protocol and all compiled source inputs, including
the shared source evaluator, split code and frozen scalar oracle. It records
every group and all 64 grouping/split/interval comparisons.
