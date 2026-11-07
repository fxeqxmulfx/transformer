# Signed directional transport

Preserving signed training contributions gives a much tighter conditional
bound for the plain complex product. At rank 32, the probability-above-half
condition covers **70.67% / 72.50%** of validation inputs for seeds 7/11; the
weaker argmax condition covers **96.60% / 97.43%**. Source accuracies remain
99.66% / 99.68%. The [Lean extension](../../review/flat_spectrum.md#signed-directional-transport)
proves the bound and its source bridge. This explains more saved predictions,
while the question of how training establishes this geometry remains open.

The [protocol](PROTOCOL.md) fixes all 24 original checkpoints, both sum/orbit
groupings, nine requested ranks and two bounds per rank. No training or model
change was made. The [summary](summary.json) retains every case, method and
split. The [complete compressed report](report.json.gz) additionally retains
every group, reference fold, singular spectrum and rank's geometry statistics.
Compression is lossless; the native program writes the uncompressed JSON.

## What changed in the bound

For normalized training states `T_i`, their center `mu`, query `x`, coefficients
`a_i` and competitor contrast `w=E_c-E_t`, keep the signed prediction:

```
r = x - mu - sum_i a_i*(T_i-mu)
prediction_c = mean_training_log_odds_c
             + sum_i a_i*(training_log_odds_i,c - mean_training_log_odds_c)
actual_query_log_odds_c = prediction_c + <w,r>.
```

The older covariance bound replaced the training contribution by an
absolute-value bound. Here it keeps its sign. The first comparison bounds
only the residual by `||r||*||w||`. The second uses training-basis vectors `B_j`
and readout coefficients `b_j=<B_j,w>`:

```
u = w - sum_j b_j*B_j
signed_upper_c = prediction_c + ||r||*||w||
dual_upper_c   = prediction_c + ||r||*||u|| + sum_j |b_j|*|<B_j,r>|.
```

The last term retains any orthogonality defect. Lean proves the inequality
for arbitrary basis vectors, and a counterexample shows that dropping this
term without orthogonality is false. A nonzero-residual witness has training
states `(1/2,0)`, `(3/2,0)`, query `(3,1/4)` and contrast `(-2,1)`: the signed
prediction is `-6` and the directional allowance is `1/4`.

For either valid upper envelope, `R=sum_(c!=t) exp(upper_c)` gives target
probability at least `1/(1+R)` and CE at most `log(1+R)`. `R<1` guarantees
target probability above half; `max upper_c<0` guarantees the correct argmax.
These remain separate conditions. All four source normalizations are retained.
Per-logit execution error `eta` adds `2*eta` to each upper bound; the diagnostic
uses eta zero and reports available allowances, not certified arithmetic errors.

## Fixed construction

Thin SVD uses centered training states only. Requested ranks are
`0,1,2,4,8,16,32,64,128`, with singular cutoff `1e-10*s_max`; each is clamped
to the available rank. Query projections determine reconstruction coefficients,
then the residual is recomputed explicitly from those coefficients and
training states. Readout weights do not select the basis. Each answer orbit
has its own basis; the model state still has 128 coordinates.

The constructor receives operand pairs, normalized states, readout weights
and training-index/label pairs. It never receives query labels, query class
logits, probabilities or target errors. Training log odds come from training
readouts. Query states enter through the training basis and geometric residuals.
Known cubic routes transfer observed labels, with agreement and route checks;
query sums are not cubed inside the constructor to create labels.

Validation queries use all eligible training references. Training queries use
the opposite half of alternating sorted group training indices, preventing
self reconstruction. Missing references cause abstention. Both complete
constructions are hashed before the separate target/score audit. Source and
scalar-oracle verification evaluate logits separately.

## Plain-product rank comparison

All entries count validation inputs out of 4705 at step 10000 using answer
orbits. “Probability” means the above-half sufficient condition. Every fixed
rank is shown, including failures. Rank zero reproduces the previous
normalized training-center baseline.

| Requested rank | Seed 7 signed probability | Seed 7 directional probability | Seed 7 directional argmax | Seed 11 signed probability | Seed 11 directional probability | Seed 11 directional argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 644 | 5 | 5 | 1103 |
| 1 | 0 | 0 | 782 | 7 | 7 | 1289 |
| 2 | 0 | 0 | 929 | 9 | 12 | 1500 |
| 4 | 0 | 2 | 1259 | 20 | 27 | 1888 |
| 8 | 18 | 38 | 2112 | 44 | 103 | 2678 |
| 16 | 188 | 576 | 3604 | 269 | 745 | 3986 |
| 32 | 1411 | 3325 | 4545 | 1546 | 3411 | 4584 |
| 64 | 3750 | 4602 | 4662 | 3985 | 4600 | 4665 |
| 128 | 4646 | 4675 | 4688 | 4652 | 4674 | 4689 |

At rank 32, signed reconstruction alone gives 29.99% / 32.86% probability
coverage. Removing represented readout directions raises it to 70.67% / 72.50%.
All validation orbit bases achieve rank 32. Their query residual norms remain
substantial: the mean ratio to the original displacement norm is
0.6863 / 0.7068. This improvement occurs with nonzero residuals.

The earlier [anchor-envelope bound](../probability_transfer/README.md) covered
0.28% / 0.91% under the probability condition. The new same-sum rank-32 bound
covers 2184/4705 and 2298/4705; using full answer orbits improves these to
3325/4705 and 3411/4705. Both groupings remain in the report.

The directional full-split mean CE ceilings are:

| Requested rank | Seed 7 ceiling | Seed 11 ceiling |
| --- | ---: | ---: |
| 0 | 3.96454 | 3.62769 |
| 16 | 1.64752 | 1.52135 |
| 32 | 0.63689 | 0.62324 |
| 64 | 0.14266 | 0.14513 |
| 128 | 0.02457 | 0.02522 |
| Actual source CE | 0.01974 | 0.01872 |

Every query contributes, including failed classification conditions. Rank 32
still gives a loose loss ceiling despite useful decision coverage.

## All arms at the endpoints

These use answer orbits and denominator 4705. The rank-64 column means up to
64 retained directions; small training groups can clamp the requested rank.

| Arm | Seed | Source accuracy | Rank-32 signed probability | Rank-32 directional probability | Rank-32 directional argmax | Rank-64 directional probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 99.43% | 0.02% | 1.08% | 36.49% | 58.24% |
| Real SwiGLU | 11 | 99.17% | 0% | 2.08% | 47.69% | 61.76% |
| Complex diagonal | 7 | 99.83% | 91.01% | 96.98% | 99.21% | 99.17% |
| Complex diagonal | 11 | 99.74% | 85.91% | 96.45% | 99.30% | 99.26% |
| Complex radial product | 7 | 99.68% | 0.98% | 9.63% | 65.55% | 86.97% |
| Complex radial product | 11 | 98.79% | 10.37% | 58.92% | 92.22% | 95.39% |
| Complex plain product | 7 | 99.66% | 29.99% | 70.67% | 96.60% | 97.81% |
| Complex plain product | 11 | 99.68% | 32.86% | 72.50% | 97.43% | 97.77% |

Coverage is not monotone with training or a model-quality ranking. For example,
SwiGLU seed 7 has rank-32 directional probability coverage 4035/4705 at step
5000 but 51/4705 at step 10000, while correctness remains 4696/4705 and
4678/4705. All intermediate and initial states remain in the report.

## Limits and the next obligation

Lean proves that zero state-reconstruction residual makes the signed prediction
equal the original score. Full-dimensional reconstruction can therefore
re-express the classifier without explaining its learning. At requested rank
128, three orbit groups in seed 7 and one in seed 11 are rank-clamped; the
minimum achieved ranks are 49 and 51. The high-rank results must be read with
this distinction in mind. The rank-32 improvement is the more informative
compressed comparison, though it still uses a separate basis per answer orbit.

The remaining theoretical obligation is to derive conditions under which
training makes these residual budgets small. The experiment supplies a useful
conditional bound at saved states. It does not prove optimizer convergence,
predict generalization before query evaluation, or provide an independent
statistical test. Known task symmetry is required. The generic linear-readout
theory covers all arms; the formal source bridge covers the plain complex
product. Complex multiplication and all original checkpoints are unchanged.

## Verification and reproduction

All 24 checkpoint identities match flat_approx's original cohort. Every
state/history/split hash and saved correctness count verifies. Maximum sampled
scalar-oracle logit discrepancy is `2.265e-14`. The largest observed probability,
CE and gap-bound discrepancies are `8.438e-15`, `1.998e-14` and `2.514e-13`,
all below the fixed `1e-8` audit tolerance. Incorrect positive counts and missing
reference counts are zero. These are FP64 audits, not verified arithmetic.

The maximum basis orthogonality error is `1.110e-14`, basis reconstruction
error `1.019e-13`, and retained defect penalty `7.313e-13`. Seven Rust tests,
formatting and strict Clippy pass. The diagnostic completed in 92.41 seconds
with four OpenBLAS threads under the five-minute cap. Full Lean build and
axiom audits pass for 16 new theorems; the 36 existing `sorry`s are preserved.
These are author checks, submitted for independent review.

```sh
cargo test --offline --release --manifest-path experiments/signed_transport/Cargo.toml
cargo fmt --manifest-path experiments/signed_transport/Cargo.toml --check
cargo clippy --offline --release --manifest-path experiments/signed_transport/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/signed_transport/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw /tmp/signed-transport-report.json
jq -f experiments/signed_transport/summary.jq /tmp/signed-transport-report.json > /tmp/signed-transport-summary.json
gzip -n -9 -c /tmp/signed-transport-report.json > /tmp/signed-transport-report.json.gz
LAKE_ARTIFACT_CACHE=false lake env lean scripts/SignedTransportAxioms.lean
```

The full report hashes the frozen protocol, all compiled source inputs and each
complete construction. `summary.jq` only formats the saved measurements; its
geometry means weight each reference fold by its query count. The compressed
report preserves the full unfiltered output, including individual group spectra.
