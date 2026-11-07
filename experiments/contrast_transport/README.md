# Training-state transport on frozen cubic checkpoints

Can training states bound the actual classifier's held-out class contrasts
without measuring held-out target errors? This diagnostic obtains positive
conditions for a subset of queries: 991/4705 and 1406/4705 at the two plain
complex-product endpoints. Every positive case agrees with the saved classifier
and the label transferred from training. Most coverage comes from the simpler
training-center bound; covariance weighting adds only two queries per endpoint.

The [protocol](PROTOCOL.md) was fixed before measurement. All 24 original
checkpoints remain in [report.json](report.json): four arms, seeds 7/11 and
steps 0/5000/10000. No model was retrained. The earlier
[spectral](../flat_approx/README.md) and
[answer-contrast](../answer_contrasts/README.md) diagnostics are unchanged.

## Construction and theory

The constructor receives operand pairs, raw query/training states, tied
embedding weights and training-index/label pairs. Its interface has no query
labels, query logits or target errors. It groups inputs by either their sum
modulo 97 or the orbit of that sum under multiplication by 1, 35 and 61.
An observed training label supplies each group's candidate answer. Every member
must have a checked route `(a,b) -> (u*a+t,u*b-t)` from the training anchor.
Lean proves that these routes preserve the cube of the sum. A group without
a training anchor abstains; the constructor never cubes a query sum for a label.
This uses known task symmetry, without assuming model equivariance.

Let `mu` be a group's training mean, `H_i = h_i-mu`, `e = h_query-mu`, and
`w_v = E_target-E_v`. For arbitrary coefficients `a`, retain the actual residual
`r = e-H^T a`. The [Lean extension](../../review/flat_spectrum.md#training-state-transport)
proves the contrast bound

```
G = sum_i a_i^2 + ||r||^2/lambda
B_v = sum_i (w_v dot H_i)^2 + lambda*||w_v||^2
query contrast >= w_v dot mu - sqrt(G*B_v).
```

It also proves the Euclidean baseline `w_v dot mu - ||e||*||w_v||`, an explicit
counterexample to dropping the residual, and a positive example with a nonzero
residual. The actual plain-product source theorem retains the final RMS
denominator `D` and requires each lower bound to exceed `2*eta*D` when executed
logit errors are at most `eta`. The generic linear-readout inequalities apply
to all four arms; the source-specific Lean bridge covers plain complex product.

The diagnostic uses `eta=0`. It solves `(H^T H+lambda I)z=e`, sets `a=Hz`,
and computes `r` explicitly. Its four fixed scales are
`lambda = factor*max(trace(H^T H)/width,1e-12)`, with factors
`[1e-4,1e-2,1,100]`. Every competitor may select the best of these four valid
bounds and the Euclidean bound. The report also retains each individual scale
and the union of complete query conditions. A query passes only when all 96
wrong-class comparisons are strictly positive.

Both group constructions are completed and hashed before the diagnostic's
separate audit reads query targets and logits. The source forward evaluation
and oracle verification still compute logits; they do not supply them to the
constructor. This is an input-dependent condition that requires evaluating
the query state, not a prediction made without running the model.

## Endpoint results

Each row uses the same 4705 held-out inputs at step 10000. Coverage is the
fraction with a positive condition, not classifier accuracy. “Best” chooses a
valid bound separately for every competitor from the fixed five candidates.

| Arm | Seed | Source accuracy | Sum Euclidean | Sum best | Orbit Euclidean | Orbit best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 99.43% | 1.00% | 1.02% | 1.28% | 1.28% |
| Real SwiGLU | 11 | 99.17% | 0.74% | 0.74% | 0.91% | 0.94% |
| Complex diagonal | 7 | 99.83% | 80.04% | 80.04% | 80.15% | 80.15% |
| Complex diagonal | 11 | 99.74% | 78.41% | 78.41% | 78.36% | 78.38% |
| Complex radial product | 7 | 99.68% | 0.53% | 0.53% | 0.47% | 0.47% |
| Complex radial product | 11 | 98.79% | 10.56% | 10.63% | 11.65% | 11.65% |
| Complex plain product | 7 | 99.66% | 20.53% | 20.57% | 21.02% | 21.06% |
| Complex plain product | 11 | 99.68% | 28.67% | 28.74% | 29.84% | 29.88% |

The plain-product orbit Euclidean counts are 989 and 1404; adding the ridge
choices increases them to 991 and 1406. Factors `1e-4` and `1e-2` pass no
held-out query at either endpoint. Factor `1` passes 1 and 5 queries for seeds
7 and 11; factor `100` passes 948 and 1337, respectively. Selecting a whole-query condition gives
the same endpoint counts as selecting separately for each competitor.
The main improvement over the previous conservative bounds is centering on
training states, not a large benefit from covariance weighting.

All eight initial states have zero positive coverage. Coverage need not track
accuracy over training: seed 7 radial product has 4690 correct queries at both
5000 and 10000, while its best orbit condition falls from 441 to 22 queries.
Orbit grouping is not uniformly better than equal-sum grouping. Every earlier
step, individual method, failed condition and arm remains in the report.
These results do not establish an architecture ranking or call for changing
complex multiplication.

## Verification and remaining limits

All 24 checkpoint identities match the earlier diagnostics. Stored train and
validation correctness counts reproduce exactly. The largest sampled
batched/scalar FP64 logit difference is `2.265e-14`. All routes and observed
training labels agree; neither split has a missing anchor. Across every arm,
step, seed and bound, the recorded maximum lower-bound violation is zero and
there are no positive-but-incorrect cases. The largest normal-equation residual
is `8.317e-11`; the bound retains the actual reconstruction residual and does
not assume the solve is exact.

The report includes remaining per-logit error allowances, calculated as the
positive lower margin divided by `2*D`. Their minimum at the two plain-product
orbit endpoints is about `4.819e-4`/`2.807e-4`. These are available budgets,
not verified backend error bounds. The calculations are FP64 numerical evidence;
they do not certify checkpoint arithmetic or the original GPU execution.
The validation split has already been examined, so this is not an independent
statistical test. There is no theorem that training discovers the required
geometry, and most plain-product queries still fail these sufficient conditions.

Seven native tests cover the BLAS operations, two independent SPD right sides,
all source arms against the frozen scalar oracle, split/mask reconstruction,
a positive nonzero-residual example without a query label, failure in an unseen
direction, and abstention without a training anchor. Formatting and strict
Clippy pass. The complete diagnostic took 9.77 seconds, excluding compilation,
with four OpenBLAS CPU threads and a five-minute cap. These are author checks.

```sh
cargo test --offline --release --manifest-path experiments/contrast_transport/Cargo.toml
cargo fmt --manifest-path experiments/contrast_transport/Cargo.toml --check
cargo clippy --offline --release --manifest-path experiments/contrast_transport/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/contrast_transport/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw /tmp/contrast-transport-report.json
```

The report hashes the fixed protocol and all compiled source inputs, including
the shared evaluator, split and frozen oracle under `flat_approx`. It retains
both groupings, all five elementary bounds, both selection rules and per-group
training/residual energy summaries. The next obligation is to explain why
training makes query states close in the directions relevant to class contrasts,
with a substantially less conservative bound and explicit arithmetic error.
