# Probability transfer from training states

Can training confidence and unlabeled query geometry bound held-out
probabilities? The [Lean extension](../../review/flat_spectrum.md#probability-transfer)
proves a conditional probability floor and loss ceiling. On the plain-product
checkpoints, the Euclidean penalty is too conservative to explain most of the
observed generalization: the sufficient condition for target probability above
one half covers only **13/4705 and 43/4705** validation inputs, despite source
accuracies of **99.66% and 99.68%**.

The [protocol](PROTOCOL.md) was fixed before measurement. The
[report](report.json) retains all 24 original checkpoints: four arms, seeds
7/11, and steps 0/5000/10000. Both training and validation splits, both
groupings and all five methods are reported. No model training or architecture
change was made.

## The transfer bound

Let `h` be a query state after its actual final RMS normalization, `h_i` a
normalized training reference, and `E_c` a tied readout row. For candidate
answer `t`, each competitor may choose its own reference `i(c)`. Set

```
U_c = log(p_i(c),c / p_i(c),t) + ||h-h_i(c)||_2 * ||E_c-E_t||_2
R = sum_(c != t) exp(U_c)
p_query,t >= 1/(1+R),       CE_query <= log(1+R).
```

Cauchy–Schwarz bounds the change in each relative logit. The sum includes all
96 wrong classes. The condition `R<1` guarantees target probability above
one half. The weaker condition `max U_c<0` guarantees a strict target winner.
The report distinguishes these two coverages. With per-logit execution error
at most `eta`, use `U_c+2*eta`; the probability condition then allows
`eta < -log(R)/2`. Reported allowances are budgets, not established numerical
error bounds.

The constructor receives raw operand pairs, normalized states, readout weights
and training-index/label pairs. It has no query-label, query-logit,
query-probability or target-error argument. Only training states are read out
inside it; query states enter through distances. Task-preserving routes
transfer an observed training label within a sum class or its orbit under
`1,35,61` modulo 97. It never cubes query sums to manufacture labels. All
training labels in a group must agree, and missing references cause abstention.

Training log odds are computed as logit differences. A training center is the
mean of normalized states; its log odds are mean training log odds, not the
log odds of averaged probabilities. Training queries exclude themselves from
both the anchor family and center. Both constructions are hashed before the
separate audit reads query labels and logits. Source/oracle verification
computes logits separately; this separation applies to bound construction.

| Fixed method | Reference selection |
| --- | --- |
| Center | Mean of eligible normalized training states |
| Nearest | Closest eligible training state; ties by index |
| Best anchor | One training state minimizing total bounded wrong odds |
| Anchor envelope | Minimum valid bound for each competitor over training states |
| Center + anchor envelope | Per-competitor minimum also includes the training center |

## Endpoint results

These are step-10000 validation results using cubic-answer orbits. Each
coverage percentage has denominator 4705. The last two columns use the same
center + anchor envelope, with the distinct conditions above.

| Arm | Seed | Source accuracy | Center probability coverage | Nearest probability coverage | Combined probability coverage | Combined argmax coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real SwiGLU | 7 | 99.43% | 0% | 0% | 0% | 0.04% |
| Real SwiGLU | 11 | 99.17% | 0% | 0% | 0% | 0.02% |
| Complex diagonal | 7 | 99.83% | 42.76% | 19.30% | 48.20% | 96.73% |
| Complex diagonal | 11 | 99.74% | 40.53% | 21.47% | 46.01% | 97.11% |
| Complex radial product | 7 | 99.68% | 0% | 0% | 0% | 0.09% |
| Complex radial product | 11 | 98.79% | 0% | 0% | 0% | 4.25% |
| Complex plain product | 7 | 99.66% | 0% | 0.26% | 0.28% | 18.21% |
| Complex plain product | 11 | 99.68% | 0.11% | 0.62% | 0.91% | 28.71% |

Every plain-product method is retained below; entries count inputs satisfying
the probability condition / the weaker argmax condition.

| Method | Seed 7, sum | Seed 7, orbit | Seed 11, sum | Seed 11, orbit |
| --- | ---: | ---: | ---: | ---: |
| Center | 0 / 664 | 0 / 644 | 4 / 1101 | 5 / 1103 |
| Nearest | 12 / 78 | 12 / 171 | 26 / 131 | 29 / 247 |
| Best anchor | 12 / 79 | 12 / 173 | 26 / 134 | 29 / 268 |
| Anchor envelope | 12 / 117 | 13 / 355 | 28 / 243 | 37 / 583 |
| Center + anchor envelope | 13 / 746 | 13 / 857 | 36 / 1222 | 43 / 1351 |

The full validation mean CE ceilings include every query, including failed
classification conditions. For the combined orbit envelope they are
**3.5043 / 3.2261**, while actual mean CE is **0.01974 / 0.01872**. These valid
ceilings are far too loose to explain the observed loss. The smallest positive
per-logit error allowances are 0.01576 / 0.00567; their existence does not
verify the arithmetic used by the diagnostic.

All initial checkpoints have zero probability coverage. No references are
missing in this cohort. There are no incorrect positive conditions in any
case, grouping, split or method. The report includes all intermediate states,
group statistics, failed conditions in the aggregate distributions, and full
split loss ceilings; no successful subset replaces a full-split average.

## Interpretation

The conditional training-to-query inequality is proved, but its practical
coverage is weak for the plain product. The diagonal arm has much larger
coverage under this geometry despite similar classifier accuracy. Coverage
therefore measures the usefulness of this sufficient condition, not model
quality. These normalized-state bounds differ from the earlier
[raw-state transport bounds](../contrast_transport/README.md); neither result
establishes a uniform improvement over the other.

The next obligation is tighter control in the relevant class-contrast
directions. A full Euclidean distance multiplied by a readout norm ignores
directional cancellation. This result supplies no evidence that complex
multiplication itself is incorrect or needs replacement.

Query model evaluation and known task symmetry are required. The result does
not prove optimizer discovery or predict generalization before evaluating
queries. The validation split has already been examined, so this is not an
independent statistical test. The generic linear-readout theorem applies to
all arms; the formal source bridge is specific to the plain complex product.
FP64 measurements at zero assumed execution error are not certified arithmetic.

## Verification and reproduction

All 24 checkpoint identities match the previous cohort. State, history and
split hashes verify, and every saved training/validation correctness count
reproduces. The largest sampled scalar-oracle logit discrepancy is
`2.265e-14`. Recorded maximum probability, CE and gap bound violations are
all zero; the fixed audit tolerance is `1e-8`.

Six native tests cover known linear algebra, all source arms against the
frozen scalar oracle, split identities, stable odds calculations, a positive
nonzero-distance witness, self exclusion and missing-reference abstention.
Formatting and strict Clippy pass. The complete diagnostic took 63.55 seconds,
excluding compilation, using four OpenBLAS threads under a five-minute cap.
These are author checks. Lean's full build and axiom audits pass for the 20
new theorems; the repository retains its 36 existing `sorry`s.

```sh
cargo test --offline --release --manifest-path experiments/probability_transfer/Cargo.toml
cargo fmt --manifest-path experiments/probability_transfer/Cargo.toml --check
cargo clippy --offline --release --manifest-path experiments/probability_transfer/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/probability_transfer/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw /tmp/probability-transfer-report.json
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ProbabilityTransferAxioms.lean
```

The report records the fixed protocol hash, all compiled source inputs and a
hash of each complete per-query construction. Shared source, split and oracle
files are included in the source hash. Earlier reports and checkpoints remain
unchanged.
