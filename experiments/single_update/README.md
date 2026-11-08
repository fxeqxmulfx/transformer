# Actual AdamW update, gradient paths and margins

Does the saved-moment discrepancy come from the tied encoder, the next
minibatch or momentum, and how does one actual update change the residual
interaction and decision margins? Task #186 answers this under the
[fixed protocol](PROTOCOL.md), continuing [task #185](../adaptive_interaction).
Task #187 adds the [joint margin/loss extension](MARGIN_PROTOCOL.md) below.
Task #188 adds [feature forecasts](FORECAST_PROTOCOL.md) frozen before
evaluating the updated network.

All 24 independent probes passed in 76.08 seconds on the RTX 3050 Laptop.
Each restores one original checkpoint in memory and takes its next scheduled
source AdamW step. All 24 resulting complete parameter/moment states match
ordinary source `train_step` **bitwise**, with the original PTX digest.
All 73 input files remain unchanged. The eight step-10000 probes are outside
the original training budget and do not revise its architecture ranking.

| Dimension | Fixed cases |
| --- | --- |
| Arms | Real SwiGLU, complex diagonal, radial product, plain product |
| Seeds and checkpoints | 7/11; 0/5000/10000 |
| Gradient measurements | Whole training set, readout/encoder paths, next scheduled minibatch, new moments |
| Geometry | Sum/orbit groups; training-only centers and bases; ranks 16/32/64; both splits |
| Outcome | Exact next update, signed interaction, reconstruction, residual budget and all competitor margins |

## What the audit found

The discrepancy has substantial encoder, sampling and moment-memory terms.
These terms cancel; their individual energies are not causal shares of the
update. For the plain-product **post-budget** probes, rank-32 orbit forcing
energies are below. Each number sums squared Frobenius norms over the 33
old fitted orbit projectors; this repeats the embedding in different bases
and is not one global model norm.

| Projected contribution | Seed 7 | Seed 11 |
| --- | ---: | ---: |
| Readout `DQR` | 520.40 | 226.48 |
| Commutator `QDR-DQR` | 58.72 | 26.80 |
| Encoder `QD(F-R)` | 18049.03 | 458.17 |
| Minibatch `QD(B-F)` | 25961.41 | 1883.97 |
| Moment memory `QD(M-B)` | 13334.11 | 1812.32 |
| Sum of individual energies | 57923.68 | 4407.75 |
| Sum of cross terms | -54136.84 | -3522.05 |
| Actual `QDM` energy | **3786.84** | **885.70** |

Here R/F are readout/full gradients of the entire training split at the old
parameters, B is the next-minibatch full gradient, and M/D are the corrected
first moment and preconditioner **after** that update. This aligns the time
indices that the earlier saved-buffer diagnostic could not separate.
Every class keeps its own diagonal scale before forming class contrasts.

Feature and basis movement also matter. At those plain-product endpoints,
the mean absolute validation contributions from readout-basis, centered-feature
and feature-basis movement are 0.01697/0.02323/0.02728 for seed 7 and
0.005824/0.008772/0.007783 for seed 11. Each individual optimizer contribution
is at most 0.000930/0.000349 respectively. These absolute contributions are
not additive magnitudes: signed cancellation remains in the exact identity.
The full mean absolute interaction changes per pair are 0.04008/0.01336.

Both plain-product probes increase the mean validation margin
(7.91200 → 7.91798; 8.35065 → 8.36498), while validation CE **increases**
(0.0197394 → 0.0199523; 0.0187227 → 0.0188030) and correctness remains
4689/4705 and 4690/4705. A larger average margin does not control the
difficult queries or the full loss. On 1082/1163 held-out queries, margin
improves even though the maximum residual budget grows.

For completeness, all post-budget validation changes are shown here; all
24 probes and every group/rank remain in the local report.

| Arm | Seed | Change in CE | Change in correct count | Change in mean margin |
| --- | ---: | ---: | ---: | ---: |
| SwiGLU | 7 | -0.00239187 | +1 | +0.051713 |
| SwiGLU | 11 | +0.00469569 | -16 | +0.022286 |
| Diagonal | 7 | -0.00010178 | 0 | -0.013259 |
| Diagonal | 11 | +0.00012119 | 0 | +0.002693 |
| Radial product | 7 | +0.00079865 | -1 | +0.005538 |
| Radial product | 11 | -0.00856179 | +13 | +0.045568 |
| Plain product | 7 | +0.00021294 | 0 | +0.005977 |
| Plain product | 11 | +0.00008035 | 0 | +0.014328 |

Across all probes, training CE decreases in 17/24 cases; rank-32 orbit
validation mean absolute interaction grows in 5 of those 17. Excluding
initial states, these counts are 9/16 and 1/9. In five trained probes, mean
validation margin increases while validation CE worsens. The radial seed-11
step-5000 probe loses 20 correct validation decisions; errors are retained.

This motivated a condition controlling the worst competitor
margin and difficult-query losses under joint forcing and feature/basis
movement. Separate norm bounds lose substantial cancellation. Neither
monotone residual contraction nor an average-margin increase is established
as a generalization mechanism. Complex multiplication is unchanged.

## Validation and limits

All 48 independent scalar-FP64 directional derivative checks pass; maximum
absolute error is 6.96e-8. Readout derivative error is at most 1.85e-7 on the
eight-row controls and 6.49e-9 for the weighted full-training gradient.
All 9360 projected decompositions pass. Maximum scaled forcing/energy
closure errors are 2.69e-15/2.47e-14; interaction/margin identity errors are
3.42e-15/2.12e-15. The largest executed-logit allowance is 1.08e-5.
No positive sufficient-margin condition misclassifies a query.

The margin diagnostic uses the **measured** directional interaction and
execution discrepancy. Its coverage is retrospective, and must not be
compared as an improved training-only bound with task #183's norm budgets.
Each orbit has its own fitted basis; this is not a single compressed network.
Only two seeds and one cubic task are represented. One next minibatch per
checkpoint does not estimate the distribution of future steps or prove
convergence. Attribution uses the declared additive order, not interventions
that independently change the encoder, sampler or momentum.

The existing Lean results
[`moving_update_interaction`](../../src/Transformer/Grokking/GradientEvidence/MovingInteraction.lean)
and [`buffered_contrast_interaction_le`](../../src/Transformer/Grokking/GradientEvidence/ContrastInteraction.lean)
are the exact-real references. The original #186 audit splits their forcing/drift
terms and retains FP32 rounding explicitly; it changed no Lean statements.

## Joint margins and difficult-query losses

Task #187 proves 35 additional Lean theorems: the finite joint step, its
mixed and rounding remainder, all-competitor margins, probability-weighted
actual CE change, subset loss bounds, and the row-specific Adam connection.
The [source derivation](../../review/flat_spectrum.md#joint-difficult-query-control)
and [Lean entry point](../../src/Transformer/Grokking/GradientEvidence/AdaptiveMargin.lean)
retain feature motion and execution error explicitly. A separate proved
counterexample shows that mean margin and mean CE can both increase even
when both predictions stay correct. There are no new `sorry` or extra axioms.

The native extension freezes each split's 471 lowest old-margin queries
before the next update, with canonical pair-index tie breaking. It checks
every competing class, allowing the worst competitor to change. Bounds use
ambient normalized features, without a fitted PCA basis. The signed terms
`<p,x> + <w,dx>` stay together; their mixed term, parameter rounding and
execution change are bounded separately. Log-sum-exp of the gap endpoints
encloses actual per-query CE, equivalently retaining old softmax weights.

All 24 probes pass in **83.75 seconds**. All 225816 query-update observations
and 21678336 competing-class pairs satisfy the bounds. Maximum scaled
identity and weighted-loss errors are 4.39e-15 and 2.79e-15; there are no gap
enclosure violations or false certificates. All 24 new state hashes equal
task #186's; all 73 original files, the source/PTX checks, 48 derivative
checks and 9360 geometry checks are preserved.

The table pools the 16 trained-state probes (steps 5000 and 10000), including
eight post-budget steps. Repeated queries across updates are observations,
not independent samples. Initial probes remain in the report.

| Query set | Observations | Actually correct after step | Joint certified correct | Separate certified correct | Joint certified CE direction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole training split | 75264 | 75218 | 75217 | 75020 | 74650 |
| Whole validation split | 75280 | 74952 | 74951 | 74463 | 74677 |
| Fixed validation bottom decile | 7536 | 7208 | 7207 | 6736 | 7489 |

Thus 99.20% of trained validation loss-change directions are certified,
leaving 603 inconclusive; the difficult subset has 47 inconclusive. Mean
validation margin interval width is 0.001595 jointly versus 1.071854 with
separate norms; for the fixed bottom decile it is 0.003102 versus 1.751311.
The separated intervals certify no direction of change **by construction**:
their symmetric allowances contain the old gaps, and hence the old margin
and loss. Their classification coverage is still useful to compare.

Every post-budget difficult-query loss outcome is shown below. These means
use FP64 CE of executed GPU logits, within the declared tolerance of the
source GPU aggregate. Each subset contains the same 471 queries before
and after the step.

| Arm | Seed | Old difficult-query CE | New difficult-query CE | Joint certified decrease / increase / inconclusive |
| --- | ---: | ---: | ---: | ---: |
| SwiGLU | 7 | 0.371318 | 0.358113 | 344 / 122 / 5 |
| SwiGLU | 11 | 0.531816 | 0.585803 | 300 / 168 / 3 |
| Diagonal | 7 | 0.102633 | 0.101534 | 224 / 247 / 0 |
| Diagonal | 11 | 0.114006 | 0.115232 | 291 / 178 / 2 |
| Radial product | 7 | 0.227908 | 0.236138 | 247 / 222 / 2 |
| Radial product | 11 | 0.459947 | 0.375267 | 368 / 100 / 3 |
| Plain product | 7 | 0.172718 | 0.174981 | 274 / 197 / 0 |
| Plain product | 11 | 0.167636 | 0.168661 | 375 / 95 / 1 |

For plain product, even this difficult subset's mean margin improves
(4.86297 → 4.87942 and 5.31682 → 5.33546), while its mean CE worsens.
The joint mean CE intervals, [0.174903, 0.175066] and
[0.168640, 0.168684], certify both increases. Correct counts stay 455/471
and 456/471; the worst competitor changes for 13 and 2 queries. Tracking
only average margins, even within a difficult subset, would miss the loss
regression. Individual probability-weighted conditions detect it.

These are **conditional local bounds**, using observed feature displacement
and execution error after the update. Tight retrospective coverage does not
make them pre-update forecasts or statistical generalization guarantees.
The next obligation is a useful training-derived bound on feature motion,
followed by validation on further seeds/tasks. Complex multiplication and
the original architecture ranking remain unchanged.

## Feature forecasts before the updated forward pass

Task #188 adds 22 Lean theorems for finite forecast propagation, anchored
normalization and sigmoid sensitivity, mixed product terms, and transport
from a feature ball to actual margin and loss intervals. The
[derivation](../../review/flat_spectrum.md#feature-forecasts) distinguishes
these proofs from verification of the native implementation or CUDA arithmetic.

The forecast uses the old network's value and directional derivative along
the proposed rounded Adam step. It propagates a Euclidean remainder through
every source layer. Linear layers retain their known mixed term; nonlinear
layers use their defect at the known nominal input and an anchored Lipschitz
bound. Nominal nonlinear values contribute to the radius only. They are
never propagated as updated features. All centers and radii are persisted
before any forward evaluation at the candidate parameters.

All **24 probes pass in 138.62 seconds**, including 225816 feature balls and
21678336 competing-class bounds. There are no allowance violations or false
certificates. Every new-state hash still matches task #186, and all 73 inputs
remain unchanged. All legacy derivative, geometry and joint interval checks
pass. The independent scalar central-difference check of the old tangent
has maximum scaled error 5.47e-9.

The fixed 1e-3 Euclidean FP32 feature allowance is respected: maximum old/new
discrepancies are 5.84e-5/6.32e-5. The fixed 1e-3 per-class readout allowance
is also respected, with maximum discrepancy 9.07e-6. These are audited
numerical assumptions, not formally proved CUDA error bounds. No allowance
was fitted or increased after observing outcomes.

**The forecasts preserve useful classification guarantees but do not certify
any loss-change direction.** All 225816 query-update loss intervals remain
inconclusive. For the 16 trained-state probes, including the eight post-budget
steps, coverage is:

| Query set | Observations | Actually correct after step | Forecast certified correct |
| --- | ---: | ---: | ---: |
| Whole training split | 75264 | 75218 | 69487 |
| Whole validation split | 75280 | 74952 | 68040 |
| Fixed validation bottom decile | 7536 | 7208 | 5555 |

This is 90.38% of trained validation observations and 73.71% of difficult
observations. Repeated queries are not independent samples. Mean validation
margin interval width rises from #187's retrospective 0.001595 to the
forecast's 4.947498; difficult-query width is 7.375966 versus 0.003102.
The retrospective 99.20% loss-direction coverage cannot be carried over
to the forecast's smaller information set.

The radii are conservative. At the plain-product endpoints, the ratio of
mean real-arithmetic radius to mean actual forecast error is 109.10 and
196.37. Every endpoint is retained below; feature means use both full splits,
while the last two columns count validation guarantees.

| Arm | Seed | Mean feature error | Mean feature radius | Certified correct / 4705 | Difficult certified correct / 471 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SwiGLU | 7 | 0.016101 | 1.904199 | 4094 | 237 |
| SwiGLU | 11 | 0.014914 | 2.004154 | 3996 | 207 |
| Diagonal | 7 | 0.005960 | 1.200560 | 4498 | 412 |
| Diagonal | 11 | 0.004368 | 0.619716 | 4683 | 449 |
| Radial product | 7 | 0.013180 | 1.741416 | 4358 | 271 |
| Radial product | 11 | 0.012664 | 1.983926 | 3971 | 174 |
| Plain product | 7 | 0.005255 | 0.573297 | 4664 | 432 |
| Plain product | 11 | 0.000640 | 0.125657 | 4690 | 456 |

This closes the dependence on measured future feature displacement for the
stated local conditions. It does not yet explain or guarantee loss descent.
The next target is to retain error directions through the block and readout,
instead of repeatedly converting them into scalar balls and multiplying by
Frobenius norms. This result gives no reason to change complex multiplication
or revise the original architecture ranking.

## Reproduction

Use Rust 1.99.1, CUDA and OpenBLAS. The manifest pins local Kernite at
`a1b6bc2c956e1a5f35960e26b209e1b6c213096c`, as the original experiment does.
`build.rs` checks the original source digest and generates the instrumented
copy in Cargo's ignored output directory. `ROTATE90_SOURCE` can override its
default `/home/misha/rust_projects/grokking/experiments/rotate90_complex` path;
the source bytes must match. The archived `source-b70811e.tar.gz` is another
way to recover those source inputs. Kernite's original workspace and binary
names are retained because both appear in CUDA symbol names. The startup
guard rejected earlier names before any numerical probe; it is not bypassed.
One scoped Clippy expectation preserves the pinned generator's public
`Random::next() -> u64` API. GEMM's original MIT notice is in `KERNITE_LICENSE`.

From this directory, with GPU access and a fresh output path:

```sh
cargo test --offline --release
cargo fmt --check
rustfmt --edition 2024 --check src/instrumentation.rs
cargo clippy --offline --release --all-targets -- -D warnings
cargo run --offline --release -- \
  /home/misha/rust_projects/grokking/experiments/rotate90_complex/runs/cubic-adamw \
  /tmp/single-update-report.json
jq -f summary.jq /tmp/single-update-report.json > /tmp/single-update-summary.json
jq -f margin-summary.jq /tmp/single-update-report.json > /tmp/margin-summary.json
jq -f forecast-summary.jq /tmp/single-update-report.json > /tmp/forecast-summary.json
```

The eighteen native tests cover independent CE derivatives, noncommuting class
scales and cross terms, explicit rounding, moving bases/features, and margins
that improve despite residual growth, plus joint cancellation, mixed terms,
competitor switches, fixed difficult-query selection and the mean-margin/CE
counterexample. GPU checks run for every measured case. The report retains
all per-query intervals and the fixed subset indices; `margin-summary.jq`
also produces the pooled counts above. With the retained original #186 report,
compare all state hashes using:

```sh
jq -r '.cases | sort_by(.arm,.seed,.step)[] | [.arm,.seed,.step,.audit.new_state_sha256] | @tsv' \
  report.json > /tmp/states-186.tsv
jq -r '.cases | sort_by(.arm,.seed,.step)[] | [.arm,.seed,.step,.audit.new_state_sha256] | @tsv' \
  /tmp/single-update-report.json > /tmp/states-187.tsv
diff -u /tmp/states-186.tsv /tmp/states-187.tsv
```

Lean checks run from the repository root with `LAKE_ARTIFACT_CACHE=false`:
`lake build`, `lake env lean scripts/Axioms.lean`, and
`lake env lean scripts/JointMarginAxioms.lean`. Regenerate `INDEX.md` with
`cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml`.
The full build and both axiom checks pass: 36 existing `sorry`, zero results
resting on them, and no extra axioms. `INDEX.md` includes all 35 new theorems.

For task #188 also run `LAKE_ARTIFACT_CACHE=false lake env lean scripts/ForecastAxioms.lean`:
all 22 additional theorems use only standard axioms; `INDEX.md` totals 21686
theorems, with the existing 36 `sorry` unchanged. The three new native tests
check source-arm tangents against the independent scalar oracle and feature
balls under nonzero steps, a cancelled product tangent with a nonzero mixed
term, and normalization with a nonzero input error near zero.

The executable also creates `/tmp/single-update-report.features.bin`; both
output paths must be fresh. Each case records its offset, byte count and
SHA256. A record begins with little-endian u64 row count and width, then
canonical pair order `(97*a+b)`, with 128 little-endian f64 center coordinates
and one real-arithmetic radius per row. The full binary's hash is in
`forecast_file`. Preserve this binary alongside the JSON before removing a
worktree. The JSON retains every query interval/error and all fixed subsets;
`forecast-summary.jq` regenerates the tables' source statistics.

Output overwrite is refused. Reports, logs, generated copies and
compiled artifacts remain untracked; this README records authored conclusions.
