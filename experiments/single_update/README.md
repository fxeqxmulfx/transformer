# Actual AdamW update, gradient paths and margins

Does the saved-moment discrepancy come from the tied encoder, the next
minibatch or momentum, and how does one actual update change the residual
interaction and decision margins? Task #186 answers this under the
[fixed protocol](PROTOCOL.md), continuing [task #185](../adaptive_interaction).

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

The next theoretical target is a condition controlling the worst competitor
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
are the exact-real references. This audit splits their forcing/drift terms
and retains FP32 rounding explicitly. It adds no new convergence theorem or
`sorry`; Lean sources and generated `INDEX.md` need no change.

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
```

The ten native tests cover independent CE derivatives, noncommuting class
scales and cross terms, explicit rounding, moving bases/features, and margins
that improve despite residual growth. GPU checks run for every measured
case. Output overwrite is refused. Reports, logs, generated copies and
compiled artifacts remain untracked; this README records authored conclusions.
