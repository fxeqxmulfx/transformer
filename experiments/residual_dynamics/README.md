# Residual learning dynamics

Training CE controls the **partial readout-gradient forcing**, but neither the
theory nor the frozen trajectories support unconditional residual contraction.
Among 12 intervals with falling training CE, rank-32 orbit readout residual
energy rises in 11; the mean validation residual-norm product rises in 3.
Complex multiplication and the checkpoints remain unchanged.

The [Lean derivation](../../review/flat_spectrum.md#residual-learning-dynamics)
adds 23 proved theorems. It derives the actual fixed-feature CE gradient,
its projected-energy bound and a sufficient Euclidean descent/decay mechanism.
An actual Adam first-step counterexample lowers CE while creating a residual
outside the fixed training-feature span. A separate identity accounts for
movement of the subspace. These distinctions matter for the tied AdamW models.

## Frozen scope

The [protocol](PROTOCOL.md) was frozen before measurement. There is no training,
parameter update, intervention or new cohort.

| Dimension | Fixed choices |
| --- | --- |
| Arms | Real SwiGLU, complex diagonal, radial product, plain product |
| Seeds and steps | 7/11; 0, 5000, 10000: all 24 original states |
| Intervals | Both adjacent intervals per arm/seed: all 16 |
| Grouping | Same sum; cubic answer orbit |
| Requested training PCA ranks | 16, 32, 64; relative singular cutoff 1e-10 |
| Splits | All 4704 training and 4705 validation queries |
| Compute | FP64, four OpenBLAS threads, five-minute analysis cap |

Each answer group gets its own centered training-state basis B. The states
retain the actual final RMS denominator. Basis construction receives observed
training labels, checks their agreement, and checks multiplier/translation
routes; it uses no validation labels, logits or errors. Readout weights do not
choose the basis. Ranks are clamped to the available training rank.

Both splits use bases fitted on the full training group. Training residuals
therefore measure fit, unlike the opposite-fold training-query construction
in the earlier signed-transport diagnostic. Validation targets enter only the
separate source-accuracy audit. These residual measurements are not themselves
a new prediction certificate.

## What is differentiated and measured

With h_i fixed, G_c = mean_train (p_i,c - 1[c=y_i]) h_i is the actual CE
partial derivative of the final readout E. The tied embedding also affects
the encoder; G is not asserted to be its full gradient or an AdamW update.
For Q=I-B^T B, Lean proves

```
||QG||_F² <= 2*training_CE*mean_all_training ||Qh_i||².
```

The feature factor includes every training row, even when B belongs to one
answer group. Its evaluation uses the full training-state SVD with every
singular direction, as a sum of nonnegative squares. A separate fixed
32-training-row sample checks the same SVD-moment identity against direct
projection. All 9,360 group/rank/checkpoint bounds and sample checks pass.

For each query and wrong-class contrast w=E_c-E_t, the diagnostic records
the energy of r=Q(h-mu), the energy of u=Qw, and mean ||r||*||u|| over all
query/competitor pairs. Group summaries are weighted by query count; each
query has 96 competitors. Relative energies divide total residual energy
by total unprojected displacement/contrast energy. Training and validation
weight groups differently, so even readout averages can differ by split.
The report also retains training-state energy, centers, achieved ranks and
gradient forcing. Small projected gradients do not bound current residuals.

## Every fixed rank retains failed contraction

Counts below are out of the 12 intervals with falling training CE. Each triple
counts increases in **state energy / readout energy / norm product**. Increase
means more than 1e-8*(1+old mean value); all intervals, including the four with
rising CE, remain in the local report and its generated summary.

| Grouping | Rank | Training increases | Validation increases |
| --- | ---: | ---: | ---: |
| Sum | 16 | 0 / 10 / 1 | 0 / 10 / 1 |
| Sum | 32 | 0 / 11 / 0 | 0 / 11 / 2 |
| Sum | 64 | 0 / 11 / 0 | 0 / 11 / 3 |
| Orbit | 16 | 0 / 11 / 1 | 0 / 10 / 1 |
| Orbit | 32 | 0 / 11 / 1 | 0 / 11 / 3 |
| Orbit | 64 | 0 / 11 / 3 | 1 / 11 / 6 |

At orbit rank 32, validation norm-product increases occur for plain product
seed 7 and SwiGLU seed 11 over steps 0–5000, and diagonal seed 7 over
5000–10000. At rank 64, diagonal seed 11's validation state residual energy
also grows from 9.59094 to 9.90417 while training CE falls. No monotone
state-contraction claim across ranks is justified.

## Plain-product trajectory

These are validation averages using orbit rank 32; CE is the training loss.
Every orbit basis achieves rank 32 in the endpoint cases.

| Seed | Step | Training CE | State residual energy | Readout residual energy | Mean norm product |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 0 | 5.088676 | 83.09539 | 0.57716 | 6.89238 |
| 7 | 5000 | 0.008026 | 29.38814 | 1.86679 | 7.30021 |
| 7 | 10000 | 0.002190 | 19.48102 | 1.92573 | 6.00481 |
| 11 | 0 | 5.064702 | 82.59084 | 0.57227 | 6.84142 |
| 11 | 5000 | 0.004165 | 22.70206 | 1.89576 | 6.43812 |
| 11 | 10000 | 0.001562 | 20.22068 | 1.96683 | 6.18257 |

At step 10000, state residuals retain 47.12% / 50.21% of displacement energy,
and readout residuals retain 78.03% / 80.20% of contrast energy. The total
partial gradient energy is only 1.546e-5 / 7.694e-6. Across endpoint orbit
bases, projected all-training feature energy ranges from 44.18 to 72.78.
These bases do not annihilate all training features, so the exact pure-decay
premise is absent. The RD2 gradient bound holds but is loose: the observed
projected energy is about 1.49e-5 to 3.37e-5 times its upper bound.

The 5000–10000 decomposition separates movement with Q frozen from changing
Q at the new vector. Entries are changes in mean residual energy:

| Seed | Quantity | Vector movement | Basis movement | Total |
| --- | --- | ---: | ---: | ---: |
| 7 | State | -6.28034 | -3.62678 | -9.90711 |
| 7 | Readout | -0.02750 | +0.08645 | +0.05894 |
| 11 | State | +0.15112 | -2.63250 | -2.48138 |
| 11 | Readout | +0.00114 | +0.06993 | +0.07106 |

For seed 7, the readout residual decreases with the old basis frozen, but
increases after accounting for the new basis. For seed 11, state movement
with the old basis increases the residual; basis movement reverses that sign.
These are exact finite-difference decompositions up to numerical error,
not isolated optimizer updates or causal effects of weight decay.

## What remains open

Euclidean readout descent at fixed features has a derived decay-plus-forcing
bound, and directions absent from every training feature decay exactly with
the decay factor. Adam's coordinate rescaling can leave that feature span
even on the first step; learned features and their PCA basis also move.
Thus the saved tied-model trajectories do not instantiate the simple
fixed-feature contraction theorem.

The next learning obligation is to control the residual **interaction** under
the actual adaptive update and moving features, or to derive sufficient
conditions that control both changes. Falling CE and small instantaneous
partial gradients alone are insufficient. The earlier signed prediction and
orthogonality-defect terms remain necessary for classification bounds.
No optimizer-convergence or independent statistical-generalization result is
claimed.

## Verification and reproduction

All 24 state/history hashes match the prior signed-transport cohort. Split
hashes, metadata, scalar-oracle samples and saved correctness counts pass.
Maximum discrepancies are: scalar oracle 2.265e-14, final normalization
2.309e-14, sample SVD moment 1.137e-12, projector defect 1.887e-15, moving-vector
identity 7.883e-15 and energy decomposition 2.022e-14. The projected-gradient
inequality has zero positive excess. These are FP64 audits, not certified
arithmetic. The run completed in 24.56 seconds.

Nine native tests, formatting and strict Clippy pass. The tests cover actual
CE finite differences, repeated nullspace decay, an actual Adam span-escape
example, moving-basis decomposition, SVD moments and the frozen source oracle.
Full Lean build and axiom audits pass: 23 new theorems, 36 existing unfinished
statements, zero proofs resting on them and zero extra axioms.

Generated `report.json` and `summary.json` are ignored and local. Review
evidence is preserved under root `.kanban/artifacts/184/`; neither report is
committed. Source, protocol, dependency lock and summary generator are tracked.
The source/protocol SHA-256 digests are recorded in each regenerated report.

```sh
cargo test --offline --manifest-path experiments/residual_dynamics/Cargo.toml
cargo fmt --manifest-path experiments/residual_dynamics/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/residual_dynamics/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/residual_dynamics/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw experiments/residual_dynamics/report.json
jq -f experiments/residual_dynamics/summary.jq experiments/residual_dynamics/report.json > experiments/residual_dynamics/summary.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/ResidualDynamicsAxioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml
```
