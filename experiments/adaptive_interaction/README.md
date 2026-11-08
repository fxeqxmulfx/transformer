# Adaptive forcing and signed interaction control

The adaptive direction needs more information than the current partial
readout gradient. All 6,240 checked forcing decompositions and bounds pass,
but the objective-based bound is very conservative. At the plain-product
endpoints it exceeds the measured projected direction energy by factors
between 1.47 million and 7.49 million across the rank-32 orbit bases.

The [Lean extension](../../review/flat_spectrum.md#adaptive-interaction-control)
adds 23 proved theorems: actual buffered Adam equations, separate diagonal
mixing and moment-discrepancy terms, a loss-controlled contribution, signed
interaction drift, and sufficient interaction contraction with explicit
forcing/basis/feature budgets. Class contrasts retain different preconditioners
for competitor and target. None of the missing terms is assumed small.

## Which complex variant performs better here?

At step 10000, averaged over the two frozen seeds, the ranking is diagonal,
plain product, then radial product:

| Variant | Mean validation accuracy | Mean validation CE |
| --- | ---: | ---: |
| Complex diagonal | **99.787%** | **0.01248** |
| Complex plain product | 99.671% | 0.01923 |
| Complex radial product | 99.235% | 0.03953 |

Diagonal is the leading baseline on this cubic task. Plain remains a useful
true-complex-product comparison; the radial gate gives no aggregate advantage
in this cohort. Two seeds and one task do not establish a universal ranking.
The complex multiplication formula is unchanged, and these results do not
show an algebraic error in it. No architecture or training change was made.

## Frozen construction and timing

The [protocol](PROTOCOL.md) was frozen before measurement. All 24 original
states and all 16 adjacent intervals remain: four arms, seeds 7/11, steps
0/5000/10000. Both sum/orbit groupings and requested ranks 16/32/64 use the
same training-only centered PCA construction as task #184, with cutoff
1e-10 and rank clamping. All 4704 training and 4705 validation queries are
included. Training residuals measure fit to the full training group.

The basis uses observed training labels and verified task-preserving routes;
validation targets enter only the separate source-accuracy audit. Model
features retain the final RMS denominator. Original state/history/split,
source/protocol identities and correctness counts are checked.

Saved states contain parameters and both optimizer moment buffers. At a
positive saved step t, the audit computes

```
M = m_t/(1-0.9^t)
D_k = 1/(sqrt(v_t,k)/sqrt(1-0.98^t)+1e-8).
```

The source uses rate 0.001 after ten warmup updates and decay 1. These formulas
are evaluated in FP64 on saved FP32 buffers; they are not a bitwise replay of
the original CUDA update. The eight initial states have zero buffers and no
historical bias-corrected direction. Their geometry is retained with an
explicit not-applicable status; the 16 positive states supply 6,240 checks.

G is the full-training partial final-readout CE derivative at the current
checkpoint, with features frozen. The saved M incorporates earlier minibatch
gradients, the encoder path and momentum at earlier parameters. M-G combines
these discrepancies. It is not identified as just a momentum error, the
current encoder gradient, or a prediction of the next update.

## Adaptive forcing decomposition

For each training residual map Q=I-B^T B, the exact identity is

```
QDM = A + B + C
A = DQG
B = QDG-DQG
C = QD(M-G).
```

B measures noncommutation of projection and coordinate scaling. The formal
mixed-axis example has QG=0 but ||B||²=1/180. Constant diagonal scaling
commutes with any Q. The audit also retains the corresponding two-term
decomposition QDM=DQM+(QDM-DQM) using the historical moment directly.

RD2 and a coordinate bound dmax=max_abs(D) imply

```
||A||_F² <= dmax²*2*training_CE*mean_all_training ||Qh_i||²
||QDM||_F² <= 3*(||A||_F²+||B||_F²+||C||_F²).
```

Replacing the A energy by the first upper bound yields the objective-based
bound. Every training row enters its feature factor, evaluated by the
untruncated SVD moment from #184. The report retains all cross terms because
the individual energies need not add to the actual direction energy.

The following endpoint entries are **unweighted means over the 33 rank-32
orbit maps**. Energies sum over all 97 embedding rows and 128 coordinates;
they are not the query/competitor interaction averages below.

| Arm | Seed | Actual energy | A energy | B energy | C energy | Cross terms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SwiGLU | 7 | 275.167 | 13.892 | 1.063 | 257.902 | +2.311 |
| Diagonal | 7 | 44.427 | 23.252 | 1.572 | 22.242 | -2.639 |
| Radial product | 7 | 380.896 | 31.786 | 2.964 | 359.284 | -13.138 |
| Plain product | 7 | 129.205 | 15.478 | 1.750 | 115.799 | -3.822 |
| SwiGLU | 11 | 262.284 | 20.317 | 1.576 | 248.508 | -8.117 |
| Diagonal | 11 | 36.583 | 27.793 | 1.343 | 9.049 | -1.602 |
| Radial product | 11 | 193.909 | 265.400 | 143.227 | 521.523 | -736.240 |
| Plain product | 11 | 27.699 | 6.728 | 0.796 | 20.621 | -0.447 |

The discrepancy term is substantial, but these energies are not causal
percentages of an update. Radial seed 11 in particular has strong cancellation.
The source partial gradient alone does not characterize the saved direction.

## Signed interactions are smaller than the norm bounds, but not monotone

For every query and wrong-class contrast w, the audit measures
J=<Qw,Q(h-mu)>, its absolute value and square, and ||Qw||*||Q(h-mu)||.
All query/competitor pairs receive equal weight. The numerical defect
<w,Q(h-mu)>-J is retained; numerical projection is not treated as exact.

At the plain-product endpoints, rank-32 orbit validation mean |J| is
0.64599 / 0.64443 for seeds 7/11, compared with mean norm products
6.00481 / 6.18257. Mean |J| is only 10.76% / 10.42% of those bounds.
Measuring J at a query is a diagnostic, not a new training-derived certificate.

There are 12 intervals with falling training CE. The table retains every
fixed rank and both splits; entries count increases in **mean |J| / mean J²**.
Increase uses tolerance 1e-8*(1+old mean value).

| Grouping | Rank | Training increases / 12 | Validation increases / 12 |
| --- | ---: | ---: | ---: |
| Sum | 16 | 0 / 0 | 6 / 7 |
| Sum | 32 | 0 / 0 | 6 / 8 |
| Sum | 64 | 0 / 0 | 6 / 10 |
| Orbit | 16 | 0 / 0 | 6 / 8 |
| Orbit | 32 | 1 / 1 | 6 / 9 |
| Orbit | 64 | 1 / 1 | 7 / 10 |

At orbit rank 32, validation mean |J| increases over 0–5000 for both seeds
of SwiGLU, radial and plain products. The training increase is plain seed 7
over that interval. Thus monotone residual-interaction decrease is not a
universal proxy for learning. Classification also depends on the signed
reconstructed contrast; an improving classifier need not shrink this term.

For adjacent checkpoints, du=u1-u0 and dr=r1-r0 give the exact identity

```
J1-J0 = <du,r0> + <u0,dr> + <du,dr>.
```

Old/new vectors already include basis movement. For plain product over
5000–10000 at orbit rank 32:

| Seed | Old mean absolute interaction | New mean absolute interaction | Mean absolute change | Mean Cauchy change bound |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.74245 | 0.64599 | 0.67679 | 15.11524 |
| 11 | 0.69510 | 0.64443 | 0.50147 | 12.83720 |

Mean absolute change differs from the change of mean absolute value. The
three signed contributions, including the mixed term, are retained in the
local report. The drift inequality holds but is loose. These intervals cover
5000 updates; they are not single Adam steps or causal attributions.

## The sufficient theorem and remaining obligation

For one update with a=1-eta*rho, f=DM, u=Qw, r=Qx and new u',r',

```
|<u',r'>| <= |a|*|<u,r>| + Gamma
Gamma = |eta|*||Qf||*||r|| + ||(Q'-Q)w'||*||r|| + ||u'||*||r'-r||.
```

For a class contrast, f=D_c M_c-D_target M_target, with distinct row
preconditioners. Lean proves strict contraction if
Gamma < (1-|a|)*|<u,r>|, with a concrete nonzero example. It does not
assume this inequality for the data. The audit does not apply a single-step
condition to coarse intervals or substitute saved moments for next-step
moments.

The next obligation is to control the full-gradient/moment discrepancy and
feature/basis drift at the same update, while retaining signed contributions
relative to the reconstructed classification margin. Current partial-gradient
CE bounds and separate residual norms are insufficient. This remains a
conditional theory, not optimizer convergence or an independent statistical
generalization result.

## Verification and reproduction

The full run completed in 32.05 seconds with four OpenBLAS threads. All 24
checkpoint identities and source metrics match #184; all previous norm-product
averages agree within 1e-10. Maximum errors: forcing identity 1.244e-14,
forcing energy identity 2.320e-11, interaction-change identity 2.665e-14,
projection defect 3.908e-14, source oracle 2.265e-14, normalization 2.310e-14.
Every fixed numerical bound passes; Cauchy/change-bound positive excesses
are zero. These are numerical audits, not certified FP error bounds.

Eleven native tests, formatting and strict Clippy pass. Full Lean build,
23 targeted axiom checks and the native INDEX generator pass with 36 existing
unfinished statements, no proofs resting on them and no extra axioms.
Generated report.json and summary.json remain ignored and local. Review
evidence is preserved in root .kanban/artifacts/185 before worktree cleanup.

```sh
cargo test --offline --manifest-path experiments/adaptive_interaction/Cargo.toml
cargo fmt --manifest-path experiments/adaptive_interaction/Cargo.toml --check
cargo clippy --offline --manifest-path experiments/adaptive_interaction/Cargo.toml --all-targets -- -D warnings
cargo run --offline --release --manifest-path experiments/adaptive_interaction/Cargo.toml -- /path/to/rotate90_complex/runs/cubic-adamw experiments/adaptive_interaction/report.json
jq -f experiments/adaptive_interaction/summary.jq experiments/adaptive_interaction/report.json > experiments/adaptive_interaction/summary.json
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
LAKE_ARTIFACT_CACHE=false lake env lean scripts/AdaptiveInteractionAxioms.lean
LAKE_ARTIFACT_CACHE=false cargo run --offline --release --manifest-path scripts/native_index/Cargo.toml
```
