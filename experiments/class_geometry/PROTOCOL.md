# Discovery and persistence of class-visible geometry

Task #199, frozen 2026-11-09 before new candidate measurements. The user
asks to pursue why training discovers and maintains the geometry, following
task #198. Preserve every prior result and failure. Rust/Kernite only.

## Candidate mechanism and scope

The training labels of `(a+b)^3 mod 97` have support A of size 33, with 64
unused output classes U. Determine this support from training labels only.
Decompose the class-centered embedding/readout into three orthogonal parts:
within-A centered rows, within-U centered rows, and the difference between
the two group means. Their class dimensions are 32, 63, and 1. Test whether
the actual trajectory builds the first component relative to the others,
and whether that change accompanies useful rank-32 training certificates.

For two unused classes, the partial CE readout gradient difference has
nonnegative inner product with the readout-row difference: exponential
probabilities have the same ordering as their logits. Thus its isotropic
gradient contribution damps unused-class pair dispersion. Prove the finite
identity and dissipative sign without assuming fixed features across steps.
Separate this partial derivative from the complete tied-embedding update.

The candidate is a supervised class-contrast mechanism, not an assumption
that all hidden states collapse or all spectra become flat. Low training CE,
one positive component, or an empirical association alone cannot establish
discovery, causal necessity or convergence. Retain failures of this mechanism.

## Fixed source and trajectories

Use the exact plain complex-product trainer and all source/PTX/Kernite,
AdamW, sampler, initialization and numerical conventions pinned by #198.
Replay original seeds 7 and 11 for all 10000 updates, checking their original
complete parameter/moment hashes at 0, 5000 and 10000. Add fixed replication
seeds 1, 3, 5, 13, 17 and 19 with the same 10000-update budget; no restarts,
seed selection, altered hyperparameters or training search. These are six
replications of the fixed candidate, not a claim of universal reliability.
Run in order 7,11,1,3,5,13,17,19. Keep a 1800-second cohort cap, recording all
failed, skipped and capped slots. Never change the cap after seeing outcomes.

Record every embedding state and every actual update's finite energy
attribution. Record training/validation accuracy and CE every 100 steps.
Evaluate geometry at 0,100,250,500,750,1000,1500,2000,3000,4000 and every
250 steps from 5000 through 10000 (31 states per seed). Report every state,
including all reversals. Persistence means performance on this fixed sampled
late window; no claim about intervening certificate values is licensed.

## Signed update accounting

For each actual minibatch, capture the partial readout gradient g_R, complete
tied-embedding gradient g_B, and actual updated moment buffers. Reconstruct
the embedding increment using decay, an isotropic readout term, adaptive
correction, tied-encoder credit, moment-memory difference and rounding:

```
Delta E = -eta E - eta alpha g_R - eta(D-alpha I)g_R
          -eta D(g_B-g_R) -eta D(m_hat-g_B) + rounding,
alpha = min_coordinate D > 0.
```

Use actual row-specific adaptive scales; do not treat D as commuting with
class centering. This per-minibatch decomposition intentionally differs from
#198's full-training/sampling split. Explain the distinction in the findings.
For each of the three fixed class projections, keep every signed radial
contribution `2<project(E),project(component)>` and the total squared increment
including all cross terms. Their sum must equal the actual energy change;
their cumulative sums must telescope over every prefix and the full run.
Audit the unused-class partial CE damping sign, retaining any floating-point
tolerance explicitly. Check instrumented versus ordinary source steps at
every declared geometry probe with a next update inside the training budget.

Prove finite component and multi-step identities and useful sufficient
dissipation/persistence conditions with nonzero witnesses. Check whether the
measured source meets those conditions before making any discovery claim.
Do not rename the measured desired outcome as an assumed theorem premise.

## Geometry and learning relevance

Keep #198's hidden-variance and class-centered readout-aware constructors,
training routes and opposite reference folds. Requested ranks are 0,32,33,128;
rank 32 is primary, rank 33 tests the extra between-group direction, and the
others are controls. Keep the cutoff, actual residuals, orthogonality defects,
feature/logit allowances and strictness tolerance unchanged. These diagnose
current-state training-to-query geometry; this study does not add forecasts
of an unevaluated future update. Query labels never select a basis or bound.

At each geometry probe measure the three class energies, total centered
energy, rank-32/33 readout tail energy, active-class versus full-readout
rank-32 subspace overlap, achieved ranks, retained energies and coverage.
Report first 50/90/99% accuracy and coverage crossings; over all 21 late
probes report minima, endpoints and regressions. Across the six replication
seeds report every result and success count, not selected examples.

Relate the class decomposition to finite low-rank structure explicitly:
within-A contrasts have at most 32 independent rows; adding the between-group
direction gives at most 33. Distinguish this structural fact from learned
energy concentration, alignment with each training span, margin sufficiency,
and optimizer discovery. Measure all links instead of equating them.

## Checks and evidence

Freeze source/configuration hashes and the full slot list before the run.
Use the independent FP64 source evaluator for score/margin checks, and verify
the source history/checkpoints. Save compact embedding traces, signed energy
tables, probe states, geometry bounds and source outcomes sufficient for an
independent verifier. Keep runtime, conditioning and all violations. Require
zero false certificates. Do not tune numerical allowances from outcomes.

Run meaningful native tests for class projections, finite/cumulative energy
identities, partial CE damping, and rank controls; strict clippy and format.
For Lean changes run full build, axiom audits, native INDEX generation and
forbidden checks. Commit source, protocol, commands and authored findings;
keep generated evidence ignored under root runs and .kanban/artifacts/199.
