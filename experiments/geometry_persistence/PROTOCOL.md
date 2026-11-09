# Observed finite-window persistence

Task #204 continues the user's main-gap and certificate requests. Freeze this
protocol before computing the new bounds. No training, optimizer changes,
Python, query-label fitting, or new seed selection is authorized by this study.

## Fixed evidence and inventory

Use all eight seeds `[7,11,1,3,5,13,17,19]` from tasks #200 (.001 main WSD),
#201 (.01 main WSD), and #202 (early settling). Input reports, parameter
states, query outcomes, and every-update embedding traces remain immutable.
Task #203's saved three-basis envelope supplies the starting certificates;
preparation costs of those bases are not included in this incremental audit.

1. For each of the 24 trajectories, inspect all 10000 recorded updates and
   certify every prefix of the fixed hold window, states 4000 through 8000.
   Derive the active/unused support only from observed training labels.
2. For each trajectory use exactly these eight `(anchor, endpoint)` pairs:
   `(4000,5000)`, `(4000,8000)`, `(5000,5250)`, `(5000,6000)`,
   `(6000,6250)`, `(6000,7000)`, `(7000,7250)`, `(7000,8000)`.
   There are 192 endpoint cases. The 4000-update horizon has one anchor;
   shorter horizons have different anchors. Report this imbalance.
3. Keep all 9409 queries and all 96 rivals. Report train and validation
   separately, every seed, horizon, and failed/inconclusive case. Do not turn
   the many correlated windows into independent training replications.

## Damped projection bounds

For each fixed class projection P, write the actual update as
`P E_(t+1) = a_t P E_t + F_t`, with `a_t=1-.1*eta_t`.
The effective forcing includes adaptive directions, tied-encoder effects,
moment history, and executed rounding. Compute it from the recorded update;
do not substitute a partial CE gradient or discard a component.

Set `q_0=1`, `R_0=0`, `q_(j+1)=a_j*q_j`,
`R_(j+1)=a_j*R_j+||F_j||`. Bound the distance from the damped anchor
by R, retaining projection arithmetic defects explicitly. Bound active norm
below by `max(0,q*A0-R_active)` and unused/between norms above by
`q*U0+R_unused`, `q*B0+R_between`. These yield a lower bound for the active
share of centered readout energy. Compare also the same construction with
one whole-embedding forcing radius, and retain the previous worst-case bound.
Report minimum lower share and counts above fixed 0%, 50%, 60%, and 2/3
thresholds. Check every bound against independently recomputed projections.
Report nonzero updates; a frozen path does not demonstrate moving persistence.

These are certificates for an observed path using its realized forcing
budgets, not forecasts from only the window's initial optimizer state.
They do not prove that all future gradients satisfy the same budgets.

## Finite parameter displacement and prediction transport

Use the existing task #196 combined Gram/RMS and signed-readout interval
construction unchanged, with `delta = parameters_endpoint - parameters_anchor`.
Those bounds allow an arbitrary finite displacement; no infinitesimal-step
assumption is introduced. The constructor may read endpoint parameters but
not endpoint query outcomes or endpoint model evaluations. Propagate the old
feature plus its directional tangent and finite remainder through the entire
network, including feature movement and all product terms.

For a training-derived target, let M be a saved old margin lower bound, g_c
the old FP64 rival-minus-target gap, and U_c a frozen future gap upper bound.
Then `M - max_(c != target)(U_c-g_c)` is a future margin lower bound.
Test the fixed old envelope ranks 32, 33, and 128. Report both positive
transported bounds and the stricter subset whose old bound was positive.
Keep the direct directional forecast as a separate control, not as a
geometry-transport certificate. Save every future gap upper bound and every
transported lower bound before independent endpoint evaluation.

Audit every pairwise FP64 bound and every resulting minimum-margin bound
against FP64 and the original GPU minimum margin. Retain the existing 1e-3
feature and 1e-3-per-logit allowances and scaled 1e-8 enclosure/strictness
tolerance. No tolerance is enlarged after observing results. Independently
reread emitted evidence, recompute old/new FP64 forwards without the forecast
constructor, verify transport arithmetic and hashes, and count false positives.

## Budget, proof, and reporting

Native CPU postprocessing and independent verification have a combined
1800-second cap; preserve all declared slots if the cap is hit. Rust unit
checks, compilation, and Lean proof development are outside this measurement
cap. Use root ignored runs and board artifacts so worktree removal loses no
evidence. Commit only source, protocol, commands, authored interpretation,
and generated INDEX.md for Lean changes.

Prove the damped forcing recurrence and norm/share consequences, plus the
scalar old-certificate transport rule. Check the complete Lean build, global
axioms, the new theorem list, native INDEX, and native tests/clippy/fmt. Keep
the unresolved discovery, nonlinear feature-alignment, and universal
multi-step optimization claims explicit. A bound that fails to certify is a
result, not a reason to alter the cohort or protocol.
