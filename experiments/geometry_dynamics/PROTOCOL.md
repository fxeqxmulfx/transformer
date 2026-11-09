# AdamW motion and support-complete certificates

Task #203, fixed before new measurements on 2026-11-09. The user asks to
pursue the optimizer discovery/persistence gap, improve certificates, and
explain persistence during continued updates at a reduced learning rate.
Use native tools only. Do not retrain or select seeds from their outcomes.

## Optimizer and persistence

Derive the exact-real beta1=.9, beta2=.95 moment invariant from zero moments,
including a version with explicit rounding errors. Use it to bound the
bias-corrected adaptive direction, coordinate magnitudes and finite-horizon
motion under the actual nonnegative learning rates and decay .1. Connect
bounded motion to preservation of an already positive linear margin, keeping
feature movement and readout movement separate. Do not assume favorable
gradient signs, a full gradient, or a stationary feature table.

Audit every saved optimizer state in the three complete cohorts and every
embedding update in their traces. Use all eight seeds [7,11,1,3,5,13,17,19],
all 31 probes, and all 10000 updates. Report raw-moment invariant defects,
the analytic motion bound and actual movement for the declared hold interval
(updates 4001–8000). Keep all failures and loose bounds. A floating-point
check uses a fixed relative allowance 1e-6 for the squared-moment invariant;
it is numerical evidence, not a proof of the FP32 kernel. The existing
independent optimizer verification and its allowances remain unchanged.

The finite-horizon theorem is conditional on an existing positive margin.
It must not be described as proving its discovery, alignment with query
labels, or convergence at a constant positive rate. A bound that is too
large to certify persistence is reported as such.

## Certificate comparison

Retain the .001, .01 and early-settling cohorts from tasks #200–202: 744
states in cohort/seed/probe order, with no skipped failures or replacement
runs. Select the 33 active labels solely from the training set. In the new
basis, center each active readout row by the mean of the 64 unused rows,
then project into the same centered training-feature span and take SVD
ranks [0,32,33,128]. This adds the previously missing between-group direction
to the available span; requested rank 32 still truncates to 32 directions.
Report actual achieved rank. Rank 33 is a separate, larger-budget control.

Keep all 97 classes in every margin bound, the same training folds/routes,
residual and orthogonality penalties, and feature/logit allowances 1e-3/1e-3.
Freeze the bounds before reading query outcomes. Regenerate the original
active-basis margins byte for byte against task #202 before interpreting
each new result. Independently verify the emitted bounds against FP64 and
recorded GPU margins; require zero false certificates or bound violations.

Also report the maximum of the three valid margin lower bounds (original
all-class, active-centered, and new support-complete) for the same predicted
target. This envelope cannot lose existing certificates. It prepares three
bases and is not a single rank-32 construction or an efficiency result.
Report both single-basis and envelope outcomes, all ranks, every paired
endpoint, late minima, crossings, gained/lost certificates and regressions.

Use one fixed 1800-second cap for construction plus independent verification.
Preserve every capped or failed slot. Commit source, derivations, authored
findings and commands. Keep generated evidence ignored under root
`experiments/geometry_dynamics/runs/` and `.kanban/artifacts/203/`.
