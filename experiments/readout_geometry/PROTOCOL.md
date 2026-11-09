# Observable geometry and actual update attribution

Task #198, frozen 2026-11-09 before candidate measurements, following the
user's request to close both coverage and learning-mechanism gaps and extend
flat spectrum. Preserve task #197 and its negative temporal result.

## Theory target

Extend the feature geometry to class-centered readouts `C_c=E_c-mean(E)`.
Their Gram matrix measures observable error:
`sum_c <C_c,e>^2 = e^T (C^T C) e`. Common score shifts and vectors in the
readout kernel are invisible to every class comparison. Derive a finite
contrast/margin bound and an explicit nonzero invisible-error witness.
Do not infer an optimizer-discovery theorem from a small observable error.
Keep the exact signed source-step attribution and every mixed term.

## Frozen comparisons

Replay exactly task #197's two plain-product seeds, 13 states per seed,
source/PTX/Kernite, optimizer, sampler, labels and one-step probes. All original
checkpoint and next-update hashes must match. Keep the 1200-second cap and
all 26 slots; no seed or state selection and no training search. The task
remains `(a+b)^3 mod 97`, separate from the original `a^3+ab` obstruction.

Compare two basis constructors at requested ranks `0,32,128`:

1. **Hidden variance:** the unchanged centered-training-state SVD from #197.
2. **Readout-aware:** project the class-centered embedding rows onto the
   available centered-training-state span. Take the leading right singular
   vectors of those projected rows. Reconstruct each chosen basis vector
   using the training SVD pseudoinverse, then compute query coefficients and
   actual residuals explicitly. Do not assume the pseudoinverse, projection
   or basis orthogonality is exact; retain the original defect terms.

Use the unchanged singular cutoff `1e-10*s_max`, clamping every requested
rank to the available span and readout rank. Rank 32 is the primary matched
comparison. Rank 0 checks identical center behavior; rank 128 is a limit
control, not low-rank discovery. Each orbit/reference fold still has its own
basis. The basis uses old training states and learned embeddings, never
query labels, query scores, future states, or measured target errors.

Keep the training routes, opposite training folds, all 97 competing labels,
old geometry, combined bridge, directional bridge and direct comparator.
Freeze both constructions before future evaluation. Use exactly #197's
feature/logit allowances and scaled tolerance. No post-outcome tuning.

## Actual optimizer attribution

At each probe capture the old full-training partial readout derivative, the
old full-training tied-embedding derivative, the actual source minibatch
derivative, and the updated Adam moment buffers. Audit the full-training
readout derivative against its independent analytic calculation. Reconstruct
the embedding increment as the signed sum of decay, diagonally scaled
readout gradient, encoder contribution, minibatch difference, moment-memory
difference, and parameter rounding. Preserve row-specific preconditioners.
The instrumented step must match ordinary source train_step byte for byte.

For each embedding component, propagate its old directional feature tangent
through the complete source graph. A seventh component contains the actual
projection-matrix increment. The component tangents must sum to the original
complete tangent. Attribute the predicted competitor-gap change to each
component's direct readout and feature effects, and retain the total mixed
readout/feature term. The nonlinear forecast remainder remains separately
bounded. These are directional contributions to one actual update, not seven
independently trained trajectories or a claim of causal long-run necessity.

Freeze the old worst competitor per query for attribution summaries; retain
all-class bounds for correctness and CE. Report component signs and magnitudes
on both complete splits and whether full-training readout credit alone has
the same sign as the actual adaptive step. A different future worst competitor
does not invalidate the all-class certificate. Keep every negative result.

## Outcomes and checks

Repeat the predeclared 50%, 90%, 99% coverage/accuracy crossings at the same
probe resolution. Report all 26 cases, old/new accuracy and loss, both basis
families and all ranks, direct forecast coverage, achieved ranks, residuals,
conditioning and preparation cost. Compare with #197 explicitly. Improvement
from a new basis would identify avoidable certificate slack; it would not by
itself explain why AdamW discovers that basis or predict eventual convergence.

Run all source/allowance/layer/interval audits and independent future FP64
verification. Require zero false certificates; retain all failures and all
input hashes. Test a low-variance but readout-relevant direction, explicit
residuals/defects, rank clamping, no-reference abstention, tangent linearity,
and exact optimizer-component closure. Prove new theory in Lean with witnesses;
run full build, axiom audit, generated INDEX and forbidden checks for src changes.
Use Rust/Kernite only. Keep outputs ignored under root runs/artifacts; commit
protocol, source, reproducibility commands, authored findings and generated INDEX.
