# Frozen local-update alignment diagnostic

Task #193, 2026-11-08. Freeze before diagnostic outcomes. The training pilot
at `c9995ba5` already failed; diagnose its rule without extending training or
selecting a replacement from outcomes. Task #190 awaits review; revalidate
its artifacts directly rather than calling them independently reviewed.

## States and scopes

Use all four arms and seeds 7/11: original step 0 plus all four step-4000
pilot endpoints (CE BP, pair BP, auxiliary BP, local pair): **40 states**,
with their actual saved moments. Pin the pilot compiled-source digest
`0f63edd42b7aa9968345bdd7b3815c9d3c9b098d378944449a18f174c2f37aa8`, endpoint
hashes, data/sampler/prototype identities and original source/PTX. Replay
each original/pilot state's full train/validation CE and correctness. Keep
inputs read-only and verify their hashes again at completion.

At each state use the next original scheduled minibatch, update 1 or 4001,
with the prefix-stable sampler and source hyperparameters. Every probe
starts from the same parameters and moments; none feeds another. Report
three scopes: update minibatch, complete train split, complete validation
split. Validation gradients are diagnostic information only.

Compute full terminal CE and pair-loss gradients in all scopes. Full-split
gradients weight each batch mean by its actual row count and accumulate in
FP64. Collect the four pilot update rules' minibatch gradients, direct pair
readout E, vocabulary E and terminal pair input-E path separately, using
the unchanged source FP32 kernels.

## Counterfactual updates

Apply one source AdamW step for each of seven current-gradient choices:

1. `ce-bp`: original CE and every gradient path.
2. `pair-bp`: terminal pair loss and every gradient path.
3. `auxiliary-bp`: all three pilot objectives and every path.
4. `local-pair`: the exact failed rule.
5. `local-no-vocabulary`: replace only local E's gradient with direct readout E.
6. `local-no-readout`: replace only local E's gradient with vocabulary E.
7. `zero-current-gradient`: zero current gradient, retaining moments and decay.

These **280 one-step probes** are not new training trajectories. Preserve
moment history within a state; another rule does not acquire the history it
would have learned elsewhere. AdamW is nonlinear in its current gradient:
the two E removals are counterfactuals, not an additive step decomposition.
Zero current gradient does not mean frozen parameters.

Measure each displacement's pairing with both terminal gradients in every
scope, on E, A, B and their sum. Use independent real coordinates for norms
and cosines: pull complex matrix gradients back as `(g_ar+g_ai,g_bi-g_br)`
and take displacements from their owners. E is already independent. Verify
that these pairings equal the expanded-coordinate pairings.

Report `D=gradient dot displacement` and descent cosine
`-D/(norm(gradient)*norm(displacement))`. Cosine >1e-3 indicates descent,
<-1e-3 ascent, and otherwise near orthogonal; zero norms have no cosine.
These are descriptive numerical categories, not rigorous roundoff bounds.
Report raw-gradient cosines separately. Also report each displacement minus
the zero-current-gradient displacement as a counterfactual contrast.

Measure actual CE, pair loss and accuracy after every complete probe in
all scopes. Additionally evaluate E-only, A-only and B-only portions of
the exact local displacement, keeping other parameters fixed. Retain
finite-change minus first-order residuals and cross-group interaction.
On the update minibatch also evaluate each block's frozen-boundary local
objective before/after: L0+partial readout for E, L1 for A and L2 for B.

## Checks, costs and limits

Reuse the pilot's tested local kernels unchanged. Check exact local-B versus
pair-B raw gradients, rounded readout+vocabulary E, and exact local-step
agreement with `pilot_step`. Check finite parameters, signed first-moment
ties and unsigned second-moment ties after all steps. Input E accumulated
from zero plus readout E can differ from the source addition order; require
reconstruction within `2e-6 + 2e-4*max(abs(x),abs(y))` and retain errors.

Independent FP64 central differences use step 1e-4 and tolerance
`2e-5 + 2e-3*max(abs(finite),abs(analytic))`. At every state use the first
eight rows of each split, two deterministic tied unit directions per E/A/B,
and both terminal losses: **960 scalar checks**. Also check the normalized
complete local displacement on both full splits for both losses:
**160 scalar checks**. Record zero displacements explicitly. Check expanded
versus independent pairings to `1e-12 + 1e-9*max(abs(x),abs(y))`. Retain all
errors and failed/capped cases; do not enlarge tolerances after outcomes.

Use Rust/Kernite only. Record full-gradient calls, evaluation batches,
optimizer probes, CPU finite-difference work and total wall time. Cap the
cohort at **600 seconds** excluding compilation, retaining all 40 case slots
if capped. Record GPU ownership before and after. This is numerical
attribution, not a speed comparison. Diagnostic backpropagation, validation
labels and counterfactuals must never be claimed as free training information.

Conclusions concern infinitesimal and one-step behavior at these states.
They neither identify an entire trajectory's cause nor prove a head removal
improves eventual validation accuracy. State the next research decision;
freeze any new training rule separately before measuring it.
