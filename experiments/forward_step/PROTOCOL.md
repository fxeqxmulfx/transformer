# Frozen hidden-block direction and scale diagnostic

Task #194, 2026-11-08. Freeze before these outcomes. Task #193 found that
A-only local updates reduce their local loss but raise full-training CE at
all eight failed local endpoints. Separate direction and finite-step size
without extending training or changing the objective in this study.

## Fixed cohort and directions

Use the same **40 states**: four arms, seeds 7/11, original initialization
and all four pilot step-4000 endpoints. Pin and revalidate the original
source, pilot, and alignment-study artifacts. The previous tasks remain in
review; direct numerical rechecks are not independent review approval.
Use original scheduled update 1 or 4001 and the actual saved AdamW moments.

Compute the exact original local update once. Let `d` be its realized A
parameter displacement in independent real coordinates, `r=norm(d)`, and
`g` the current local A gradient pulled back through the signed complex
parameter expansion. The two diagnostic directions are:

* **adamw:** `d`, including the source moment history, adaptive scaling,
  decay and rounding already present in that realized displacement.
* **gradient:** `-r*g/norm(g)`, a current-gradient direction of the same
  independent-coordinate length. It includes no extra decay or moment term.

The second direction is not claimed to be zero-moment AdamW. Its norm uses
the actual AdamW step as a diagnostic reference, not a newly specified
standalone training optimizer. If either norm is zero, retain explicit
zero-direction entries without division or silently dropping states.

For both directions evaluate **five fixed scales**:
`1, 1/4, 1/16, 1/64, 1/256`. This is **400 parameter probes**. No additional
scale, line search, selected restart or continuation is permitted. Include
the unchanged state as the shared baseline. Do not summarize only the best
scale at each state.

Only A changes. Round `old_owner + scale*direction` to FP32, then rebuild
the signed complex aliases. E and B stay bitwise identical to the saved
state. Each probe starts from that state; no probe trains the next probe.
Probe moments are not advanced. Scaling an already rounded displacement
does not claim bitwise equivalence to changing AdamW's learning rate.

## Measurements and controls

For the next update minibatch, full training split and full validation split,
retain source CE, terminal pair loss, correct counts and accuracy before and
after every probe. Evaluate the frozen-boundary hidden objective L1 on the
update minibatch before/after, with incoming features and prototypes fixed.

Report the requested and realized independent-coordinate norms, rounding
error norm, cosine to the current local gradient, and pairing with the
complete terminal CE/pair gradients in each scope. Predictions and finite
remainders use the **realized** displacement. Report each scale separately.
For loss changes classify improvement below `-1e-7*(1+abs(before))`, worsening
above the positive threshold, and otherwise near-flat. This is a fixed
descriptive dead zone, not a certified roundoff bound. Also retain raw changes.
Accuracy differences are exact count differences. Do not turn validation
gradients or results into a training-time rule.

Check source/pilot baseline metrics and reproduce task #193's full local
update state hash. At adamw scale 1, require bitwise A equality with that
update and reproduce prior A-only metrics and local-objective evaluations.
Compare counts exactly, CE/pair means within 1e-12, and FP64 local-objective
values within 1e-12. Record the same-state origin; counterfactual directions
at another method's endpoint retain that method's moment history.

## Numerical checks and budget

Pin the previous compiled-source digests:
pilot `0f63edd42b7aa9968345bdd7b3815c9d3c9b098d378944449a18f174c2f37aa8`,
alignment `b2d670389e3dbfb36f1b95b42edfbe9fc90d3a94dfa8a95d06f148f8c034d3ee`.
Reuse the tested kernels, gradient aggregation and independent-coordinate
mapping. Verify finite tied parameters and bitwise unchanged E/B after
all probes; validate source AdamW state/moment ties. The unrounded
direction norms must agree within `1e-12*(1+r)` and the normalized negative
gradient cosine must be -1 within 1e-12 when nonzero.

For each nonzero direction, independently check normalized full-train and
full-validation derivatives of both losses in FP64: **320 scalar checks**,
central step 1e-4 and tolerance `2e-5+2e-3*max(abs(finite),abs(analytic))`.
Use the unchanged source graph and stable loss formulas. Check independent
versus expanded pairings for every realized probe to
`1e-12+1e-9*max(abs(independent),abs(expanded))`. Retain every error and zero
case; never relax a tolerance based on outcomes.

Use Rust/Kernite only. Record all GPU forward/backward batches, local
Jacobians, source optimizer steps, CPU reference evaluations, probes and
wall time. Cap the cohort at **600 seconds**, excluding compilation; retain
all 40 case slots and any failures/caps. Hash-check all input files after
the study. Preserve generated evidence outside the temporary worktree and
keep it uncommitted. This is attribution, not an efficiency benchmark.

A smaller successful one-step probe does not prove useful training or
generalization. Any subsequent training rule, learning-rate policy and
evaluation cohort must be specified and frozen separately. These saved
validation states have already informed the research direction.
