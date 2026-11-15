# Complete shared-head comparison

Task330 asks whether the reviewed329 shared-feature head improves raw-label
accuracy and ordinary cross-entropy on the complete original35-task suite.
The target is99% per task, with full batch, minibatch and forward-only results
reported separately. The reviewed319 suite remains34/35 raw99 in every mode;
no new heldout result is claimed by this prospective specification.

The implementation must retain both105-choice policies, every unchanged choice,
all old endpoints, conventional/depth/DiffGPT references and failed attempts.
Eligibility is the already selected signed-kernel representation, fixed before
new scores. Both prior/supplement roles retain their original input routing.
For each role, a block is covered exactly when an original training record
routes there. Covered inputs use the complete329 head; uncovered inputs use
the complete saved319 predictor. The gate uses no task name or heldout labels.
All new scores remain visible even when they regress.

Lean first: `scripts/SharedHeadFallbackAxioms.lean` audits CM38 in
`review/flat_spectrum.md` and `ConvexDepth/Fallback/{Basic,Domain,Batch}`.
The finite disjoint coordinate sum implements the exact selected logits;
fixing the entire parent block is an affine constraint. Both stored coefficient
budgets are charged. Ordinary CE is convex in every candidate coefficient.
Every original training record is covered, so the routed objective and its
derivative agree with the actual shared-feature objective, including repeated
record IDs and a shorter final minibatch. The audit uses non-Dirac rows and
an unsupported query with nonzero parent logits. All seven printed declarations
use only `propext`, `Classical.choice`, and `Quot.sound`.

These existing results suffice; no new theorem, assumption or `sorry` is added.
Coverage does not prove unseen-label correctness. The two roles are trained
on their own original records; their summed standalone objectives are convex.
The actual composite predictor's observed CE must also be reported, without
claiming that independent role updates minimize a routing-conditioned objective.
Full and minibatch change all real/imaginary head coordinates; forward-only
retains the initializer with zero CE gradients. Flat spectrum supplies the
feature/coefficient bound, not the missing labels.

Before any heldout score, freeze the complete adapter, source closure, binary,
inputs, eligibility, shape-derived construction/training/metric/audit costs
and disk-sealed coefficients. All processes use at most64GiB. Preserve exact
old/new raw logits, native counts, field readouts, ordinary CE changes,
fixes/regressions, and task298's lower-CE tradeoffs. A fresh complete run and
saved-state replay must reproduce all artifacts. Generated outputs stay local
on storage; source, protocols and authored interpretation are committed.
