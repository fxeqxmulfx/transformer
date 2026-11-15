# Shared decision geometry with an actual convex complex head

Task331 follows the failed shared-head improvement in330. That complete
comparison retains35/35 original success flags but only34/35 raw99 outcomes
in every mode. Fibonacci remains97.94–97.96%, with higher ordinary heldout
CE than319. Of forward's96 errors,91 have a true raw label already observed
in their routed block. The next question is whether shared decision geometry
can use observations across blocks to constrain their uncertain boundaries.
This is a hypothesis to test, not an identification or accuracy guarantee.

Lean first: `scripts/SharedGeometryAxioms.lean` audits CD8/CFt5/CM38 in
`review/flat_spectrum.md`, `ConvexDepth/{JointHead,ComplexHead,Fallback/Batch}`
and Mathlib's `ConvexOn.comp_affineMap`. For a fixed affine map from compact
parameters to every real/imaginary raw-class coefficient, ordinary CE remains
convex on the preimage of the complete coefficient budget. Original records
remain distinct even when their inputs repeat; normalized CE uses each actual
batch cardinality, including a short final batch. The audit constructs a
nonconstant map sharing two parameters across two complete four-state complex
blocks and both raw classes, with a nonzero fixed coefficient offset. An
explicit nonzero parameter point satisfies the expanded budget. The actual
Fourier features and expanded budget give the same logit bound as CD8.

Existing theorems suffice for that exact preimage contract. CM55 now adds a
sufficient weighted parameter domain suitable for native projection. For
`H[k,c] = B[k,c] + sum_p A[p,k,c] * theta[p]`, each weight bounds the complete
column mass `sum_k abs(A[p,k,c])` for every raw class. If the fixed offset
has radius `T` and `sum_p w[p]*abs(theta[p]) <= R`, the actual complete head
has radius `T+R`. The new Lean module proves this bound, domain convexity,
the actual logit bound and ordinary full/minibatch CE convexity. A nonzero
two-parameter example attains the charged radius. No `sorry` is introduced.
Native work must check every actual coefficient/logit. Numerical
projection, gradients and state replay remain separate implementation checks.
Both the feature library and the raw-class decoder/orientation map are fixed
during ordinary CE optimization. Discovering those maps is not jointly
convex and must use only training records. Flat component spectra supply
bounds, never missing raw labels.

The prospective component will compare a declared generic shared feature
library with the reviewed329 local-head control. Any choice of geometry,
decoder orientation or initializer must use original training records and
training-only folds or record exclusion. Raw labels remain categorical;
permuting their names must permute decoder outputs consistently. No task name,
arithmetic identity, generator answer or heldout label may choose a prediction.
Full and minibatch modes differentiate the actual ordinary full-vocabulary CE
through the fixed coefficient map; forward-only performs zero CE gradients.
All old/new candidate states, refusals, tails, work and CE increases remain.
Declare complete constructor/fold/gradient/projection/prediction and storage
costs before fitting, under64GiB, and verify exact saved-state reconstruction.

The reviewed329 source supplies the mathematical audit. Task330's
independent repeat has passed. Task332 depends on
both330 and331: it must freeze a separate all35 comparison and retain the
stronger319 result and330 regression before reporting any new99% attainment.
At this stage there is no native implementation or new accuracy measurement.

```sh
export LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
prlimit --as=68719476736:68719476736 -- lake build
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/SharedGeometryAxioms.lean
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
prlimit --as=68719476736:68719476736 -- scripts/native_index/target/release/lean-index --check
```
