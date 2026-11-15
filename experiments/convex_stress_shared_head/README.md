# Refit the complete head on shared input features

Can ordinary raw-label cross-entropy improve the remaining original35 miss
when the same shared features are used on observed and unseen inputs? Reviewed
task319 leaves Fibonacci at97.9592% full-batch and98.0230% minibatch/forward raw
accuracy. Its observed CE is exactly unchanged: the old feature maps preserve
every observed source row. The new component will replace those features and
refit the complete head. There are no new model results yet.

The Lean audit precedes native implementation. CM42's
`ContextInterpolation.batchLoss_convex` quantifies over arbitrary fixed real
context weights, source features, input and raw-label maps, original record
IDs and batch. It has no Dirac-row hypothesis. CD8's `headLoss_convex` and
`headBudget_convex` likewise permit a freshly initialized complete head and
arbitrary fixed features. Thus ordinary full-vocabulary CE of the actual new
observed logits is convex in every real and imaginary coefficient, with the
per-class absolute-sum budget. Repeated inputs keep distinct record IDs and
the final minibatch uses its own cardinality.

CM51's `normalized_mass_le_one` and `normalized_features_bound` apply to any
fixed signed feature row divided by the larger of one and its absolute mass.
Bounded complex source coordinates therefore stay bounded. CD8's
`headLogits_bound` charges every coefficient and bounds each actual logit by
the head budget times that coordinate bound. A block selector fixed by an
earlier training-only representation is part of the fixed feature map; it is
not another factor optimized jointly with the head. Neither unitary source
features nor these bounds identify unseen labels.

The audit deliberately does not invoke CM42's `batch_loss_eq`,
`batch_fderiv_eq` or `batch_hasFDerivAt`: their observed Dirac-row premise is
absent. Observed CE, gradients, head coefficients and predictions can change.
CM45 permits a learned mixture only with its template head fixed; it does
not justify multiplying learned mixture weights by this fresh learned head.
No such learned factorization is proposed. These existing statements cover
the required algebra, convex domain and bounds; no new theorem is needed.
`scripts/SharedHeadAxioms.lean` checks their axioms and instantiates genuinely
shared observed rows, repeated records and a short final minibatch.

The prospective native feature family uses ordinal coordinates of the last
two input tokens and the fixed learned source block. Within a block it uses
`[1,u,v,u*u,u*v,v*v,0,0]`, with coordinates centered in[-1,1], absolute-mass
normalization and the complete8-state complex Fourier basis. All raw classes
and both coordinates of every coefficient remain explicit. The coefficient
budget is65536 per class across the entire model. Ordinal, quadratic and
block-sharing biases are declared; no arithmetic answer formula, task name,
generator or held-out score enters construction or selection.

Initialization will use only original training inputs and raw labels. A fixed
finite bank of ordinal directions can separate two observed class intervals;
training statistics can supply a quadratic fallback when they overlap. Every
candidate, selected threshold, numerical completion and rejected separator
must be recorded. This is explicit initialization/search work. Subsequent
full/minibatch updates optimize only the ordinary full-vocabulary CE of the
same complete logits used in prediction. Forward has zero CE gradients and
updates; initialization/search is still charged. No initializer's accuracy,
native floating-point refinement or optimizer convergence is certified by the
Lean audit.

Task329 next implements and checks the component on synthetic fixtures and
training-only inspections, then freezes source, protocol and binary. Task330
separately evaluates all35 tasks, preserving both old selection policies,
all full/mini/forward endpoints and transformer references. Every changed
prediction and observed CE tradeoff must remain visible. The99% target is per
task, not pooled accuracy, and no attainment is inferred before evaluation.

```sh
export LAKE_ARTIFACT_CACHE=false LEAN_NUM_THREADS=2
prlimit --as=68719476736:68719476736 -- lake build
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/SharedHeadAxioms.lean
prlimit --as=68719476736:68719476736 -- lake env lean -j 2 scripts/Axioms.lean
prlimit --as=68719476736:68719476736 -- scripts/native_index/target/release/lean-index --check
```

Generated evidence remains on storage under64GiB. Native implementation and
its reproduction commands will follow the completed Lean audit.
