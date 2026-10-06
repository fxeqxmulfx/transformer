import Transformer.GPTMini.Convex.Reparameterization.Completion

/-!
# Coordinate-independent head-class obstruction and a finite convex completion

New two-query, three-key shared-memory controls derived from
arXiv:2211.11052v1, §3, and Boyd and Vandenberghe (2004), §3.5.
No nonlinear parameterization of the unchanged scalar-head class that
covers two physical witnesses has convex likelihoods for every category.
Four unrestricted sequential utilities cover the entire positive finite
prediction table with convex likelihoods by enlarging that class.

These controls have fixed one-hot values and explicitly specified
log-probability readouts. Neither is a compact causal GPTMini replacement.
-/
