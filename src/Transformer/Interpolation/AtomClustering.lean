/-
# Measure-to-measure interpolation — clustering to a single point mass

ArXiv:2411.04551v3, §2, `prop: targets.atoms`.

`Section2_DiracApproximation` proves stationary Dirac laws, explains why
W∞ must satisfy a support-radius bound, and derives point-mass
approximation from diameter collapse. `Section2_AttentionClustering`
proves the actual support diameter collapse and the complete existing
`clustering_to_atom` declaration for every original weak probability law.

The existing declaration uses support diameter, a spherical approximating
point and the constrained W∞ parameter. The source also states
geodesic-convex-hull diameter collapse, membership in the initial hull,
and a logarithmic W₂ hitting-time rate; these further claims are not
asserted by that declaration.
-/

import Transformer.Interpolation.Section2_DiracApproximation
import Transformer.Interpolation.Section2_AttentionClustering
