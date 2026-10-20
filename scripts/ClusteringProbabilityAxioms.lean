import Transformer.Perspective.Section3_ClusteringProbability

/-!
Axiom checks for arXiv:2312.10794v5, Section 4, `th:beta_small`,
`e:Ps0n` and their finite-time probability events.
No statement below may depend on `sorryAx` or a nonstandard extra axiom.
-/

open Transformer.Perspective

#print axioms isProbabilityMeasure_uniformTuple
#print axioms mem_Sset0_iff
#print axioms isOpen_Sset0
#print axioms measurableSet_Sset0
#print axioms sset0_probability_tendsto_one
#print axioms clustering_probability_small_beta
