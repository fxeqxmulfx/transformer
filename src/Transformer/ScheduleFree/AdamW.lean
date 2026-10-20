module

public import Transformer.ScheduleFree.AdamW.PaperComparison

/-!
# The authors' stability revision of Schedule-Free AdamW

Formalizes facebookresearch/schedule_free commit
`59ed9b662d5ec5799aded7390b2482758423653c` (version 1.3), which moves the
second-moment bias correction out of the learning rate and into the denominator.
These formulas also describe commit `70785b53e778d0e872c0bbb75ff4ee54ee10c291`
with default inner momentum zero. Includes the general code averaging powers,
running maximum rate, zero-total fallback, independent two-buffer/reference
trajectories and their equivalence, evaluation recovery, denominator/variance/
weight invariants, and exact comparisons with arXiv:2405.15682v4, §4.4,
Algorithm 1. The revision changes decay, epsilon scaling and averaging weights.
No general optimization convergence or floating-point stability theorem is claimed.
-/
