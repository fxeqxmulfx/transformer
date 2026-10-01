/-
# Heavy-tailed class imbalance

Formalization of arXiv:2402.19449v2, Kunstner, Yadav, Milligan, Schmidt,
Bietti, Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient
Descent on Language Models. The PDF and TeX sources are preserved locally
in papers/arXiv-2402.19449v2. Corrections are documented at each result.

Coverage:
* Section 2 / Appendix A.2 and A.6: dataset arithmetic and actual Euclidean
  normalized-GD, ordinary sign, and heavy-ball update definitions.
* Section 3.1: exact weighted-quadratic trajectories, frequency cancellation
  in sign updates, fixed-step oscillation, and a counterexample to the
  published instability threshold (the correct threshold is 2/pi_max).
* Section 3.2 / Proposition 2 / Appendix H: actual cross-entropy partial
  derivatives and Hessians, corrected gradient sign, initialization,
  exact class-assignment decompositions, quantitative data-bounded errors,
  and the correlation limit with explicit fixed-moment assumptions.
* Appendix H.1: exact Hessian block balance and diagonal dominance of traces.
* Section 3.3 / Theorem 3 / Appendix I, Lemmas 4--7: verified full matrix
  trajectories and uniqueness, the exact Lambert-W formula, finite-time
  gradient-flow loss bounds and Theta(1/(pi_k*t)), and corrected ordinary
  sign-descent loss and Theta(exp(-2*t)). The published exp(-c*t) loss and
  rate are refuted for three classes.
* Appendix H's Zipf argument: normalized finite class probabilities and
  exact prefix frequency bounds. Harmonic and ceiling asymptotics prove
  that the retained prefix has mass tending to one and its minimum
  rescaled class frequency tends to infinity.

Section 2's experimental learning curves and Section 4's possible extensions
to embeddings, intermediate features, and stochastic training are empirical
observations and research suggestions. They do not specify universal
mathematical guarantees, and no optimizer ranking is asserted here.
-/

import Transformer.Imbalance.Section2_Datasets
import Transformer.Imbalance.Section2_Algorithms
import Transformer.Imbalance.Section3_CorrelationEquivalent
import Transformer.Imbalance.Section3_HessianBlocks
import Transformer.Imbalance.Section3_LambertSolution
import Transformer.Imbalance.Section3_Objective
import Transformer.Imbalance.Section3_RateRefutation
import Transformer.Imbalance.Section3_Zipf
