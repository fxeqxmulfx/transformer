/-
Formalization of:
  Duerinckx, Geshkovski, Rossi,
  "Kinetic theory for Transformers and the lost-in-the-middle phenomenon",
  arXiv:2605.09213v1.

A minimal causal decoder on the torus with the ALiBi positional bias. Its
empirical measure converges to a layered McKean-Vlasov equation, but retrieval
lives one order further out, in the time correlations: the cross-correlation
with the source token is `O(N⁻¹)`, solves a linearized equation forced through
the graphon `k_λ(σ,σ₀)`, and diagonalizes into Volterra-Hardy profiles that are
explicit in modified Bessel functions. Summing them gives the accuracy's
leading correction, and that correction is U-shaped — primacy, recency, and a
minimum in the middle.
-/

import Transformer.Kinetic.Defs
import Transformer.Kinetic.FourierDecay
import Transformer.Kinetic.Hardy
import Transformer.Kinetic.MeanField
import Transformer.Kinetic.Fluctuations
import Transformer.Kinetic.Codewords
import Transformer.Kinetic.Accuracy
import Transformer.Kinetic.Correlations
import Transformer.Kinetic.SoftAccuracy
