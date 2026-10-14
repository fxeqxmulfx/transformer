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

The printed uniform mean-field rate `eq:conv-rate-muNf` omits position
derivative control. `mean_field_rate_false` proves a smooth grid-test
counterexample, with actual stationary finite and continuum solutions.
The claimed unique finite-accuracy minimum is also refuted: its actual
integer source cells give an interval of minimizers at every minimum.
-/

import Transformer.Kinetic.Defs
import Transformer.Kinetic.FourierDecay
import Transformer.Kinetic.Hardy
import Transformer.Kinetic.Section6_PowerSeries
import Transformer.Kinetic.Section6_PositiveCoefficients
import Transformer.Kinetic.Section6_ModeProfile
import Transformer.Kinetic.Section6_HardyVariable
import Transformer.Kinetic.Section6_HardyKernel
import Transformer.Kinetic.Section6_GoursatSeries
import Transformer.Kinetic.Section6_HardyPrimitive
import Transformer.Kinetic.Section6_HardyRegularity
import Transformer.Kinetic.Section6_HardyContraction
import Transformer.Kinetic.Section6_HardyUniqueness
import Transformer.Kinetic.Section6_ModeCurvature
import Transformer.Kinetic.Section6_WeightedCoefficients
import Transformer.Kinetic.Section6_SeriesProfiles
import Transformer.Kinetic.Section6_WeightedProfiles
import Transformer.Kinetic.Section6_WeightedConvexity
import Transformer.Kinetic.Section6_InteriorMinimum
import Transformer.Kinetic.Section6_OutputAggregation
import Transformer.Kinetic.Section6_SourcePosition
import Transformer.Kinetic.Section6_SourceLimits
import Transformer.Kinetic.Section6_SoftLimits
import Transformer.Kinetic.Section6_UShape
import Transformer.Kinetic.Section1_RecencyEndpoint
import Transformer.Kinetic.Riemann
import Transformer.Kinetic.MeanField
import Transformer.Kinetic.Section1_GridQuadrature
import Transformer.Kinetic.Section1_SourceCells
import Transformer.Kinetic.Fluctuations
import Transformer.Kinetic.Codewords
import Transformer.Kinetic.Accuracy
import Transformer.Kinetic.CorrelationEquations
import Transformer.Kinetic.Correlations
import Transformer.Kinetic.UniformPrompt
import Transformer.Kinetic.SoftAccuracy
