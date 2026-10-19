module

public import Transformer.Kinetic.Defs
public import Transformer.Kinetic.FourierDecay
public import Transformer.Kinetic.Hardy
public import Transformer.Kinetic.Section6_PowerSeries
public import Transformer.Kinetic.Section6_PositiveCoefficients
public import Transformer.Kinetic.Section6_ModeProfile
public import Transformer.Kinetic.Section6_HardyVariable
public import Transformer.Kinetic.Section6_HardyKernel
public import Transformer.Kinetic.Section6_GoursatSeries
public import Transformer.Kinetic.Section6_HardyPrimitive
public import Transformer.Kinetic.Section6_HardyRegularity
public import Transformer.Kinetic.Section6_HardyContraction
public import Transformer.Kinetic.Section6_HardyUniqueness
public import Transformer.Kinetic.Section6_ModeCurvature
public import Transformer.Kinetic.Section6_WeightedCoefficients
public import Transformer.Kinetic.Section6_SeriesProfiles
public import Transformer.Kinetic.Section6_WeightedProfiles
public import Transformer.Kinetic.Section6_WeightedConvexity
public import Transformer.Kinetic.Section6_InteriorMinimum
public import Transformer.Kinetic.Section6_OutputAggregation
public import Transformer.Kinetic.Section6_SourcePosition
public import Transformer.Kinetic.Section6_SourceLimits
public import Transformer.Kinetic.Section6_SoftLimits
public import Transformer.Kinetic.Section6_UShape
public import Transformer.Kinetic.Section1_RecencyEndpoint
public import Transformer.Kinetic.Riemann
public import Transformer.Kinetic.MeanField
public import Transformer.Kinetic.Section1_GridQuadrature
public import Transformer.Kinetic.Section1_SourceCells
public import Transformer.Kinetic.Fluctuations
public import Transformer.Kinetic.Section3_ThreeTokens
public import Transformer.Kinetic.Section3_AperiodicCovariance
public import Transformer.Kinetic.Section3_CumulantBounds
public import Transformer.Kinetic.Section3_Observables
public import Transformer.Kinetic.Section3_ThirdCumulant
public import Transformer.Kinetic.Codewords
public import Transformer.Kinetic.Accuracy
public import Transformer.Kinetic.CorrelationEquations
public import Transformer.Kinetic.Correlations
public import Transformer.Kinetic.UniformPrompt
public import Transformer.Kinetic.SoftAccuracy

/-!
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
The cumulant bounds of `lem:correl` need the source's periodic test: a test
on `ℝ` reads the winding numbers of the angles, and
`correlation_bounds_aperiodic_false` makes their covariance unbounded.
For the periodic test they hold (`correlation_bounds`).
-/

@[expose] public section
