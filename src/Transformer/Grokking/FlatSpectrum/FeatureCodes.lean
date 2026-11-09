module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.RankSeparation
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.NonuniqueWitness
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ApproximateRecovery
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.CubicQuotient
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.AnswerContrast
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TrainingTransport
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.CubicTransport
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.ProbabilityTransfer
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.SignedProbability
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.TransportForecastOutcomes

/-!
# Joint feature recovery and robust classification

Source: `review/flat_spectrum.md`, *Joint feature recovery and classification*.

Finite banks of unit-modulus complex features classify exactly when their positively
weighted coordinates jointly separate answers. The score gap is half the squared code
distance; a finite separating bank has a positive margin, and score errors smaller than
half that margin preserve the winner. Weighted feature error bounds score error.

The sum of the feature kernels' nuclear norms uniquely recovers a bank when its component
kernels satisfy the existing recovery theorems, including the arbitrary-rank certificate.
The recovered bank classifies all inputs if its answer code separates the labels.

For every prime and every natural exponent, powered division on nonzero operands has
rank-one character kernels, recovered by the same connected masks with perfect matchings.
A faithful character yields a positive decoding margin independent of the exponent.
An r-feature bank's real class-score matrices have rank at most 2r. For ordinary division,
one faithful complex feature classifies correctly although its identity-class indicator
has full rank p-1. These results distinguish matrix completion, feature separation, and
classification; they do not assert convergence of the actual Rotate90 training process.

Unique completion can also be weakened to recovery of class-score differences.
Under a strict dual certificate, it suffices that the score contrasts annihilate
unobserved tangent directions. Two explicit distinct global minima demonstrate
that all answers can agree while both feature matrices and absolute scores differ.

Quantitative recovery allows noisy fitting and a finite nuclear-norm objective gap.
Readout conditioning is finite whenever hidden tangent directions are harmless;
its magnitude, the certificate slack and the fitting error stay explicit.

Answer-fiber readouts preserve classification while ignoring distinctions between
latent inputs with the same answer. Their error budget is on the combined fiber
contrast. The cubic modulus-97 example proves that distinct full Fourier features
for sums 1 and 35 give identical complete cubic score vectors.

For arbitrary learned answer decoders, pooling contracts absolute error and
commutes with linear prediction. Unit mass allows contrast coefficients to be
centered; half their range bounds the combined error with a sharp constant.

Training-state transport bounds each contrast from observed training variation
and a query-state reconstruction, retaining its residual outside the training
span. Exact cubic input routes transfer observed labels to same-answer queries.
-/
