module

public import Transformer.Grokking.GradientEvidence.BinaryObjectives
public import Transformer.Grokking.GradientEvidence.SharedTarget
public import Transformer.Grokking.GradientEvidence.ContrastExpansion
public import Transformer.Grokking.GradientEvidence.AdaptiveResidual
public import Transformer.Grokking.GradientEvidence.ContrastInteraction
public import Transformer.Grokking.GradientEvidence.AdaptiveMargin
public import Transformer.Grokking.GradientEvidence.MarginLossWitness
public import Transformer.Grokking.GradientEvidence.ForecastOutcomes
public import Transformer.Grokking.GradientEvidence.ForecastDirectional
public import Transformer.Grokking.GradientEvidence.ClassTrajectory
public import Transformer.Grokking.GradientEvidence.MainAdamCertificates
public import Transformer.Grokking.GradientEvidence.PersistenceCertificates
public import Transformer.Grokking.GradientEvidence.FutureAdamTube
public import Transformer.Grokking.GradientEvidence.CEMomentumControl
public import Transformer.Grokking.GradientEvidence.FeatureMetricControl
public import Transformer.Grokking.GradientEvidence.ProspectiveCE
public import Transformer.Grokking.GradientEvidence.CEGram
public import Transformer.Grokking.GradientEvidence.RecipeRescale
public import Transformer.Grokking.GradientEvidence.RecipeToward
public import Transformer.Grokking.GradientEvidence.TrainingTransport
public import Transformer.Grokking.GradientEvidence.CubicTransport
public import Transformer.Grokking.GradientEvidence.ProbabilityTransfer
public import Transformer.Grokking.GradientEvidence.SignedProbability
public import Transformer.Grokking.GradientEvidence.TransportForecastOutcomes
public import Transformer.Grokking.GradientEvidence.ObservableMargin
public import Transformer.Grokking.GradientEvidence.UpdateAttribution
public import Transformer.Grokking.GradientEvidence.ClassDecomposition

/-! Actual-loss derivative checks and counterexamples for interpreting
gradient agreement as evidence of rule learning during grokking. `RecipeDecay` and
`RecipeRescale` count what the decay of a recipe removes within its run, and rescale a recipe
on a loss without a scale (`review/diffgpt/steps.md`, *The rate of a recipe*). `RecipeToward`
is the same AdamW with a decay toward a fixed table (*The numbers of the tokens*, NT3).

Training-state transport (`review/actual_network/transport.md`) bounds each class contrast
from observed training variation and a query-state reconstruction, retaining its residual
outside the training span; exact cubic input routes transfer observed labels to same-answer
queries. `ObservableGeometry`, `ObservableMargin`, `ClassDecomposition` and
`UpdateAttribution` are from `review/actual_network/dynamics.md`. These eleven modules moved
here from `FlatSpectrum.FeatureCodes`. -/

@[expose] public section
