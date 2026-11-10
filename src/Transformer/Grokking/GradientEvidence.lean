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

/-! Actual-loss derivative checks and counterexamples for interpreting
gradient agreement as evidence of rule learning during grokking. `RecipeDecay` and
`RecipeRescale` count what the decay of a recipe removes within its run, and rescale a recipe
on a loss without a scale (`review/flat_spectrum.md`, *The rate of a recipe*). -/

@[expose] public section
