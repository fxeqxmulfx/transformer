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

/-! Actual-loss derivative checks and counterexamples for interpreting
gradient agreement as evidence of rule learning during grokking. -/

@[expose] public section
