module

public import Transformer.Grokking.AdamW.FirstStep
public import Transformer.Grokking.AdamW.LossDirection
public import Transformer.Grokking.AdamW.LossDescent
public import Transformer.Grokking.AdamW.CenteredFamily
public import Transformer.Grokking.AdamW.FiniteThreshold
public import Transformer.Grokking.AdamW.Overshoot
public import Transformer.Grokking.AdamW.GradientError
public import Transformer.Grokking.AdamW.SecondStep
public import Transformer.Grokking.AdamW.MomentFeedback
public import Transformer.Grokking.AdamW.MomentLimits
public import Transformer.Grokking.AdamW.PartialReset
public import Transformer.Grokking.AdamW.ScalarStability
public import Transformer.Grokking.AdamW.ScalarLimits
public import Transformer.Grokking.AdamW.ScalarDenominator
public import Transformer.Grokking.AdamW.ScalarInputDissipation
public import Transformer.Grokking.AdamW.RetainedInputDissipation
public import Transformer.Grokking.AdamW.GeometricError
public import Transformer.Grokking.AdamW.ScalarBiasError
public import Transformer.Grokking.AdamW.MomentSquare
public import Transformer.Grokking.AdamW.SquareHistory
public import Transformer.Grokking.AdamW.SquareDirection
public import Transformer.Grokking.AdamW.SquareParameterBox
public import Transformer.Grokking.AdamW.ScheduledHistory
public import Transformer.Grokking.AdamW.ScheduledParameterBox
public import Transformer.Grokking.AdamW.PairEnvelope
public import Transformer.Grokking.AdamW.PairComparison
public import Transformer.Grokking.AdamW.PairContraction
public import Transformer.Grokking.AdamW.GradientClipping
public import Transformer.Grokking.AdamW.MomentumCounterexample
public import Transformer.Grokking.AdamW.CoupledFirstStep
public import Transformer.Grokking.AdamW.CoupledThreshold
public import Transformer.Grokking.AdamW.CurvatureBound
public import Transformer.Grokking.AdamW.CurvatureCounterexample

/-! Native AdamW update semantics and limits of transferring Euclidean
effective-flow explanations of grokking to the actual optimizer. -/

@[expose] public section
