import Transformer.Grokking.AdamW.FirstStep
import Transformer.Grokking.AdamW.LossDirection
import Transformer.Grokking.AdamW.LossDescent
import Transformer.Grokking.AdamW.CenteredFamily
import Transformer.Grokking.AdamW.FiniteThreshold
import Transformer.Grokking.AdamW.Overshoot
import Transformer.Grokking.AdamW.GradientError
import Transformer.Grokking.AdamW.SecondStep
import Transformer.Grokking.AdamW.MomentFeedback
import Transformer.Grokking.AdamW.MomentLimits
import Transformer.Grokking.AdamW.PartialReset
import Transformer.Grokking.AdamW.ScalarStability
import Transformer.Grokking.AdamW.ScalarLimits
import Transformer.Grokking.AdamW.ScalarDenominator
import Transformer.Grokking.AdamW.PairEnvelope
import Transformer.Grokking.AdamW.GradientClipping
import Transformer.Grokking.AdamW.MomentumCounterexample
import Transformer.Grokking.AdamW.CoupledFirstStep
import Transformer.Grokking.AdamW.CoupledThreshold
import Transformer.Grokking.AdamW.CurvatureBound
import Transformer.Grokking.AdamW.CurvatureCounterexample

/-! Native AdamW update semantics and limits of transferring Euclidean
effective-flow explanations of grokking to the actual optimizer. -/
