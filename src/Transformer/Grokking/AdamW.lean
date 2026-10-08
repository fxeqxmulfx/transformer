import Transformer.Grokking.AdamW.FirstStep
import Transformer.Grokking.AdamW.LossDirection
import Transformer.Grokking.AdamW.LossDescent
import Transformer.Grokking.AdamW.CenteredFamily
import Transformer.Grokking.AdamW.FiniteThreshold
import Transformer.Grokking.AdamW.Overshoot
import Transformer.Grokking.AdamW.GradientError
import Transformer.Grokking.AdamW.SecondStep
import Transformer.Grokking.AdamW.MomentumCounterexample
import Transformer.Grokking.AdamW.CoupledFirstStep
import Transformer.Grokking.AdamW.CoupledThreshold

/-! Native AdamW update semantics and limits of transferring Euclidean
effective-flow explanations of grokking to the actual optimizer. -/
