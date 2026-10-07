import Transformer.GPTMini.Convex.Structured.Basic
import Transformer.GPTMini.Convex.Structured.Factorial
import Transformer.GPTMini.Convex.Structured.PointerTraining
import Transformer.GPTMini.Convex.Structured.ChannelMarginals
import Transformer.GPTMini.Convex.Structured.PointerValues
import Transformer.GPTMini.Convex.Structured.PointerControls
import Transformer.GPTMini.Convex.Structured.MarkovChain
import Transformer.GPTMini.Convex.Structured.MarkovTraining
import Transformer.GPTMini.Convex.Structured.MarkovMarginals
import Transformer.GPTMini.Convex.Structured.MarkovEmissions
import Transformer.GPTMini.Convex.Structured.MarkovObjective
import Transformer.GPTMini.Convex.Structured.StatePathProduct
import Transformer.GPTMini.Convex.Structured.SharedSlots
import Transformer.GPTMini.Convex.Structured.SharedPointer
import Transformer.GPTMini.Convex.Structured.Concentration
import Transformer.GPTMini.Convex.Structured.MarkovTeacher
import Transformer.GPTMini.Convex.Structured.ParityReference
import Transformer.GPTMini.Convex.Structured.MarkovConfidence
import Transformer.GPTMini.Convex.Structured.OutputCodes
import Transformer.GPTMini.Convex.Structured.OutputMargins
import Transformer.GPTMini.Convex.Structured.SharedReference
import Transformer.GPTMini.Convex.Structured.SharedInterface
import Transformer.GPTMini.Convex.Structured.SharedParity
import Transformer.GPTMini.Convex.Structured.DepthCompression
import Transformer.GPTMini.Convex.Structured.DepthScan
import Transformer.GPTMini.Convex.Structured.DepthReference
import Transformer.GPTMini.Convex.Structured.SharedDepth
import Transformer.GPTMini.Convex.Structured.Binding
import Transformer.GPTMini.Convex.Structured.RecallPositions
import Transformer.GPTMini.Convex.Structured.ChannelGaps
import Transformer.GPTMini.Convex.Structured.RecallPotentials
import Transformer.GPTMini.Convex.Structured.PointerDecoder
import Transformer.GPTMini.Convex.Structured.RawBinding
import Transformer.GPTMini.Convex.Structured.RecallEnergy
import Transformer.GPTMini.Convex.Structured.RecallGap
import Transformer.GPTMini.Convex.Structured.BindingInterface
import Transformer.GPTMini.Convex.Structured.SharedRecall
import Transformer.GPTMini.Convex.Structured.MixedHeads
import Transformer.GPTMini.Convex.Structured.MixedTraining
import Transformer.GPTMini.Convex.Structured.MixedConfidence
import Transformer.GPTMini.Convex.Structured.MixedMargins
import Transformer.GPTMini.Convex.Structured.MixedInterface
import Transformer.GPTMini.Convex.Structured.MixedBasis
import Transformer.GPTMini.Convex.Structured.RecallDataRoutes
import Transformer.GPTMini.Convex.Structured.RecallDataTargets
import Transformer.GPTMini.Convex.Structured.BasisTraining
import Transformer.GPTMini.Convex.Structured.TensorEmbedding
import Transformer.GPTMini.Convex.Structured.TensorRecovery
import Transformer.GPTMini.Convex.Structured.TensorState
import Transformer.GPTMini.Convex.Structured.TensorPointer
import Transformer.GPTMini.Convex.Structured.TensorHeads
import Transformer.GPTMini.Convex.Structured.TensorReadout
import Transformer.GPTMini.Convex.Structured.TensorCausalBlock
import Transformer.GPTMini.Convex.Structured.TensorTraining
import Transformer.GPTMini.Convex.Structured.TensorLikelihood
import Transformer.GPTMini.Convex.Structured.TensorBasisTraining
import Transformer.GPTMini.Convex.Structured.TensorStream
import Transformer.GPTMini.Convex.Structured.TensorObservations
import Transformer.GPTMini.Convex.Structured.TensorObservationInference
import Transformer.GPTMini.Convex.Structured.TensorDeferredFFN
import Transformer.GPTMini.Convex.Structured.TensorStack
import Transformer.GPTMini.Convex.Structured.TensorStackTraining
import Transformer.GPTMini.Convex.Structured.TensorStackInterface
import Transformer.GPTMini.Convex.Structured.TensorBasisModel
import Transformer.GPTMini.Convex.Structured.TensorBasisStackTraining

/-!
# Structured alternatives guided by raw Basis semantics

Source: complete Basis semantics at d640a91 and the finite log-sum-exp
argument in arXiv:2305.05465v6, §7. The actual compact shared-weight
embedding/attention tensor stack solves all six complete raw Basis
recipes at explicit finite weights. Correct data-supervised complete
sample/minibatch NLL is jointly convex in all unrestricted learned
parameters, with FFN fixed zero and readonly decoder/anchor axes.
Output-only CE, independent deep matrices and AdamW/FLOP results are
outside this mathematical prototype's scope.
-/
