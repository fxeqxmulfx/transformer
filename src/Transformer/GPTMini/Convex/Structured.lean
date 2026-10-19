module

public import Transformer.GPTMini.Convex.Structured.Basic
public import Transformer.GPTMini.Convex.Structured.Factorial
public import Transformer.GPTMini.Convex.Structured.PointerTraining
public import Transformer.GPTMini.Convex.Structured.ChannelMarginals
public import Transformer.GPTMini.Convex.Structured.PointerValues
public import Transformer.GPTMini.Convex.Structured.PointerControls
public import Transformer.GPTMini.Convex.Structured.MarkovChain
public import Transformer.GPTMini.Convex.Structured.MarkovTraining
public import Transformer.GPTMini.Convex.Structured.MarkovMarginals
public import Transformer.GPTMini.Convex.Structured.MarkovEmissions
public import Transformer.GPTMini.Convex.Structured.MarkovObjective
public import Transformer.GPTMini.Convex.Structured.StatePathProduct
public import Transformer.GPTMini.Convex.Structured.SharedSlots
public import Transformer.GPTMini.Convex.Structured.SharedPointer
public import Transformer.GPTMini.Convex.Structured.Concentration
public import Transformer.GPTMini.Convex.Structured.MarkovTeacher
public import Transformer.GPTMini.Convex.Structured.ParityReference
public import Transformer.GPTMini.Convex.Structured.MarkovConfidence
public import Transformer.GPTMini.Convex.Structured.OutputCodes
public import Transformer.GPTMini.Convex.Structured.OutputMargins
public import Transformer.GPTMini.Convex.Structured.SharedReference
public import Transformer.GPTMini.Convex.Structured.SharedInterface
public import Transformer.GPTMini.Convex.Structured.SharedParity
public import Transformer.GPTMini.Convex.Structured.DepthCompression
public import Transformer.GPTMini.Convex.Structured.DepthScan
public import Transformer.GPTMini.Convex.Structured.DepthReference
public import Transformer.GPTMini.Convex.Structured.SharedDepth
public import Transformer.GPTMini.Convex.Structured.Binding
public import Transformer.GPTMini.Convex.Structured.RecallPositions
public import Transformer.GPTMini.Convex.Structured.ChannelGaps
public import Transformer.GPTMini.Convex.Structured.RecallPotentials
public import Transformer.GPTMini.Convex.Structured.PointerDecoder
public import Transformer.GPTMini.Convex.Structured.RawBinding
public import Transformer.GPTMini.Convex.Structured.RecallEnergy
public import Transformer.GPTMini.Convex.Structured.RecallGap
public import Transformer.GPTMini.Convex.Structured.BindingInterface
public import Transformer.GPTMini.Convex.Structured.SharedRecall
public import Transformer.GPTMini.Convex.Structured.MixedHeads
public import Transformer.GPTMini.Convex.Structured.MixedTraining
public import Transformer.GPTMini.Convex.Structured.MixedConfidence
public import Transformer.GPTMini.Convex.Structured.MixedMargins
public import Transformer.GPTMini.Convex.Structured.MixedInterface
public import Transformer.GPTMini.Convex.Structured.MixedBasis
public import Transformer.GPTMini.Convex.Structured.RecallDataRoutes
public import Transformer.GPTMini.Convex.Structured.RecallDataTargets
public import Transformer.GPTMini.Convex.Structured.BasisTraining
public import Transformer.GPTMini.Convex.Structured.TensorEmbedding
public import Transformer.GPTMini.Convex.Structured.TensorRecovery
public import Transformer.GPTMini.Convex.Structured.TensorState
public import Transformer.GPTMini.Convex.Structured.TensorPointer
public import Transformer.GPTMini.Convex.Structured.TensorHeads
public import Transformer.GPTMini.Convex.Structured.TensorReadout
public import Transformer.GPTMini.Convex.Structured.TensorCausalBlock
public import Transformer.GPTMini.Convex.Structured.TensorTraining
public import Transformer.GPTMini.Convex.Structured.TensorLikelihood
public import Transformer.GPTMini.Convex.Structured.TensorBasisTraining
public import Transformer.GPTMini.Convex.Structured.TensorStream
public import Transformer.GPTMini.Convex.Structured.TensorObservations
public import Transformer.GPTMini.Convex.Structured.TensorObservationInference
public import Transformer.GPTMini.Convex.Structured.TensorDeferredFFN
public import Transformer.GPTMini.Convex.Structured.TensorStack
public import Transformer.GPTMini.Convex.Structured.TensorStackTraining
public import Transformer.GPTMini.Convex.Structured.TensorStackInterface
public import Transformer.GPTMini.Convex.Structured.TensorBasisModel
public import Transformer.GPTMini.Convex.Structured.TensorBasisStackTraining
public import Transformer.GPTMini.Convex.Structured.TensorGain
public import Transformer.GPTMini.Convex.Structured.TensorLossCertificate
public import Transformer.GPTMini.Convex.Structured.LocalConfidence
public import Transformer.GPTMini.Convex.Structured.ParityDataList
public import Transformer.GPTMini.Convex.Structured.ParityRows
public import Transformer.GPTMini.Convex.Structured.ParitySemanticRows
public import Transformer.GPTMini.Convex.Structured.TensorParityCertificate
public import Transformer.GPTMini.Convex.Structured.ParityLogitCertificate
public import Transformer.GPTMini.Convex.Structured.LearnedParityData
public import Transformer.GPTMini.Convex.Structured.LearnedParityParameters
public import Transformer.GPTMini.Convex.Structured.LearnedParityRows
public import Transformer.GPTMini.Convex.Structured.LearnedParityConfidence
public import Transformer.GPTMini.Convex.Structured.TensorLearnedParity

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

@[expose] public section
