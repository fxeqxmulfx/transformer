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

/-!
# Structured alternatives guided by raw Basis semantics

Source: complete Basis semantics at d640a91 and the finite log-sum-exp
argument in arXiv:2305.05465v6, §7. Complete affine Gibbs objectives are
convex in all raw parameters. A compact trainable embedding/attention
replacement and full task capability are still construction obligations.
-/
