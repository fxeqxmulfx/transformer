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

/-!
# Structured alternatives guided by raw Basis semantics

Source: complete Basis semantics at d640a91 and the finite log-sum-exp
argument in arXiv:2305.05465v6, §7. Complete affine Gibbs objectives are
convex in all raw parameters. A compact trainable embedding/attention
replacement and full task capability are still construction obligations.
-/
