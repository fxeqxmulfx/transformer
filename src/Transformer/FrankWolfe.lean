module

public import Transformer.FrankWolfe.Section1_Models
public import Transformer.FrankWolfe.Section2_LinearBranches
public import Transformer.FrankWolfe.Section2_Derivations
public import Transformer.FrankWolfe.Section2_HullFailure
public import Transformer.FrankWolfe.Section3_NegativeDefinite
public import Transformer.FrankWolfe.Section3_FWClusterFalse
public import Transformer.FrankWolfe.Section4_Cells
public import Transformer.FrankWolfe.Section4_Polytope
public import Transformer.FrankWolfe.Section4_Faces
public import Transformer.FrankWolfe.Section4_ExpFast
public import Transformer.FrankWolfe.Section4_Edges
public import Transformer.FrankWolfe.Section4_VertexGenericity
public import Transformer.FrankWolfe.Section4_ODE
public import Transformer.FrankWolfe.Section4_ODESolution
public import Transformer.FrankWolfe.Section4_ODEUnique
public import Transformer.FrankWolfe.Section4_ODEWellPosed
public import Transformer.FrankWolfe.Section5_Process
public import Transformer.FrankWolfe.Section5_Transition
public import Transformer.FrankWolfe.Section5_TransitionAS
public import Transformer.FrankWolfe.Section5_HistoryKernel
public import Transformer.FrankWolfe.Section5_ProcessLaw
public import Transformer.FrankWolfe.Section5_MetaWeights
public import Transformer.FrankWolfe.Section5_Contraction
public import Transformer.FrankWolfe.Section5_Collapse
public import Transformer.FrankWolfe.Section5_Metastability
public import Transformer.FrankWolfe.Section5_FirstPhaseGeometry
public import Transformer.FrankWolfe.Section5_FirstPhaseStarts
public import Transformer.FrankWolfe.Section5_FirstPhaseRefutation
public import Transformer.FrankWolfe.Section5_MetaLevel
public import Transformer.FrankWolfe.Section5_RareEvents
public import Transformer.FrankWolfe.Section5_BadEvents
public import Transformer.FrankWolfe.Section5_FirstExit
public import Transformer.FrankWolfe.Section5_MetaTail
public import Transformer.FrankWolfe.Section5_BinomialExp
public import Transformer.FrankWolfe.Section5_MetaBound

/-!
# Attention's forward pass and Frank-Wolfe

Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1 (ICML 2026).

The self-attention layer is read as a Frank-Wolfe step for the quadratic
`𝒥(x) = ½⟨Bx, x⟩` over the convex hull of the current configuration. One
module per section of the manuscript: §1 the models, §2 the derivations and
the counterexample to hull shrinkage for a general value matrix, §3 the
negative-definite regime, §4 the cells, the polytope solution, the vertex
genericity and the singular ODE, §5 finite `β`, the collapse of softmax
attention and dynamic metastability.
Everything lives in the namespace `Transformer.FrankWolfe`.
-/

@[expose] public section
