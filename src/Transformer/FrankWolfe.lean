/-
# Attention's forward pass and Frank-Wolfe

Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1 (ICML 2026).

The self-attention layer is read as a Frank-Wolfe step for the quadratic
`𝒥(x) = ½⟨Bx, x⟩` over the convex hull of the current configuration.  One
module per section of the manuscript: §1 the models, §2 the derivations and
the counterexample to hull shrinkage for a general value matrix, §3 the
negative-definite regime, §4 the cells, the polytope solution, the vertex
genericity and the singular ODE, §5 finite `β` and dynamic metastability.
Everything lives in the namespace `Transformer.FrankWolfe`.
-/

import Transformer.FrankWolfe.Section1_Models
import Transformer.FrankWolfe.Section2_Derivations
import Transformer.FrankWolfe.Section2_HullFailure
import Transformer.FrankWolfe.Section3_NegativeDefinite
import Transformer.FrankWolfe.Section4_Cells
import Transformer.FrankWolfe.Section4_Polytope
import Transformer.FrankWolfe.Section4_Faces
import Transformer.FrankWolfe.Section4_ExpFast
import Transformer.FrankWolfe.Section4_VertexGenericity
import Transformer.FrankWolfe.Section4_ODE
import Transformer.FrankWolfe.Section5_Process
import Transformer.FrankWolfe.Section5_Metastability
