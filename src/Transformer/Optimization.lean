/-
# Training convergence safeguards shared by Muon and DASH

Additional deterministic loss assumptions and explicit algorithm corrections
for the user-requested training analysis of arXiv:2502.16982, §2.1–2.2,
and arXiv:2602.02016v2, §2–4. These are not claimed as printed paper theorems.
-/

import Transformer.Optimization.Basic
import Transformer.Optimization.Descent
import Transformer.Optimization.Stationarity
import Transformer.Optimization.StrongConvexity
import Transformer.Optimization.StrongConvergence
import Transformer.Optimization.Quadratic
import Transformer.Optimization.Matrix
