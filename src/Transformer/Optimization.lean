module

public import Transformer.Optimization.Basic
public import Transformer.Optimization.Descent
public import Transformer.Optimization.Stationarity
public import Transformer.Optimization.StrongConvexity
public import Transformer.Optimization.StrongConvergence
public import Transformer.Optimization.Quadratic
public import Transformer.Optimization.Matrix

/-!
# Training convergence safeguards shared by Muon and DASH

Additional deterministic loss assumptions and explicit algorithm corrections
for the user-requested training analysis of arXiv:2502.16982, §2.1–2.2,
and arXiv:2602.02016v2, §2–4. These are not claimed as printed paper theorems.
-/

@[expose] public section
