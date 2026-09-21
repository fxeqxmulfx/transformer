/-
Formalization of:
  Tran, Le,
  "On the Convergence Proof of AMSGrad and a New Version",
  arXiv:1904.03590v4.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* `γ = β₁/√β₂ < 1` throughout, where the source has `γ ≤ 1` and divides by
  `1 - γ`; `0 < β₂ < 1` and `0 ≤ β_{1,t} ≤ β₁ < 1` are made explicit;
* every regret bound holds for every `x* ∈ F`, not only for the minimizer;
* Lemma 2.3's `Σ_{t≥1} αᵗ = 1/(1-α)` is false and refuted.

Not transcribed, deliberately: the experiments of §6.
-/

import Transformer.AMSGrad.Section1_AMSGrad
import Transformer.AMSGrad.Section1_TheoremA
import Transformer.AMSGrad.Section2_Prelim
import Transformer.AMSGrad.Section2_Proj
import Transformer.AMSGrad.Section3_Issue
import Transformer.AMSGrad.Section3_Example
import Transformer.AMSGrad.Section3_Optimal
