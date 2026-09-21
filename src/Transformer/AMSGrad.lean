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
* Lemma 2.3's `Σ_{t≥1} αᵗ = 1/(1-α)` is false and refuted;
* Theorem 4.1's `t₀` is chosen before `T`, not `1 ≤ t₀ ≤ T`;
* Corollary 4.5's `lim R(T)/T = 0` is kept as its upper half only; the lower
  half is false, and the counterexample is stated (`not_cor_lower`);
* Theorem 5.1's second term carries `(1-β₁)²`, as its proof gives, not `(1-β₁)`;
* Corollaries 5.5 and 5.6 are kept as their upper halves: with `β_{1,t} = 0`
  AdamX is AMSGrad (`adamX_eq_amsgrad`), so `not_cor_lower` refutes them too;
* Corollary 5.6's `β_{1,t} = 1/t` contradicts `β_{1,1} < 1` and is corrected
  to `β_{1,t} = β₁/t`.

Not transcribed, deliberately: the experiments of §6.
-/

import Transformer.AMSGrad.Section1_AMSGrad
import Transformer.AMSGrad.Section1_TheoremA
import Transformer.AMSGrad.Section2_Prelim
import Transformer.AMSGrad.Section2_Proj
import Transformer.AMSGrad.Section3_Step
import Transformer.AMSGrad.Section3_Issue
import Transformer.AMSGrad.Section3_Example
import Transformer.AMSGrad.Section3_Optimal
import Transformer.AMSGrad.Section4_Lemmas
import Transformer.AMSGrad.Section4_MainLemma
import Transformer.AMSGrad.Section4_Theorem
import Transformer.AMSGrad.Section4_Counter
import Transformer.AMSGrad.Section5_AdamX
import Transformer.AMSGrad.Section5_Theorem
