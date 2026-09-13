/-
# Mean-Field Dynamics of Transformers — survey

Formalization of arXiv:2512.01868v4 — "The Mean-Field Dynamics of Transformers"
(survey by Geshkovski, Polyanskiy, Rigollet).

This file collects:

* `eq: attention`             — the discrete attention operator,
* `eq: SA`, `eq: USA`         — the simplified continuous-time models,
* `eq: kuramoto`              — Kuramoto reduction on the circle (d = 2),
* `eq: continuity`            — mean-field continuity equation,
* `eq: E`                     — interaction energy `𝒠_β`,
* `eq: NA`                    — normalized attention dynamics (re-exporting
                                the unified speed-regulation formulation).
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators
open Real

namespace Transformer
namespace MeanField

variable (d n : ℕ)

/-- **Equation (eq: attention).** Discrete attention operator:

  `Attention(𝐗)_i =
        Σ_j (e^{β ⟨Q X_i, K X_j⟩} / Σ_k e^{β ⟨Q X_i, K X_k⟩}) V X_j`. -/
noncomputable def attentionDiscrete
    (β : ℝ) (Q K V : ParamMatrix d)
    (X : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  let Z := ∑ k : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (Q (X i)) (K (X k)))
  Z⁻¹ • ∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (Q (X i)) (K (X j))) • V (X j)

/-- **Equation (eq: E).** Interaction energy on the sphere:

  `𝒠_β(μ) = (1/(2β)) ∬ e^{β ⟨x,y⟩} dμ(x) dμ(y)`. -/
noncomputable def Eβ_mf
    (β : ℝ) (μ : SectionFlowMap.ProbSphere d) : ℝ :=
  SectionFlowMap.interactionEnergy d β μ

end MeanField
end Transformer
