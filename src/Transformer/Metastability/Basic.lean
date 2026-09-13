/-
# Metastability — Basic definitions

Formalization of arXiv:2410.06833v1 — "Dynamic metastability in the
self-attention model" (Geshkovski, Koubbi, Polyanskiy, Rigollet).

This file collects the basic objects used throughout the paper:

* the simplified self-attention model `SA` and its un-normalised variant
  `USA` (recalled from §1 of the survey 2312.10794v5);
* the interaction energy `𝖤_β` in the form used in 2410.06833v1;
* spherical caps `𝒮_q(ε) = {x ∈ 𝕊^{d-1} : ⟨x, w_q⟩ ≥ 1 - ε}`;
* `(β, ε)`-separated configurations (`Definition (hyp: init)`);
* the parameters `α(ε)`, `γ(β)` and `λ(β)` of the main theorem;
* shorthand notation for `α(ε) = max ⟨x, y⟩` over distinct caps.
-/

import Transformer.Basic
import Transformer.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **Interaction energy** in the rescaled form used in 2410.06833v1:

  `𝖤_β(x_1,…,x_n) = (1/(2 β e^β n²)) Σ_i Σ_j exp(β ⟨x_i, x_j⟩)`. -/
noncomputable def Eβ (β : ℝ) (X : SphereTuple d n) : ℝ :=
  (1 / (2 * β * Real.exp β * (n : ℝ)^2)) *
    ∑ i : Idx n, ∑ j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))

/-- **Spherical cap (eq: cones).**

  `𝒮_q(ε) = { x ∈ 𝕊^{d-1} : ⟨x, w_q⟩ ≥ 1 - ε }`. -/
def sphericalCap (w : SSphere d) (ε : ℝ) : Set (SSphere d) :=
  { x | (1 : ℝ) - ε ≤ inner (𝕜 := ℝ) ((x : EucSpace d)) ((w : EucSpace d)) }

/-- **Pairwise-max inner product (eq: alpha.dist).**

  `α(ε) = max { ⟨x, y⟩ : x ∈ 𝒮_i(2ε), y ∈ 𝒮_j(2ε), i ≠ j }`. -/
noncomputable def αDist
    (k : ℕ) (w : Idx k → SSphere d) (ε : ℝ) : ℝ := by
  -- supremum over `i ≠ j` of `⟨x, y⟩` with `x ∈ 𝒮_i(2ε), y ∈ 𝒮_j(2ε)`.
  exact 0

/-- **Parameter γ (eq: gamma).**

  `γ(β) := 1 - α - 8 ε - β⁻¹ log(2 n² / ε)`. -/
noncomputable def γβ
    (n : ℕ) (β α ε : ℝ) : ℝ :=
  1 - α - 8 * ε - β⁻¹ * Real.log (2 * (n : ℝ)^2 / ε)

/-- **`(β, ε)`-separated configuration (Definition hyp: init).**

`(x_1,…,x_n) ∈ (𝕊^{d-1})^n` is `(β, ε)`-separated if there exist
`k ≤ n` centres `w_1,…,w_k ∈ 𝕊^{d-1}` such that:

1. each `x_i(0)` lies in `⋃_q 𝒮_q(ε)`,
2. `γ(β) > 0` and `γ(β) = Ω(1)`. -/
def isSeparated
    (β ε : ℝ) (X : SphereTuple d n) : Prop :=
  ∃ (k : ℕ) (hk : k ≤ n) (w : Idx k → SSphere d),
    (∀ i : Idx n, ∃ q : Idx k, X i ∈ sphericalCap d (w q) ε)
    ∧ 0 < γβ n β (αDist d k w ε) ε

/-- Pair `(T₁, T₂)` of times appearing in the main theorem.
`T₁` bounds the cap-collapse time, `T₂` the cap-escape time. -/
structure MetastabilityTimes (β ε α : ℝ) (n : ℕ) (lam : ℝ) where
  T1 : ℝ
  T2 : ℝ
  pos    : 0 < T1
  ord    : T1 < T2
  upperT1 : T1 ≤ 2 * (n : ℝ) * Real.exp (8 * ε * β)
              + Real.exp 1 * (n : ℝ) * lam * β^2 / (β - 1)
  lowerT2 : (ε / n) * Real.exp ((1 - α) * β) ≤ T2

end Metastability
end Transformer
