/-
# Metastability — from cap containment and cap collapse to `thm: metastability`

The statement of `thm: metastability` (arXiv:2410.06833v1, §2) is about the caps `𝒮_q(2ε)` of
the centres `w_q`; the proof works with the *classes* `I_q = {i : x_i(0) ∈ 𝒮_q(ε)}` of the tokens.
Because `γ(β) > 0` forces `α < 1`, the caps `𝒮_q(2ε)` are pairwise disjoint (`cap_eq_of_mem`),
so a token lying in `𝒮_q(2ε)` at time `t ≤ T₂` belongs to the class `I_q`.  This file turns

* `hcont` — the token `i` stays in `𝒮_{q(i)}(2ε)` on `[0, T₂]` (`cap_containment`), and
* `hpair` — tokens of one class have `⟨x_i, x_j⟩ ≥ 1 - δ` on `[T₁, T₂]` (`collapse_dynamic`),

into the two items of the theorem.
-/

import Transformer.Metastability.CapGeometry

open scoped BigOperators InnerProductSpace
open Real Set

namespace Transformer
namespace Metastability

variable {d n : ℕ}

/-- **The threshold `λ_* = β⁻¹ log(1/(8ε))`** of `eq: lambda.3`: `e^{-λ_*β} = 8ε`. -/
noncomputable def lamStar (β ε : ℝ) : ℝ := β⁻¹ * Real.log (1 / (8 * ε))

theorem exp_neg_lamStar_mul {β ε : ℝ} (hβ : 0 < β) (hε : 0 < ε) :
    Real.exp (-(lamStar β ε * β)) = 8 * ε := by
  have h : lamStar β ε * β = Real.log (1 / (8 * ε)) := by
    unfold lamStar
    field_simp
  rw [h, Real.exp_neg, Real.exp_log (by positivity)]
  field_simp

/-- Bigger caps contain smaller ones: `𝒮_w(c) ⊆ 𝒮_w(c')` for `c ≤ c'`. -/
theorem sphericalCap_mono {w : SSphere d} {c c' : ℝ} (h : c ≤ c') :
    sphericalCap d w c ⊆ sphericalCap d w c' := fun x hx => by
  have hx' : 1 - c ≤ ⟪(x : EucSpace d), (w : EucSpace d)⟫_ℝ := hx
  show 1 - c' ≤ _
  linarith

/-- **Assembly of `thm: metastability`.**  Let `q i` be a cap with `x_i(0) ∈ 𝒮_{q(i)}(ε)`, and
assume `α = α(ε) < 1`.  If token `i` stays in `𝒮_{q(i)}(2ε)` on `[0, T₂]` and tokens of one class
satisfy `⟨x_i, x_j⟩ ≥ 1 - δ` on `[T₁, T₂]`, then

1. if `x_i(0) ∈ 𝒮_{q'}(ε)` then `x_i(t) ∈ 𝒮_{q'}(2ε)` for `t ∈ [0, T₂]`;
2. if `x_i(t), x_j(t) ∈ 𝒮_{q'}(2ε)` for `t ∈ [T₁, T₂]` then `‖x_i(t) - x_j(t)‖² ≤ 2δ`. -/
theorem metastability_assemble {ε δ T₁ T₂ : ℝ} {k : ℕ} (w : Idx k → SSphere d)
    (hα : αDist d k w ε < 1) (hε : 0 < ε) (q : Idx n → Idx k) (X₀ : SphereTuple d n)
    (hq : ∀ i, X₀ i ∈ sphericalCap d (w (q i)) ε) {X : ℝ → SphereTuple d n}
    (hcont : ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t → t ≤ T₂ → X t i ∈ sphericalCap d (w (q i)) (2 * ε))
    (hpair : ∀ t : ℝ, T₁ ≤ t → t ≤ T₂ → ∀ i j : Idx n, q i = q j →
      1 - δ ≤ ⟪(X t i : EucSpace d), (X t j : EucSpace d)⟫_ℝ) (hT₁ : 0 ≤ T₁) :
    (∀ (i : Idx n) (q' : Idx k), X₀ i ∈ sphericalCap d (w q') ε → ∀ t : ℝ, 0 ≤ t → t ≤ T₂ →
        X t i ∈ sphericalCap d (w q') (2 * ε)) ∧
      ∀ (q' : Idx k) (t : ℝ), T₁ ≤ t → t ≤ T₂ → ∀ i j : Idx n,
        X t i ∈ sphericalCap d (w q') (2 * ε) → X t j ∈ sphericalCap d (w q') (2 * ε) →
          ‖(X t i : EucSpace d) - (X t j : EucSpace d)‖ ^ 2 ≤ 2 * δ := by
  refine ⟨fun i q' hi t ht0 htT => ?_, fun q' t ht1 htT i j hi hj => ?_⟩
  · have h : q' = q i :=
      cap_eq_of_mem hα (sphericalCap_mono (by linarith) hi)
        (sphericalCap_mono (by linarith) (hq i))
    rw [h]
    exact hcont i t ht0 htT
  · have ht0 : 0 ≤ t := hT₁.trans ht1
    have hqi : q i = q' := cap_eq_of_mem hα (hcont i t ht0 htT) hi
    have hqj : q j = q' := cap_eq_of_mem hα (hcont j t ht0 htT) hj
    have h := hpair t ht1 htT i j (hqi.trans hqj.symm)
    rw [norm_sq_sub_eq]
    linarith

end Metastability
end Transformer
