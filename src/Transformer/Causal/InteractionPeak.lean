/-
# Causal attention — The peak of the interaction potential (§B of 2411.04990v2)

`lemma:interaction` (3): on `[0, π]` the potential `h(x) = e^{β(cos x - 1)} sin x`
rises to a single peak and falls back.

The peak is where `g = h'` changes sign.  Since

  `g(x) = e^{β(cos x - 1)} (β cos²x + cos x - β)`,

that happens exactly where `cos x` crosses the positive root
`c_β = (-1 + √(4β² + 1)) / (2β)` of `βu² + u - β`, and `τ_β^* = arccos c_β`.
The second factor of the factorization `βu² + u - β = (u - c_β)(β(u + c_β) + 1)`
is positive on all of `[-1, 1]`, so the sign of `g` is the sign of
`cos x - c_β` and the two monotonicity intervals follow from the mean value
theorem.

That `τ_β^*` sits at the interaction scale is the pair of Taylor bounds on the
cosine read against `c_β`:

* `c_β > 1 - 1/(2β) + 1/(24β²) ≥ cos β^{-1/2}` because `√(4β²+1) > 2β + 1/(12β)`;
* `c_β < 1 - 1/(2β+1) ≤ cos (β+1/2)^{-1/2}` because `√(4β²+1) (2β+1) < 4β²+2β+1`,
  the two sides differing by exactly `4β²` after squaring.

This is the one item of `lemma:interaction` that needs the quartic bound
`cos_le_quartic`, which is why it comes after `Causal.InteractionBounds`.
-/

import Transformer.Causal.InteractionBounds

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

/-- **Lemma (lemma:interaction), 3.** *`h` is unimodal on `[0, π]`.*

`h` increases on `[0, τ_β^*]` and decreases on `[τ_β^*, π]`, where

  `cos τ_β^* = (-1 + √(4β² + 1)) / (2β)`,

and for `β ≥ 1` the peak sits at the interaction scale,

  `(β + 1/2)^{-1/2} < τ_β^* < β^{-1/2}`.

The peak is existentially quantified rather than constructed: `τ_β^*` is the
arccosine of the displayed value, and what the statement keeps of it is what
the proof of `thm: fixed_centers` uses — it lies in `(0, π)`, it is where the
two monotonicity intervals meet, and it is of order `β^{-1/2}`.

Source: arXiv:2411.04990v2, §B, `lemma:interaction` (3). -/
theorem h_pot_unimodal (β : ℝ) (hβ : 1 ≤ β) :
    ∃ τ : ℝ, 0 < τ ∧ τ < Real.pi ∧
      Real.cos τ = (-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β) ∧
      (β + 1 / 2) ^ (-(1 / 2 : ℝ)) < τ ∧ τ < β ^ (-(1 / 2 : ℝ)) ∧
      StrictMonoOn (h_pot β) (Set.Icc 0 τ) ∧
      StrictAntiOn (h_pot β) (Set.Icc τ Real.pi) := by
  have hβ0 : (0 : ℝ) < β := by linarith
  have hβne : β ≠ 0 := ne_of_gt hβ0
  -- the positive root of `βu² + u - β`
  set S : ℝ := Real.sqrt (4 * β ^ 2 + 1) with hSdef
  have hS2 : S ^ 2 = 4 * β ^ 2 + 1 := Real.sq_sqrt (by positivity)
  have hSgt : 2 * β < S := (Real.lt_sqrt (by positivity)).mpr (by nlinarith)
  have hSlt : S < 2 * β + 1 :=
    (Real.sqrt_lt (by positivity) (by positivity)).mpr (by nlinarith)
  set c : ℝ := (-1 + S) / (2 * β) with hcdef
  have hc0 : 0 < c := div_pos (by linarith) (by positivity)
  have hc1 : c < 1 := (div_lt_one (by positivity)).mpr (by linarith)
  have hroot : β * c ^ 2 + c - β = 0 := by
    have hexp : β * c ^ 2 + c - β = (S ^ 2 - 1 - 4 * β ^ 2) / (4 * β) := by
      rw [hcdef]; field_simp; ring
    rw [hexp, hS2]; ring
  have hbc : β * c = (S - 1) / 2 := by rw [hcdef]; field_simp; ring
  have hsecond : ∀ u : ℝ, -1 ≤ u → 0 < β * (u + c) + 1 := by
    intro u hu
    nlinarith [mul_nonneg hβ0.le (by linarith : (0 : ℝ) ≤ u + 1)]
  -- the peak
  set τ : ℝ := Real.arccos c with hτdef
  have hcos : Real.cos τ = c := Real.cos_arccos (by linarith) hc1.le
  have hτ0 : 0 < τ := Real.arccos_pos.mpr hc1
  have hτπ : τ < Real.pi := Real.arccos_lt_pi.mpr (by linarith)
  have hτmem : τ ∈ Set.Icc 0 Real.pi := ⟨hτ0.le, hτπ.le⟩
  -- the sign of `g` is the sign of `cos x - c`
  have hg : ∀ x : ℝ, g_pot β x
      = Real.exp (β * (Real.cos x - 1))
        * ((Real.cos x - c) * (β * (Real.cos x + c) + 1)) := by
    intro x
    rw [g_pot]
    congr 1
    linear_combination (-β) * Real.sin_sq_add_cos_sq x + hroot
  have hcont : Continuous (h_pot β) := by
    unfold h_pot
    fun_prop
  -- the interaction scale
  have hy : (0 : ℝ) < β ^ (-(1 / 2 : ℝ)) := Real.rpow_pos_of_pos hβ0 _
  have hy2 : β * (β ^ (-(1 / 2 : ℝ))) ^ 2 = 1 := by
    have h2 : (β ^ (-(1 / 2 : ℝ))) ^ 2 = β⁻¹ := by
      rw [← Real.rpow_natCast (β ^ (-(1 / 2 : ℝ))) 2, ← Real.rpow_mul hβ0.le]
      norm_num
      rw [Real.rpow_neg_one]
    rw [h2, mul_inv_cancel₀ hβne]
  have hz : (0 : ℝ) < (β + 1 / 2) ^ (-(1 / 2 : ℝ)) :=
    Real.rpow_pos_of_pos (by linarith) _
  have hz2 : (β + 1 / 2) * ((β + 1 / 2) ^ (-(1 / 2 : ℝ))) ^ 2 = 1 := by
    have h2 : ((β + 1 / 2) ^ (-(1 / 2 : ℝ))) ^ 2 = (β + 1 / 2)⁻¹ := by
      rw [← Real.rpow_natCast ((β + 1 / 2) ^ (-(1 / 2 : ℝ))) 2,
        ← Real.rpow_mul (by linarith)]
      norm_num
      rw [Real.rpow_neg_one]
    rw [h2, mul_inv_cancel₀ (by positivity)]
  set y : ℝ := β ^ (-(1 / 2 : ℝ)) with hydef
  set z : ℝ := (β + 1 / 2) ^ (-(1 / 2 : ℝ)) with hzdef
  have hy2le : y ^ 2 ≤ 1 := by
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ β - 1) (sq_nonneg y)]
  have hyle : y ≤ 1 := by nlinarith
  have hymem : y ∈ Set.Icc 0 Real.pi := ⟨hy.le, by linarith [Real.pi_gt_three]⟩
  have hz2le : z ^ 2 ≤ 2 / 3 := by
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ β - 1) (sq_nonneg z)]
  have hzle : z ≤ 1 := by nlinarith
  have hzmem : z ∈ Set.Icc 0 Real.pi := ⟨hz.le, by linarith [Real.pi_gt_three]⟩
  refine ⟨τ, hτ0, hτπ, hcos, ?_, ?_, ?_, ?_⟩
  · -- `(β + 1/2)^{-1/2} < τ`: `c < 1 - z²/2 ≤ cos z`
    have hby4 : β * z ^ 2 = 1 - z ^ 2 / 2 := by linarith [hz2]
    have hSlt2 : S < 2 * β + z ^ 2 / 2 := by
      refine (Real.sqrt_lt (by positivity) (by positivity)).mpr ?_
      nlinarith [hby4, sq_nonneg (z ^ 2)]
    have hclt : c < 1 - z ^ 2 / 2 := by
      rw [hcdef, div_lt_iff₀ (by positivity)]
      nlinarith [hby4]
    have hcosz : 1 - z ^ 2 / 2 ≤ Real.cos z := Real.one_sub_sq_div_two_le_cos
    exact (Real.strictAntiOn_cos.lt_iff_gt hτmem hzmem).mp (by rw [hcos]; linarith)
  · -- `τ < β^{-1/2}`: `cos y ≤ 1 - y²/2 + y⁴/24 < c`
    have hy4 : y ^ 2 * y ^ 2 ≤ 1 := by nlinarith [sq_nonneg y]
    have hSgt2 : 2 * β + y ^ 2 / 12 < S := by
      refine (Real.lt_sqrt (by positivity)).mpr ?_
      nlinarith [hy2, hy4]
    have hby4 : β * y ^ 4 = y ^ 2 := by
      have h : β * y ^ 4 = β * y ^ 2 * y ^ 2 := by ring
      rw [h, hy2, one_mul]
    have hclow : 1 - y ^ 2 / 2 + y ^ 4 / 24 < c := by
      rw [hcdef, lt_div_iff₀ (by positivity)]
      nlinarith [hSgt2, hy2, hby4]
    have hcosy : Real.cos y ≤ 1 - y ^ 2 / 2 + y ^ 4 / 24 := cos_le_quartic y
    exact (Real.strictAntiOn_cos.lt_iff_gt hymem hτmem).mp (by rw [hcos]; linarith)
  · -- `h` rises on `[0, τ]`
    refine strictMonoOn_of_deriv_pos (convex_Icc 0 τ) hcont.continuousOn ?_
    intro x hx
    rw [interior_Icc] at hx
    rw [(hasDerivAt_h_pot β x).deriv, hg x]
    have hcx : c < Real.cos x := by
      rw [← hcos]
      exact Real.cos_lt_cos_of_nonneg_of_le_pi hx.1.le hτπ.le hx.2
    exact mul_pos (Real.exp_pos _)
      (mul_pos (by linarith) (hsecond _ (Real.neg_one_le_cos x)))
  · -- `h` falls on `[τ, π]`
    refine strictAntiOn_of_deriv_neg (convex_Icc τ Real.pi) hcont.continuousOn ?_
    intro x hx
    rw [interior_Icc] at hx
    rw [(hasDerivAt_h_pot β x).deriv, hg x]
    have hcx : Real.cos x < c := by
      rw [← hcos]
      exact Real.cos_lt_cos_of_nonneg_of_le_pi hτ0.le hx.2.le hx.1
    exact mul_neg_of_pos_of_neg (Real.exp_pos _)
      (mul_neg_of_neg_of_pos (by linarith) (hsecond _ (Real.neg_one_le_cos x)))

/-- The hypothesis of `h_pot_unimodal` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≤ 1 := le_rfl

end Causal
end Transformer
