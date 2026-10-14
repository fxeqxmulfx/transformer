/-
# Kinetic theory for Transformers — the output variable at both ends

The proof of arXiv:2605.09213v1, §6, `thm:U-shape`, transfers the
coercive right tail of `ℋ` to primacy using `Y(σ₀) → ∞` as `σ₀ ↓ 0`.
At the recency endpoint the actual variable tends to zero and its
actual derivative tends to the finite negative number
`-λ e^λ/(e^λ-1)`. These are limits of the original `Yfun`, on its
actual positive-position domain, including the continuous value at one.
-/

import Transformer.Kinetic.Section6_SourcePosition

open Real Set Filter
open scoped Topology

namespace Transformer
namespace Kinetic

/-- The original denominator tends to zero through positive values when
the actual source position tends to zero from the right.
Source: arXiv:2605.09213v1, `eq:Ylambda`, and §6, primacy in `thm:U-shape`. -/
theorem exp_mul_sub_one_tendsto_zero {lam : ℝ} (hlam : 0 < lam) :
    Tendsto (fun s : ℝ => Real.exp (lam * s) - 1) (𝓝[>] 0) (𝓝[>] 0) := by
  have hc : Continuous (fun s : ℝ => Real.exp (lam * s) - 1) := by fun_prop
  apply tendsto_nhdsWithin_iff.mpr
  refine ⟨?_, ?_⟩
  · simpa only [mul_zero, Real.exp_zero, sub_self] using
      (hc.tendsto 0).mono_left nhdsWithin_le_nhds
  · filter_upwards [self_mem_nhdsWithin (a := (0 : ℝ)) (s := Ioi 0)] with s hs
    exact sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hlam hs))

example := exp_mul_sub_one_tendsto_zero one_pos

/-- The original output variable diverges at the primacy endpoint.
Source: arXiv:2605.09213v1, §6, the bijection and primacy step of `thm:U-shape`. -/
theorem Yfun_one_tendsto_atTop {lam : ℝ} (hlam : 0 < lam) :
    Tendsto (Yfun lam 1) (𝓝[>] 0) atTop := by
  have he : 0 < Real.exp lam - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)
  have hi := (exp_mul_sub_one_tendsto_zero hlam).inv_tendsto_nhdsGT_zero
  have hratio := hi.const_mul_atTop he
  have ht := Real.tendsto_log_atTop.comp hratio
  convert ht using 1
  funext s
  simp only [Yfun, ite_eq_right hlam.ne', mul_one, div_eq_mul_inv, Pi.inv_apply,
    Function.comp_apply]

example := Yfun_one_tendsto_atTop one_pos

/-- At the output endpoint, the actual change of variable takes value zero.
Source: arXiv:2605.09213v1, §6, the endpoint bijection in `thm:U-shape`. -/
theorem Yfun_one_one {lam : ℝ} (hlam : 0 < lam) : Yfun lam 1 1 = 0 := by
  rw [Yfun_one_eq_log_sub hlam one_pos, mul_one, sub_self]

example := Yfun_one_one one_pos

/-- The original output variable tends to zero at the recency endpoint.
Source: arXiv:2605.09213v1, §6, the recency step of `thm:U-shape`. -/
theorem Yfun_one_tendsto_zero {lam : ℝ} (hlam : 0 < lam) :
    Tendsto (Yfun lam 1) (𝓝[<] 1) (𝓝 0) := by
  have ht := (hasDerivAt_Yfun_one hlam one_pos).continuousAt.tendsto.mono_left
    (nhdsWithin_le_nhds (s := Iio 1))
  rwa [Yfun_one_one hlam] at ht

example := Yfun_one_tendsto_zero one_pos

/-- The actual derivative factor is strictly negative at every positive
source position. Source: arXiv:2605.09213v1, §6, the decreasing variable in `thm:U-shape`. -/
theorem Yfun_one_derivative_neg {lam s : ℝ} (hlam : 0 < lam) (hs : 0 < s) :
    -(lam * Real.exp (lam * s) / (Real.exp (lam * s) - 1)) < 0 := by
  apply neg_neg_of_pos
  exact div_pos (mul_pos hlam (Real.exp_pos _))
    (sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hlam hs)))

example := Yfun_one_derivative_neg one_pos one_pos

/-- The actual derivative factor is continuous at every positive position.
Source: arXiv:2605.09213v1, §6, the smooth change of variable in `thm:U-shape`. -/
theorem continuousAt_Yfun_one_derivative {lam s : ℝ} (hlam : 0 < lam) (hs : 0 < s) :
    ContinuousAt (fun s : ℝ => -(lam * Real.exp (lam * s) / (Real.exp (lam * s) - 1))) s := by
  have hden : Real.exp (lam * s) - 1 ≠ 0 :=
    (sub_pos.mpr (Real.one_lt_exp_iff.mpr (mul_pos hlam hs))).ne'
  apply ContinuousAt.neg
  apply ContinuousAt.div
  · fun_prop
  · fun_prop
  · exact hden

example := continuousAt_Yfun_one_derivative one_pos one_pos

/-- The actual derivative factor tends to its finite negative output value.
Source: arXiv:2605.09213v1, §6, the recency step of `thm:U-shape`. -/
theorem Yfun_one_derivative_tendsto {lam : ℝ} (hlam : 0 < lam) :
    Tendsto (fun s : ℝ => -(lam * Real.exp (lam * s) / (Real.exp (lam * s) - 1)))
      (𝓝[<] 1) (𝓝 (-(lam * Real.exp lam / (Real.exp lam - 1)))) := by
  simpa only [mul_one] using (continuousAt_Yfun_one_derivative hlam one_pos).tendsto.mono_left
    (nhdsWithin_le_nhds (s := Iio 1))

example := Yfun_one_derivative_tendsto one_pos

/-- The derivative of the original variable equals its computed factor
at every positive position. Source: arXiv:2605.09213v1, §6, `eq:Ylambda`. -/
theorem deriv_Yfun_one {lam s : ℝ} (hlam : 0 < lam) (hs : 0 < s) :
    deriv (Yfun lam 1) s = -(lam * Real.exp (lam * s) / (Real.exp (lam * s) - 1)) :=
  (hasDerivAt_Yfun_one hlam hs).deriv

example := deriv_Yfun_one one_pos one_half_pos

/-- The original source-position variable strictly decreases on its
positive domain. Source: arXiv:2605.09213v1, §6, the bijection in `thm:U-shape`. -/
theorem strictAntiOn_Yfun_one {lam : ℝ} (hlam : 0 < lam) :
    StrictAntiOn (Yfun lam 1) (Ioi 0) := by
  apply strictAntiOn_of_deriv_neg (convex_Ioi (0 : ℝ))
  · intro s hs
    exact (hasDerivAt_Yfun_one hlam hs).continuousAt.continuousWithinAt
  · intro s hs
    have hs0 : 0 < s := interior_subset hs
    rw [deriv_Yfun_one hlam hs0]
    exact Yfun_one_derivative_neg hlam hs0

example := strictAntiOn_Yfun_one one_pos

/-- The actual inverse position formula is continuous on the whole line.
Source: arXiv:2605.09213v1, §6, the smooth inverse of `eq:Ylambda`. -/
theorem continuous_sourcePosition {lam : ℝ} (hlam : 0 < lam) :
    Continuous (sourcePosition lam) := by
  have he : 0 < Real.exp lam - 1 := sub_pos.mpr (Real.one_lt_exp_iff.mpr hlam)
  apply continuous_iff_continuousAt.mpr
  intro y
  have ha : 1 + (Real.exp lam - 1) * Real.exp (-y) ≠ 0 := by positivity
  have hc : ContinuousAt (fun y : ℝ => 1 + (Real.exp lam - 1) * Real.exp (-y)) y := by fun_prop
  exact (hc.log ha).div_const lam

example := continuous_sourcePosition one_pos

/-- The explicit inverse tends to the primacy position as its real
argument tends to infinity. Source: arXiv:2605.09213v1, §6, the inverse of `eq:Ylambda`. -/
theorem sourcePosition_tendsto_zero (lam : ℝ) :
    Tendsto (sourcePosition lam) atTop (𝓝 0) := by
  have harg : Tendsto (fun y : ℝ => 1 + (Real.exp lam - 1) * Real.exp (-y)) atTop (𝓝 1) := by
    simpa only [mul_zero, add_zero] using
      (Real.tendsto_exp_neg_atTop_nhds_zero.const_mul (Real.exp lam - 1)).const_add 1
  change Tendsto (fun y : ℝ => Real.log (1 + (Real.exp lam - 1) * Real.exp (-y)) / lam)
    atTop (𝓝 0)
  simpa only [Real.log_one, zero_div] using (harg.log one_ne_zero).div_const lam

example := sourcePosition_tendsto_zero 1

end Kinetic
end Transformer
