/-
# Kinetic theory for Transformers — the eigenvalues `a_n` tend to `0`

The footnote of `thm:U-shape` of arXiv:2605.09213v1: since `w_β` is smooth,
`a_n = n² ŵ_β(n) → 0`, which is what makes the smallness condition
`eq:affine-smallness` non-empty for `t > 0`.

Three integrations by parts on `[0, 2π]` move three derivatives onto `w_β`:
the boundary terms vanish because `sin(2πn) = sin 0` and `cos(2πn) = cos 0`,
and the one step that needs `w_β` itself at the endpoints — the second — sees
`w_β'`, which vanishes at both.  So `ŵ_β(n) = n⁻³ (2π)⁻¹ ∫ sin(nθ) w_β'''(θ) dθ`,
and `|a_n| ≤ C / n`.
-/

import Transformer.Kinetic.Defs
import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts

open Real

namespace Transformer
namespace Kinetic

/-- Integration by parts against `cos(nθ)` over one period: the boundary term
is `g · sin(nθ)/n`, which vanishes at `0` and at `2π`. -/
theorem integral_cos_mul_eq {g g' : ℝ → ℝ} (hg : ∀ x, HasDerivAt g (g' x) x)
    (hg' : Continuous g') {n : ℕ} (hn : 0 < n) :
    ∫ θ in (0 : ℝ)..2 * π, Real.cos (n * θ) * g θ
      = -(1 / n) * ∫ θ in (0 : ℝ)..2 * π, Real.sin (n * θ) * g' θ := by
  have hn' : (n : ℝ) ≠ 0 := by positivity
  have hv : ∀ x, HasDerivAt (fun θ => Real.sin (n * θ) / n) (Real.cos (n * x)) x := by
    intro x
    have := (((hasDerivAt_id x).const_mul (n : ℝ)).sin).div_const (n : ℝ)
    simp only [id, mul_one] at this
    convert this using 1
    field_simp
  have h := intervalIntegral.integral_mul_deriv_eq_deriv_mul (a := 0) (b := 2 * π)
    (fun x _ => hg x) (fun x _ => hv x) (hg'.intervalIntegrable _ _)
    ((by fun_prop : Continuous fun x : ℝ => Real.cos (n * x)).intervalIntegrable _ _)
  have hs : Real.sin (n * (2 * π)) = 0 := by
    rw [show (n : ℝ) * (2 * π) = ((2 * n : ℕ) : ℝ) * π by push_cast; ring]
    exact Real.sin_nat_mul_pi _
  simp only [mul_zero, Real.sin_zero, hs, zero_div, sub_zero, zero_sub] at h
  calc _ = ∫ θ in (0 : ℝ)..2 * π, g θ * Real.cos (n * θ) := by
        congr 1; ext θ; ring
    _ = -∫ θ in (0 : ℝ)..2 * π, g' θ * (Real.sin (n * θ) / n) := h
    _ = _ := by
        rw [neg_mul, ← intervalIntegral.integral_const_mul]
        congr 1; congr 1; ext θ; field_simp

/-- Integration by parts against `sin(nθ)` over one period, for a `g` that
takes the same value at both ends: the boundary term `-g · cos(nθ)/n` cancels. -/
theorem integral_sin_mul_eq {g g' : ℝ → ℝ} (hg : ∀ x, HasDerivAt g (g' x) x)
    (hg' : Continuous g') (hper : g (2 * π) = g 0) {n : ℕ} (hn : 0 < n) :
    ∫ θ in (0 : ℝ)..2 * π, Real.sin (n * θ) * g θ
      = (1 / n) * ∫ θ in (0 : ℝ)..2 * π, Real.cos (n * θ) * g' θ := by
  have hn' : (n : ℝ) ≠ 0 := by positivity
  have hv : ∀ x, HasDerivAt (fun θ => -Real.cos (n * θ) / n) (Real.sin (n * x)) x := by
    intro x
    have := (((hasDerivAt_id x).const_mul (n : ℝ)).cos).neg.div_const (n : ℝ)
    simp only [id, mul_one, Pi.neg_apply] at this
    convert this using 1
    field_simp
  have h := intervalIntegral.integral_mul_deriv_eq_deriv_mul (a := 0) (b := 2 * π)
    (fun x _ => hg x) (fun x _ => hv x) (hg'.intervalIntegrable _ _)
    ((by fun_prop : Continuous fun x : ℝ => Real.sin (n * x)).intervalIntegrable _ _)
  simp only [mul_zero, Real.cos_zero, Real.cos_nat_mul_two_pi, hper, sub_self,
    zero_sub] at h
  calc _ = ∫ θ in (0 : ℝ)..2 * π, g θ * Real.sin (n * θ) := by
        congr 1; ext θ; ring
    _ = -∫ θ in (0 : ℝ)..2 * π, g' θ * (-Real.cos (n * θ) / n) := h
    _ = _ := by
        rw [← intervalIntegral.integral_neg, ← intervalIntegral.integral_const_mul]
        congr 1; ext θ; field_simp

/-- **The footnote of `thm:U-shape`.**  Since `w_β` is smooth, `a_n → 0`.

The source's footnote is read at every `β`: the argument uses only the
smoothness of `w_β`, not the sign of `β`.

Source: arXiv:2605.09213v1, `thm:U-shape`, footnote. -/
theorem aCoeff_tendsto_zero (β : ℝ) :
    Filter.Tendsto (aCoeff β) Filter.atTop (nhds 0) := by
  set w := wBeta β
  have hw : ContDiff ℝ 3 w :=
    Real.contDiff_exp.comp (contDiff_const.mul Real.contDiff_cos)
  have hd : ∀ k < 3, ∀ x, HasDerivAt (iteratedDeriv k w) (iteratedDeriv (k + 1) w x) x :=
    fun k hk x => by
      rw [iteratedDeriv_succ]
      exact ((hw.differentiable_iteratedDeriv k (by exact_mod_cast hk)) x).hasDerivAt
  have hc : ∀ k ≤ 3, Continuous (iteratedDeriv k w) :=
    fun k hk => hw.continuous_iteratedDeriv k (by exact_mod_cast hk)
  have h1 : iteratedDeriv 1 w = wBetaDeriv β := by
    rw [iteratedDeriv_one]; ext x; exact (hasDerivAt_wBeta β x).deriv
  have hper : iteratedDeriv 1 w (2 * π) = iteratedDeriv 1 w 0 := by
    simp [h1, wBetaDeriv]
  obtain ⟨C, hC⟩ := isCompact_Icc.exists_bound_of_continuousOn
    ((hc 3 le_rfl).continuousOn (s := Set.Icc 0 (2 * π)))
  have hbound : ∀ n : ℕ, 0 < n → ‖aCoeff β n‖ ≤ C / n := by
    intro n hn
    have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    have hI : ∫ θ in (0 : ℝ)..2 * π, Real.cos (n * θ) * w θ
        = (1 / n) ^ 3 * ∫ θ in (0 : ℝ)..2 * π, Real.sin (n * θ) * iteratedDeriv 3 w θ := by
      have e0 : w = iteratedDeriv 0 w := rfl
      conv_lhs => rw [e0]
      rw [integral_cos_mul_eq (hd 0 (by norm_num)) (hc 1 (by norm_num)) hn,
        integral_sin_mul_eq (hd 1 (by norm_num)) (hc 2 (by norm_num)) hper hn,
        integral_cos_mul_eq (hd 2 (by norm_num)) (hc 3 le_rfl) hn]
      ring
    have hJ : ‖∫ θ in (0 : ℝ)..2 * π, Real.sin (n * θ) * iteratedDeriv 3 w θ‖
        ≤ C * (2 * π) := by
      have := intervalIntegral.norm_integral_le_of_norm_le_const (a := 0) (b := 2 * π)
        (C := C) (f := fun θ => Real.sin (n * θ) * iteratedDeriv 3 w θ) fun x hx => by
          rw [Set.uIoc_of_le (by positivity)] at hx
          rw [norm_mul]
          calc _ ≤ 1 * ‖iteratedDeriv 3 w x‖ := by
                gcongr; exact Real.abs_sin_le_one _
            _ ≤ C := by rw [one_mul]; exact hC x (Set.Ioc_subset_Icc_self hx)
      rwa [sub_zero, abs_of_pos (by positivity)] at this
    have hC0 : 0 ≤ C := (norm_nonneg _).trans (hC 0 ⟨le_rfl, by positivity⟩)
    rw [Real.norm_eq_abs] at hJ
    rw [aCoeff, wHat, hI, Real.norm_eq_abs, abs_mul, abs_mul, abs_mul, abs_pow, abs_pow,
      abs_of_pos hn', abs_of_pos (by positivity : (0 : ℝ) < (2 * π)⁻¹),
      abs_of_pos (by positivity : (0 : ℝ) < 1 / n)]
    calc (n : ℝ) ^ 2 * ((2 * π)⁻¹ * ((1 / n) ^ 3 * |_|))
        ≤ (n : ℝ) ^ 2 * ((2 * π)⁻¹ * ((1 / n) ^ 3 * (C * (2 * π)))) := by gcongr
      _ = C / n := by field_simp
  refine squeeze_zero_norm' ?_ (tendsto_const_div_atTop_nhds_zero_nat C)
  filter_upwards [Filter.eventually_gt_atTop 0] with n hn using hbound n hn

end Kinetic
end Transformer
