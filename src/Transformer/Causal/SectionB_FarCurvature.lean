module

public import Transformer.Causal.SectionB_CurvaturePolynomial

/-!
# The increasing far branch of the original interaction curvature

ArXiv:2411.04990v2, Appendix B and §5, proof of `thm:fixed_centers`.
The original frozen-center stability proof compares every far
negative interaction with `g` at the minimum intercluster distance.
That comparison needs the increasing branch of the original `g`.

We prove that its actual derivative is positive between twice
the original scale `beta^(-1/2)` and pi, for every positive beta.
For beta at least one, the original quartic cosine bound and the
concave derivative polynomial prove the assertion. For smaller
beta, the distance is necessarily near pi: three rational cosine
bounds cover the remaining ranges. The statement remains valid
when the indicated interval is empty. A unit-temperature example
below realizes a nonempty interval and both comparison distances.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

/-- The square of the original positive interaction scale is exactly the inverse original temperature.
Source: arXiv:2411.04990v2, Appendix B, the `beta^(-1/2)` scale used in §5, `thm:fixed_centers`. -/
theorem interactionScale_sq (β : ℝ) (hβ : 0 < β) :
    (β ^ (-(1 / 2 : ℝ))) ^ 2 = β⁻¹ := by
  rw [← Real.rpow_natCast (β ^ (-(1 / 2 : ℝ))) 2, ← Real.rpow_mul hβ.le]
  norm_num
  rw [Real.rpow_neg_one]

/-- The actual original cosine has its quadratic antipodal upper bound.
Source: arXiv:2411.04990v2, Appendix B, the cosine Taylor bound used in §5, `thm:fixed_centers`. -/
theorem cos_le_antipodal_quadratic (x : ℝ) : Real.cos x ≤ -1 + (Real.pi - x) ^ 2 / 2 := by
  have h := Real.one_sub_sq_div_two_le_cos (x := Real.pi - x)
  rw [Real.cos_pi_sub] at h
  linarith

/-- The actual derivative factor of the original curvature is positive beyond twice the original interaction scale.
Source: arXiv:2411.04990v2, Appendix B and §5, the increasing far-curvature branch used in `thm:fixed_centers`. -/
theorem g_pot_derivative_factor_pos_far (β : ℝ) (hβ : 0 < β) (x : ℝ) (hxpi : x ≤ Real.pi)
    (hfar : 2 * β ^ (-(1 / 2 : ℝ)) ≤ x) :
    0 < β ^ 2 * Real.sin x ^ 2 - 3 * β * Real.cos x - 1 := by
  let y := β ^ (-(1 / 2 : ℝ))
  have hy : 0 < y := Real.rpow_pos_of_pos hβ _
  have hx : 0 < x := by linarith
  have hy2 : y ^ 2 = β⁻¹ := interactionScale_sq β hβ
  have hby2 : β * y ^ 2 = 1 := by rw [hy2, mul_inv_cancel₀ hβ.ne']
  have hsq := mul_self_le_mul_self (show 0 ≤ 2 * y by positivity) hfar
  have h4 : 4 ≤ β * x ^ 2 := by
    have h := mul_le_mul_of_nonneg_left hsq hβ.le
    nlinarith [hby2]
  by_cases hlarge : 1 ≤ β
  · apply g_pot_derivative_factor_pos_of_cos_quartic β x hlarge
    have hc := Real.cos_le_cos_of_nonneg_of_le_pi (show 0 ≤ 2 * y by positivity) hxpi hfar
    have ht := cos_le_quartic (2 * y)
    have hid : 1 - (2 * y) ^ 2 / 2 + (2 * y) ^ 4 / 24 =
        1 - 2 / β + 2 / (3 * β ^ 2) := by
      calc
        _ = 1 - 2 * y ^ 2 + (2 / 3) * (y ^ 2) ^ 2 := by ring
        _ = _ := by rw [hy2]; field_simp
    linarith
  rw [g_pot_derivative_factor_eq_polynomial]
  have hlow : 2 / 5 ≤ β := by
    have hsqpi := mul_self_le_mul_self hx.le hxpi
    have hx10 : x ^ 2 ≤ 10 := by nlinarith [Real.pi_lt_d2, Real.pi_pos]
    have hm := mul_le_mul_of_nonneg_left hx10 hβ.le
    nlinarith
  have hcos := cos_le_antipodal_quadratic x
  have hpi : Real.pi < 63 / 20 := by linarith [Real.pi_lt_d2]
  have hgap : 0 ≤ Real.pi - x := by linarith
  by_cases hhalf : β ≤ 1 / 2
  · have hxlo : 14 / 5 ≤ x := by
      have h := mul_le_mul_of_nonneg_left hhalf (sq_nonneg x)
      nlinarith
    have hgap2 := mul_self_le_mul_self hgap (show Real.pi - x ≤ 7 / 20 by linarith)
    apply curvature_polynomial_pos_four_fifths β (Real.cos x) hlow (Real.neg_one_le_cos x)
    nlinarith
  by_cases hthree : β ≤ 3 / 4
  · have hxlo : 23 / 10 ≤ x := by
      have h := mul_le_mul_of_nonneg_left hthree (sq_nonneg x)
      nlinarith
    have hgap2 := mul_self_le_mul_self hgap (show Real.pi - x ≤ 17 / 20 by linarith)
    apply curvature_polynomial_pos_three_fifths β (Real.cos x) (by linarith) (Real.neg_one_le_cos x)
    nlinarith
  · have hxlo : 2 ≤ x := by
      have h := mul_le_mul_of_nonneg_left (le_of_not_ge hlarge) (sq_nonneg x)
      nlinarith
    have hgap2 := mul_self_le_mul_self hgap (show Real.pi - x ≤ 23 / 20 by linarith)
    apply curvature_polynomial_pos_three_tenths β (Real.cos x) (by linarith) (Real.neg_one_le_cos x)
    nlinarith

/-- The actual original curvature derivative is positive on the nonempty interior of its far semicircle branch.
Source: arXiv:2411.04990v2, §5, the far `g` comparison in the proof of `thm:fixed_centers`. -/
theorem deriv_g_pot_pos_far (β : ℝ) (hβ : 0 < β) (x : ℝ) (hxpi : x < Real.pi)
    (hfar : 2 * β ^ (-(1 / 2 : ℝ)) ≤ x) : 0 < deriv (g_pot β) x := by
  rw [deriv_g_pot]
  have hx : 0 < x := by linarith [Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))]
  exact mul_pos (mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi hx hxpi))
    (g_pot_derivative_factor_pos_far β hβ x hxpi.le hfar)

/-- The actual original curvature strictly increases from twice the original interaction scale to the antipode.
Source: arXiv:2411.04990v2, §5, the minimum-distance comparison for negative `g` terms in `thm:fixed_centers`. -/
theorem g_pot_strictMonoOn_far (β : ℝ) (hβ : 0 < β) :
    StrictMonoOn (g_pot β) (Icc (2 * β ^ (-(1 / 2 : ℝ))) Real.pi) := by
  have hc : Continuous (g_pot β) := by unfold g_pot; fun_prop
  apply strictMonoOn_of_deriv_pos (convex_Icc _ _) hc.continuousOn
  intro x hx
  rw [interior_Icc] at hx
  exact deriv_g_pot_pos_far β hβ x hx.2 hx.1.le

/-- Every actual original far interaction has curvature bounded below by the value at its original minimum distance.
Source: arXiv:2411.04990v2, §5, the negative-curvature bound used in `thm:fixed_centers`. -/
theorem g_pot_far_lower (β : ℝ) (hβ : 0 < β) (r x : ℝ)
    (hr : 2 * β ^ (-(1 / 2 : ℝ)) ≤ r) (hrx : r ≤ x) (hx : x ≤ Real.pi) :
    g_pot β r ≤ g_pot β x :=
  (g_pot_strictMonoOn_far β hβ).monotoneOn ⟨hr, hrx.trans hx⟩ ⟨hr.trans hrx, hx⟩ hrx

/-- Every actual original far semicircle interaction has strictly negative curvature.
Source: arXiv:2411.04990v2, Appendix B and §5, the negative far interactions in `thm:fixed_centers`. -/
theorem g_pot_far_neg (β : ℝ) (hβ : 0 < β) (x : ℝ) (hxpi : x ≤ Real.pi)
    (hfar : 2 * β ^ (-(1 / 2 : ℝ)) ≤ x) : g_pot β x < 0 := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have ht := interactionPeak_properties β hβ
  have hpeak : interactionPeak β < x := by linarith [interactionPeak_lt_scale β hβ]
  have hcos := Real.cos_lt_cos_of_nonneg_of_le_pi ht.1.le hxpi hpeak
  rw [ht.2.2] at hcos
  rw [g_pot_peak_factor β hβ]
  exact mul_neg_of_pos_of_neg (Real.exp_pos _) (mul_neg_of_neg_of_pos (sub_neg.mpr hcos)
    ((interactionPeakCos_properties β hβ).2.2.2 _ (Real.neg_one_le_cos x)))

/-- The actual original far curvature lies between its minimum-distance value and zero with the precise signs needed for weighted tails.
Source: arXiv:2411.04990v2, §5, the signed far-curvature bound in `thm:fixed_centers`. -/
theorem g_pot_far_bounds (β : ℝ) (hβ : 0 < β) (r x : ℝ)
    (hr : 2 * β ^ (-(1 / 2 : ℝ)) ≤ r) (hrx : r ≤ x) (hx : x ≤ Real.pi) :
    g_pot β r ≤ g_pot β x ∧ g_pot β x < 0 :=
  ⟨g_pot_far_lower β hβ r x hr hrx hx, g_pot_far_neg β hβ x hx (hr.trans hrx)⟩

/-- Unit original temperature and distance three realize the strict far-derivative hypotheses on the actual semicircle.
Source: arXiv:2411.04990v2, Appendix B and §5, the original far-interaction branch. -/
example : (0 : ℝ) < 1 ∧ (3 : ℝ) < Real.pi ∧ 2 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 3 := by
  norm_num
  exact Real.pi_gt_three

/-- Distinct original distances two and three jointly realize the actual far-curvature comparison hypotheses.
Source: arXiv:2411.04990v2, §5, the original far-curvature lower bound in `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 2 ∧
    (2 : ℝ) ≤ 3 ∧ (3 : ℝ) ≤ Real.pi := by
  norm_num
  exact Real.pi_gt_three.le

end Transformer.Causal
