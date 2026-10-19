module

public import Transformer.Causal.SectionB_CentralCurvature

/-!
# Polynomial control of the original interaction-curvature derivative

ArXiv:2411.04990v2, Appendix B and §5, proof of `thm:fixed_centers`.
The actual derivative of `g=h'` contains the factor
`beta² sin²(x) - 3 beta cos(x) - 1`. After the original identity
`sin²(x)=1-cos²(x)`, this is a concave quadratic in the cosine.

The elementary interpolation identity below proves positivity
between two positive endpoint values. On the negative cosine
interval, increasing the original positive temperature increases
the quadratic. Three rational bounds cover small temperatures.
For temperatures at least one, the quartic cosine bound at twice
the original interaction scale supplies the final endpoint. These
are auxiliary calculations for the original far-interaction bound;
no original interaction or temperature hypothesis is redefined.
-/

@[expose] public section

open Real

namespace Transformer.Causal

/-- Positive endpoints control the actual concave cosine polynomial in the derivative of the original curvature.
Source: arXiv:2411.04990v2, Appendix B, the original `g=h'` used in §5's proof of `thm:fixed_centers`. -/
theorem curvature_polynomial_pos_between (β q u : ℝ) (hq : -1 < q)
    (hu : -1 ≤ u) (huq : u ≤ q) (hleft : 0 < 3 * β - 1)
    (hright : 0 < β ^ 2 * (1 - q ^ 2) - 3 * β * q - 1) :
    0 < β ^ 2 * (1 - u ^ 2) - 3 * β * u - 1 := by
  have hqu : 0 ≤ q - u := by linarith
  have hu1 : 0 ≤ u + 1 := by linarith
  have hq1 : 0 < q + 1 := by linarith
  have hsum : 0 < (q - u) * (3 * β - 1) +
      (u + 1) * (β ^ 2 * (1 - q ^ 2) - 3 * β * q - 1) := by
    rcases lt_or_eq_of_le huq with h | h
    · have hp := mul_pos (show 0 < q - u by linarith) hleft
      have hn := mul_nonneg hu1 hright.le
      linarith
    · rw [h]
      simpa only [sub_self, zero_mul, zero_add] using mul_pos hq1 hright
  have hid : (q + 1) * (β ^ 2 * (1 - u ^ 2) - 3 * β * u - 1) =
      (q - u) * (3 * β - 1) + (u + 1) * (β ^ 2 * (1 - q ^ 2) - 3 * β * q - 1) +
        β ^ 2 * (q + 1) * (u + 1) * (q - u) := by ring
  have hn : 0 ≤ β ^ 2 * (q + 1) * (u + 1) * (q - u) := by positivity
  apply (mul_pos_iff_of_pos_left hq1).mp
  rw [hid]
  linarith

/-- The actual curvature-derivative polynomial increases with positive temperature at every negative semicircle cosine.
Source: arXiv:2411.04990v2, Appendix B, the derivative calculation used for the far terms in §5, `thm:fixed_centers`. -/
theorem curvature_polynomial_mono_temperature (b β u : ℝ) (hb : 0 ≤ b) (hβ : b ≤ β)
    (hu : -1 ≤ u) (hu0 : u ≤ 0) :
    b ^ 2 * (1 - u ^ 2) - 3 * b * u - 1 ≤ β ^ 2 * (1 - u ^ 2) - 3 * β * u - 1 := by
  have hu2 : 0 ≤ 1 - u ^ 2 := by nlinarith
  have hb2 : 0 ≤ β ^ 2 - b ^ 2 := by nlinarith
  have h1 := mul_nonneg hb2 hu2
  have h2 := mul_nonneg (sub_nonneg.mpr hβ) (neg_nonneg.mpr hu0)
  nlinarith

/-- The first actual negative-cosine range has positive original curvature-derivative factor for temperatures at least two fifths.
Source: arXiv:2411.04990v2, Appendix B and §5, the far-interaction calculation in `thm:fixed_centers`. -/
theorem curvature_polynomial_pos_four_fifths (β u : ℝ) (hβ : 2 / 5 ≤ β)
    (hu : -1 ≤ u) (huq : u ≤ -(4 / 5 : ℝ)) :
    0 < β ^ 2 * (1 - u ^ 2) - 3 * β * u - 1 := by
  have hp := curvature_polynomial_pos_between (2 / 5) (-(4 / 5)) u
    (by norm_num) hu huq (by norm_num) (by norm_num)
  exact hp.trans_le (curvature_polynomial_mono_temperature (2 / 5) β u
    (by norm_num) hβ hu (by linarith))

/-- The second actual negative-cosine range has positive original curvature-derivative factor for temperatures at least one half.
Source: arXiv:2411.04990v2, Appendix B and §5, the far-interaction calculation in `thm:fixed_centers`. -/
theorem curvature_polynomial_pos_three_fifths (β u : ℝ) (hβ : 1 / 2 ≤ β)
    (hu : -1 ≤ u) (huq : u ≤ -(3 / 5 : ℝ)) :
    0 < β ^ 2 * (1 - u ^ 2) - 3 * β * u - 1 := by
  have hp := curvature_polynomial_pos_between (1 / 2) (-(3 / 5)) u
    (by norm_num) hu huq (by norm_num) (by norm_num)
  exact hp.trans_le (curvature_polynomial_mono_temperature (1 / 2) β u
    (by norm_num) hβ hu (by linarith))

/-- The third actual negative-cosine range has positive original curvature-derivative factor for temperatures at least three quarters.
Source: arXiv:2411.04990v2, Appendix B and §5, the far-interaction calculation in `thm:fixed_centers`. -/
theorem curvature_polynomial_pos_three_tenths (β u : ℝ) (hβ : 3 / 4 ≤ β)
    (hu : -1 ≤ u) (huq : u ≤ -(3 / 10 : ℝ)) :
    0 < β ^ 2 * (1 - u ^ 2) - 3 * β * u - 1 := by
  have hp := curvature_polynomial_pos_between (3 / 4) (-(3 / 10)) u
    (by norm_num) hu huq (by norm_num) (by norm_num)
  exact hp.trans_le (curvature_polynomial_mono_temperature (3 / 4) β u
    (by norm_num) hβ hu (by linarith))

/-- The quartic cosine endpoint at twice the original interaction scale is strictly above minus one.
Source: arXiv:2411.04990v2, Appendix B, the original cosine Taylor bound used in §5, `thm:fixed_centers`. -/
theorem curvature_quartic_endpoint_gt_neg_one (β : ℝ) (hβ : 1 ≤ β) :
    -1 < 1 - 2 / β + 2 / (3 * β ^ 2) := by
  have hβ0 : 0 < β := by linarith
  have hid : 3 * β ^ 2 * ((1 - 2 / β + 2 / (3 * β ^ 2)) - (-1)) =
      6 * β * (β - 1) + 2 := by field_simp; ring
  have hp : 0 < 3 * β ^ 2 * ((1 - 2 / β + 2 / (3 * β ^ 2)) - (-1)) := by
    rw [hid]
    nlinarith [mul_nonneg hβ0.le (sub_nonneg.mpr hβ)]
  exact sub_pos.mp ((mul_pos_iff_of_pos_left (show 0 < 3 * β ^ 2 by positivity)).mp hp)

/-- The actual curvature-derivative polynomial is positive at the quartic cosine endpoint for original temperatures at least one.
Source: arXiv:2411.04990v2, Appendix B and §5, the far-interaction calculation in `thm:fixed_centers`. -/
theorem curvature_polynomial_pos_quartic_endpoint (β : ℝ) (hβ : 1 ≤ β) :
    0 < β ^ 2 * (1 - (1 - 2 / β + 2 / (3 * β ^ 2)) ^ 2) -
      3 * β * (1 - 2 / β + 2 / (3 * β ^ 2)) - 1 := by
  have hβ0 : 0 < β := by linarith
  apply (mul_pos_iff_of_pos_left (show 0 < 9 * β ^ 2 by positivity)).mp
  have hid : 9 * β ^ 2 * (β ^ 2 * (1 - (1 - 2 / β + 2 / (3 * β ^ 2)) ^ 2) -
      3 * β * (1 - 2 / β + 2 / (3 * β ^ 2)) - 1) =
      9 * β ^ 2 * (β - 1) + 6 * β ^ 2 + 6 * β - 4 := by field_simp; ring
  rw [hid]
  nlinarith [mul_nonneg (sq_nonneg β) (sub_nonneg.mpr hβ)]

/-- The original curvature-derivative polynomial is positive throughout the cosine interval provided by the quartic bound.
Source: arXiv:2411.04990v2, Appendix B and §5, the far-interaction calculation in `thm:fixed_centers`. -/
theorem curvature_polynomial_pos_quartic (β u : ℝ) (hβ : 1 ≤ β)
    (hu : -1 ≤ u) (huq : u ≤ 1 - 2 / β + 2 / (3 * β ^ 2)) :
    0 < β ^ 2 * (1 - u ^ 2) - 3 * β * u - 1 :=
  curvature_polynomial_pos_between β _ u (curvature_quartic_endpoint_gt_neg_one β hβ)
    hu huq (by linarith) (curvature_polynomial_pos_quartic_endpoint β hβ)

/-- The actual original trigonometric derivative factor equals its displayed cosine polynomial exactly.
Source: arXiv:2411.04990v2, Appendix B, the original trigonometric interaction used in §5, `thm:fixed_centers`. -/
theorem g_pot_derivative_factor_eq_polynomial (β x : ℝ) :
    β ^ 2 * Real.sin x ^ 2 - 3 * β * Real.cos x - 1 =
      β ^ 2 * (1 - Real.cos x ^ 2) - 3 * β * Real.cos x - 1 := by
  linear_combination β ^ 2 * Real.sin_sq_add_cos_sq x

/-- The quartic cosine bound implies positivity of the actual original curvature-derivative factor.
Source: arXiv:2411.04990v2, Appendix B and §5, the actual far-curvature calculation in `thm:fixed_centers`. -/
theorem g_pot_derivative_factor_pos_of_cos_quartic (β x : ℝ) (hβ : 1 ≤ β)
    (hx : Real.cos x ≤ 1 - 2 / β + 2 / (3 * β ^ 2)) :
    0 < β ^ 2 * Real.sin x ^ 2 - 3 * β * Real.cos x - 1 := by
  rw [g_pot_derivative_factor_eq_polynomial]
  exact curvature_polynomial_pos_quartic β (Real.cos x) hβ (Real.neg_one_le_cos x) hx

/-- Actual negative cosines jointly realize the small-temperature endpoint and monotonicity hypotheses.
Source: arXiv:2411.04990v2, Appendix B, the original far-interaction derivative used in §5. -/
example : (0 : ℝ) ≤ 2 / 5 ∧ (2 / 5 : ℝ) ≤ 1 / 2 ∧ (-1 : ℝ) ≤ -1 ∧
    (-1 : ℝ) ≤ -(4 / 5) ∧ (-1 : ℝ) ≤ -(3 / 5) ∧ (-1 : ℝ) ≤ -(3 / 10) ∧
    (1 / 2 : ℝ) ≤ 3 / 4 ∧ (-1 : ℝ) ≤ 0 := by norm_num

/-- Unit original temperature and the actual antipodal cosine satisfy all three rational comparison hypotheses together.
Source: arXiv:2411.04990v2, Appendix B, the original semicircle endpoints used in §5. -/
example : (2 / 5 : ℝ) ≤ 1 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (3 / 4 : ℝ) ≤ 1 ∧
    (-1 : ℝ) ≤ -1 ∧ (-1 : ℝ) ≤ -(4 / 5) ∧ (-1 : ℝ) ≤ -(3 / 5) ∧
    (-1 : ℝ) ≤ -(3 / 10) := by norm_num

/-- The actual antipodal angle satisfies the quartic comparison at original temperature one.
Source: arXiv:2411.04990v2, Appendix B, the original circle interaction used in §5. -/
example : (1 : ℝ) ≤ 1 ∧ Real.cos Real.pi ≤ 1 - 2 / 1 + 2 / (3 * (1 : ℝ) ^ 2) := by
  rw [Real.cos_pi]
  norm_num

/-- An actual positive original temperature jointly satisfies the interpolation and quartic hypotheses.
Source: arXiv:2411.04990v2, Appendix B, the original cosine polynomial used in §5. -/
example : (-1 : ℝ) < -(1 / 3) ∧ (-1 : ℝ) ≤ -(1 / 2) ∧
    (-(1 / 2) : ℝ) ≤ -(1 / 3) ∧ (0 : ℝ) < 3 * 1 - 1 ∧
    (0 : ℝ) < 1 ^ 2 * (1 - (-(1 / 3) : ℝ) ^ 2) - 3 * 1 * (-(1 / 3)) - 1 ∧
    (1 : ℝ) ≤ 1 ∧ (-(1 / 2) : ℝ) ≤ 1 - 2 / 1 + 2 / (3 * 1 ^ 2) := by norm_num

end Transformer.Causal
