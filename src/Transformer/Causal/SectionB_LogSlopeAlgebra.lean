/-
# Polynomial bounds for the original interaction's logarithmic slope

ArXiv:2411.04990v2, Appendix B and §5, the actual original
force-window comparison in `thm:fixed_centers`.
The original slope `h'/h` on the open semicircle is
`cos(x)/sin(x)-beta sin(x)`. Its derivative reduces to
`-1/sin²(x)-beta cos(x)`. On the back semicircle, the exact
unit-circle relation controls the relevant quadratic expression.
The calculation below proves it strictly negative for beta at
least two and slope parameter squared at most `beta+1/2`.

At twice the actual original peak, the slope bound instead
reduces to `4c³-2c²+1>=2cb`, with the actual half-angle relation
`2b²=1+c`. These are explicit polynomial estimates for the
original trigonometric factors; no force or dynamics is changed.
-/

import Transformer.Causal.SectionB_DoublePeak

open Real Set

namespace Transformer.Causal

/-- The original negative-cosine derivative factor has the unit-interval cubic bound needed at temperatures at most two.
Source: arXiv:2411.04990v2, Appendix B and §5, auxiliary algebra for the original interaction slope in `thm:fixed_centers`. -/
theorem unit_interval_sub_cube_le_half (u : ℝ) (hu : u ∈ Icc 0 1) : u - u ^ 3 ≤ 1 / 2 := by
  have h1 := mul_nonneg (sq_nonneg (u - 1 / 2)) (show 0 ≤ u + 1 by linarith [hu.1])
  have h2 : 0 ≤ (1 / 4 : ℝ) * (1 - u) := mul_nonneg (by norm_num) (sub_nonneg.mpr hu.2)
  nlinarith

/-- At negative original cosine and temperature at least two, the actual original curvature plus the indicated force term is strictly negative.
Source: arXiv:2411.04990v2, Appendix B and §5, the original unit-circle factors in the far-force comparison for `thm:fixed_centers`. -/
theorem negative_cosine_slope_factor_lt (β A c s : ℝ) (hβ : 2 ≤ β) (hc : c ≤ 0)
    (hunit : c ^ 2 + s ^ 2 = 1) (hA : A ^ 2 ≤ β + 1 / 2) : c - β * s ^ 2 + A * s < 0 := by
  let u := -c
  have hu0 : 0 ≤ u := neg_nonneg.mpr hc
  have hu2 : u ^ 2 ≤ 1 := by dsimp [u]; nlinarith [sq_nonneg s]
  have hu1 : u ≤ 1 := by nlinarith
  have hgap : 0 ≤ 1 - u := sub_nonneg.mpr hu1
  have hs2 : s ^ 2 = 1 - u ^ 2 := by dsimp [u]; nlinarith
  have hu3 : u ^ 3 ≤ u := by nlinarith [mul_nonneg hu0 (sub_nonneg.mpr hu2)]
  have hβgap : 0 ≤ β - 2 := by linarith
  have hpoly : 0 ≤ (1 / 2 : ℝ) + 9 / 2 * u - 4 * u ^ 3 := by nlinarith
  have hid2 : (2 * s ^ 2 + u) ^ 2 - (5 / 2 : ℝ) * s ^ 2 - 1 =
      (1 - u) * (1 / 2 + 9 / 2 * u - 4 * u ^ 3) := by rw [hs2]; ring
  have hR2 : 1 ≤ (2 * s ^ 2 + u) ^ 2 - (5 / 2 : ℝ) * s ^ 2 := by
    have hp := mul_nonneg hgap hpoly
    rw [← hid2] at hp
    linarith
  have hbracket : 0 ≤ (β + 2) * s ^ 2 + 2 * u - 1 := by
    have hid : (β + 2) * s ^ 2 + 2 * u - 1 =
        1 + (β - 2) * s ^ 2 + 2 * (1 - u) * (2 * u + 1) := by rw [hs2]; ring
    rw [hid]
    positivity
  have hdiff := mul_nonneg (mul_nonneg hβgap (sq_nonneg s)) hbracket
  have hident : (β * s ^ 2 + u) ^ 2 - (β + 1 / 2) * s ^ 2 -
      ((2 * s ^ 2 + u) ^ 2 - (5 / 2 : ℝ) * s ^ 2) =
        (β - 2) * s ^ 2 * ((β + 2) * s ^ 2 + 2 * u - 1) := by ring
  rw [← hident] at hdiff
  have hR : 1 ≤ (β * s ^ 2 + u) ^ 2 - (β + 1 / 2) * s ^ 2 := by linarith
  have hL : 1 ≤ β * s ^ 2 + u := by
    have hid : β * s ^ 2 + u = 1 + (β - 2) * s ^ 2 + (1 - u) * (2 * u + 1) := by rw [hs2]; ring
    rw [hid]
    have hp := mul_nonneg hgap (show 0 ≤ 2 * u + 1 by linarith)
    nlinarith [mul_nonneg hβgap (sq_nonneg s)]
  have hAsq := mul_le_mul_of_nonneg_right hA (sq_nonneg s)
  have hstrict : A * s < β * s ^ 2 + u := by
    by_contra h
    have hm := mul_self_le_mul_self (show 0 ≤ β * s ^ 2 + u by linarith) (le_of_not_gt h)
    nlinarith only [hR, hAsq, hm]
  have hcu : c = -u := by dsimp [u]; ring
  rw [hcu]
  linarith

/-- The actual original half-angle relation controls the exact cubic factor of the slope at twice the original peak.
Source: arXiv:2411.04990v2, Appendix B and §5, auxiliary algebra for the original double-peak slope in `thm:fixed_centers`. -/
theorem double_peak_cosine_polynomial_lower (c b : ℝ) (hc : c ∈ Icc 0 1)
    (hb2 : 2 * b ^ 2 = 1 + c) : 2 * c * b ≤ 4 * c ^ 3 - 2 * c ^ 2 + 1 := by
  have hb1 : b ≤ 1 := by nlinarith [hc.2]
  have hcb := mul_le_mul_of_nonneg_left hb1 (show 0 ≤ 2 * c by linarith [hc.1])
  by_cases hlo : c ≤ 1 / 2
  · have h1 : 2 * c - 1 ≤ 0 := by linarith
    have h2 : 2 * c ^ 2 - 1 ≤ 0 := by nlinarith [hc.1]
    have hp := mul_nonneg_of_nonpos_of_nonpos h1 h2
    nlinarith
  by_cases hhi : 3 / 4 ≤ c
  · have h1 : 0 ≤ 2 * c - 1 := by linarith
    have h2 : 0 ≤ 2 * c ^ 2 - 1 := by nlinarith
    have hp := mul_nonneg h1 h2
    nlinarith
  · let t := 2 * c - 1
    have ht : 0 ≤ t := by dsimp [t]; linarith
    have hpositive : 0 ≤ (t - 1 / 4) ^ 2 + t ^ 3 / 2 := by positivity
    have hid : 4 * c ^ 3 - 2 * c ^ 2 + 1 - 2 * c + 1 / 16 = (t - 1 / 4) ^ 2 + t ^ 3 / 2 := by
      dsimp [t]
      ring
    rw [← hid] at hpositive
    have hgap : (1 - c) / 4 ≤ 1 - b := by nlinarith [sq_nonneg (b - 1)]
    have hm := mul_le_mul_of_nonneg_left hgap (show 0 ≤ 2 * c by linarith [hc.1])
    have hcross := mul_nonneg (show 0 ≤ c - 1 / 2 by linarith) (show 0 ≤ 3 / 4 - c by linarith)
    have hcp : (3 / 16 : ℝ) ≤ c * (1 - c) := by nlinarith
    nlinarith only [hpositive, hm, hcp]

/-- At original temperatures at most two, the numerator of the actual original logarithmic-slope derivative is nonpositive everywhere.
Source: arXiv:2411.04990v2, Appendix B and §5, the original interaction slope used to compare force windows in `thm:fixed_centers`. -/
theorem logSlope_derivative_numerator_nonpos (β : ℝ) (hβ0 : 0 ≤ β) (hβ2 : β ≤ 2) (x : ℝ) :
    -1 - β * Real.cos x * Real.sin x ^ 2 ≤ 0 := by
  by_cases hc : 0 ≤ Real.cos x
  · nlinarith [mul_nonneg (mul_nonneg hβ0 hc) (sq_nonneg (Real.sin x))]
  · let u := -Real.cos x
    have hu : u ∈ Icc 0 1 := ⟨by dsimp [u]; linarith, by dsimp [u]; linarith [Real.neg_one_le_cos x]⟩
    have hcube := unit_interval_sub_cube_le_half u hu
    have hid : (-Real.cos x) * Real.sin x ^ 2 = u - u ^ 3 := by
      dsimp [u]
      linear_combination (-Real.cos x) * Real.sin_sq_add_cos_sq x
    rw [← hid] at hcube
    have hm := mul_le_mul_of_nonneg_left hcube hβ0
    nlinarith only [hm, hβ2]

/-- The actual original negative-cosine trigonometric factor satisfies the proved strict force-slope bound at temperatures at least two.
Source: arXiv:2411.04990v2, Appendix B and §5, the actual far-semicircle factors of `thm:fixed_centers`. -/
theorem negative_cosine_trig_slope_factor_lt (β A x : ℝ) (hβ : 2 ≤ β) (hc : Real.cos x ≤ 0)
    (hA : A ^ 2 ≤ β + 1 / 2) : Real.cos x - β * Real.sin x ^ 2 + A * Real.sin x < 0 :=
  negative_cosine_slope_factor_lt β A (Real.cos x) (Real.sin x) hβ hc (Real.cos_sq_add_sin_sq x) hA

/-- At an actual original negative cosine and positive sine, the original logarithmic force slope has the proved strict upper bound.
Source: arXiv:2411.04990v2, Appendix B and §5, the original far-interaction slope used in `thm:fixed_centers`. -/
theorem negative_cosine_logSlope_lt (β A x : ℝ) (hβ : 2 ≤ β) (hc : Real.cos x ≤ 0)
    (hs : 0 < Real.sin x) (hA : A ^ 2 ≤ β + 1 / 2) :
    Real.cos x / Real.sin x - β * Real.sin x < -A := by
  have h := negative_cosine_trig_slope_factor_lt β A x hβ hc hA
  rw [sub_lt_iff_lt_add, div_lt_iff₀ hs]
  nlinarith only [h]

/-- Actual unit-circle cosine and sine data jointly realize the back-semicircle slope-factor hypotheses.
Source: arXiv:2411.04990v2, Appendix B, the actual original trigonometric factors used in §5. -/
example : (0 : ℝ) ∈ Icc 0 1 ∧ (2 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 0 ∧
    (0 : ℝ) ^ 2 + 1 ^ 2 = 1 ∧ (1 : ℝ) ^ 2 ≤ 2 + 1 / 2 := by norm_num

/-- An actual positive half-angle cosine jointly satisfies the cubic comparison's original unit-interval and half-angle premises.
Source: arXiv:2411.04990v2, Appendix B, the original peak half-angle identity used in §5. -/
example : (3 / 4 : ℝ) ∈ Icc 0 1 ∧ 2 * Real.sqrt (7 / 8 : ℝ) ^ 2 = 1 + (3 / 4 : ℝ) := by
  rw [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 7 / 8)]
  norm_num

/-- Actual original temperatures and the actual antipodal angle jointly realize both temperature cases for the original slope numerators.
Source: arXiv:2411.04990v2, Appendix B, the actual original semicircle slope calculation used in §5. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧ (2 : ℝ) ≤ 2 ∧ Real.cos Real.pi ≤ 0 ∧
    (1 : ℝ) ^ 2 ≤ 2 + 1 / 2 := by rw [Real.cos_pi]; norm_num

/-- The actual original half-circle endpoint has negative-or-zero cosine and strictly positive sine in the original large-temperature case.
Source: arXiv:2411.04990v2, Appendix B, the actual original slope factors used in §5. -/
example : (2 : ℝ) ≤ 2 ∧ Real.cos (Real.pi / 2) ≤ 0 ∧ 0 < Real.sin (Real.pi / 2) ∧
    (1 : ℝ) ^ 2 ≤ 2 + 1 / 2 := by rw [Real.cos_pi_div_two, Real.sin_pi_div_two]; norm_num

end Transformer.Causal
