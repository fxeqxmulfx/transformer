/-
# The original interaction's actual logarithmic force slope

ArXiv:2411.04990v2, Appendix B and §5, the force-window
comparison in `thm:fixed_centers`.
On the open original semicircle, the exact ratio `g/h` is
`cos(x)/sin(x)-beta sin(x)`. Its actual derivative is
`(-1-beta cos(x) sin²(x))/sin²(x)`.
The proved numerator bounds give monotonicity throughout that
semicircle at temperatures at most two, and on its front half
at every original nonnegative temperature.

The original peak equation and elementary sine and cosine
bounds also give `1<=(beta+1/2) tau²`, controlling the inverse
original peak. On the original central interval starting at
half the actual peak, the actual logarithmic force slope is
strictly less than `2/tau`. All identities concern the original
interaction functions, rather than a replacement potential.
-/

import Transformer.Causal.SectionB_LogSlopeAlgebra

open Real Set

namespace Transformer.Causal

/-- The actual ratio of original curvature to original force wherever the original sine is nonzero.
Source: arXiv:2411.04990v2, Appendix B, the original `g=h'` used in §5, `thm:fixed_centers`. -/
noncomputable def interactionLogSlope (β x : ℝ) : ℝ := Real.cos x / Real.sin x - β * Real.sin x

/-- The original curvature equals the original force times its exact original logarithmic slope.
Source: arXiv:2411.04990v2, Appendix B and §5, the original force and curvature factors in `thm:fixed_centers`. -/
theorem g_pot_eq_h_pot_mul_logSlope (β x : ℝ) (hs : Real.sin x ≠ 0) :
    g_pot β x = h_pot β x * interactionLogSlope β x := by
  unfold g_pot h_pot interactionLogSlope
  field_simp

/-- The actual original logarithmic force slope has its exact original scalar derivative.
Source: arXiv:2411.04990v2, Appendix B and §5, the original interaction slope used in `thm:fixed_centers`. -/
theorem hasDerivAt_interactionLogSlope (β x : ℝ) (hs : Real.sin x ≠ 0) :
    HasDerivAt (interactionLogSlope β) ((-1 - β * Real.cos x * Real.sin x ^ 2) / Real.sin x ^ 2) x := by
  have h := ((Real.hasDerivAt_cos x).fun_div (Real.hasDerivAt_sin x) hs).fun_sub
    ((Real.hasDerivAt_sin x).const_mul β)
  apply h.congr_deriv
  field_simp
  nlinarith only [Real.sin_sq_add_cos_sq x]

/-- The actual derivative of the original logarithmic force slope is its proved trigonometric quotient.
Source: arXiv:2411.04990v2, Appendix B and §5, the actual original interaction slope in `thm:fixed_centers`. -/
theorem deriv_interactionLogSlope (β x : ℝ) (hs : Real.sin x ≠ 0) :
    deriv (interactionLogSlope β) x = (-1 - β * Real.cos x * Real.sin x ^ 2) / Real.sin x ^ 2 :=
  (hasDerivAt_interactionLogSlope β x hs).deriv

/-- The actual original logarithmic force slope vanishes at the actual original interaction peak.
Source: arXiv:2411.04990v2, Appendix B, the original zero derivative at the peak used in §5, `thm:fixed_centers`. -/
theorem interactionLogSlope_peak_eq_zero (β : ℝ) (hβ : 0 < β) : interactionLogSlope β (interactionPeak β) = 0 := by
  have ht := interactionPeak_properties β hβ
  have hs := Real.sin_pos_of_pos_of_lt_pi ht.1 (by linarith [Real.pi_pos])
  have hp : 0 < h_pot β (interactionPeak β) := mul_pos (Real.exp_pos _) hs
  have hz := g_pot_peak_eq_zero β hβ
  rw [g_pot_eq_h_pot_mul_logSlope β (interactionPeak β) hs.ne', mul_eq_zero] at hz
  exact hz.resolve_left hp.ne'

/-- At original temperatures at most two, the actual original logarithmic force slope decreases throughout the open original semicircle.
Source: arXiv:2411.04990v2, Appendix B and §5, the force-window comparison for `thm:fixed_centers`. -/
theorem interactionLogSlope_antitone_small_temperature (β : ℝ) (hβ0 : 0 ≤ β) (hβ2 : β ≤ 2) :
    AntitoneOn (interactionLogSlope β) (Ioo 0 Real.pi) := by
  have hd : ∀ x ∈ Ioo (0 : ℝ) Real.pi, HasDerivAt (interactionLogSlope β)
      ((-1 - β * Real.cos x * Real.sin x ^ 2) / Real.sin x ^ 2) x :=
    fun x hx => hasDerivAt_interactionLogSlope β x (Real.sin_pos_of_pos_of_lt_pi hx.1 hx.2).ne'
  apply antitoneOn_of_deriv_nonpos (convex_Ioo _ _) (fun x hx => (hd x hx).continuousAt.continuousWithinAt)
  · intro x hx
    rw [interior_Ioo] at hx
    exact (hd x hx).differentiableAt.differentiableWithinAt
  · intro x hx
    rw [interior_Ioo] at hx
    rw [(hd x hx).deriv]
    exact div_nonpos_of_nonpos_of_nonneg (logSlope_derivative_numerator_nonpos β hβ0 hβ2 x) (sq_nonneg _)

/-- The actual original logarithmic force slope decreases on the front original semicircle at every original nonnegative temperature.
Source: arXiv:2411.04990v2, Appendix B and §5, the original central force comparison in `thm:fixed_centers`. -/
theorem interactionLogSlope_antitone_front (β : ℝ) (hβ : 0 ≤ β) :
    AntitoneOn (interactionLogSlope β) (Ioo 0 (Real.pi / 2)) := by
  have hd : ∀ x ∈ Ioo (0 : ℝ) (Real.pi / 2), HasDerivAt (interactionLogSlope β)
      ((-1 - β * Real.cos x * Real.sin x ^ 2) / Real.sin x ^ 2) x := by
    intro x hx
    exact hasDerivAt_interactionLogSlope β x
      (Real.sin_pos_of_pos_of_lt_pi hx.1 (by linarith [hx.2, Real.pi_pos])).ne'
  apply antitoneOn_of_deriv_nonpos (convex_Ioo _ _) (fun x hx => (hd x hx).continuousAt.continuousWithinAt)
  · intro x hx
    rw [interior_Ioo] at hx
    exact (hd x hx).differentiableAt.differentiableWithinAt
  · intro x hx
    rw [interior_Ioo] at hx
    rw [(hd x hx).deriv]
    have hc : 0 < Real.cos x := Real.cos_pos_of_mem_Ioo ⟨by linarith [hx.1, Real.pi_pos], hx.2⟩
    apply div_nonpos_of_nonpos_of_nonneg _ (sq_nonneg _)
    nlinarith [mul_nonneg (mul_nonneg hβ hc.le) (sq_nonneg (Real.sin x))]

/-- The actual original peak equation implies the inverse-square lower-scale bound directly, at every original positive temperature.
Source: arXiv:2411.04990v2, Appendix B, the lower original peak-scale comparison in `lemma:interaction`, used in §5. -/
theorem interactionPeak_sq_lower_scale (β : ℝ) (hβ : 0 < β) :
    1 ≤ (β + 1 / 2) * interactionPeak β ^ 2 := by
  have ht := interactionPeak_properties β hβ
  have hz := g_pot_peak_eq_zero β hβ
  rw [g_pot, mul_eq_zero] at hz
  have heq := hz.resolve_left (Real.exp_ne_zero _)
  have hsin := mul_le_mul_of_nonneg_left (sin_sq_lt_sq ht.1).le hβ.le
  nlinarith [Real.one_sub_sq_div_two_le_cos (x := interactionPeak β)]

/-- The square of the actual original inverse peak is at most the actual original temperature plus one half.
Source: arXiv:2411.04990v2, Appendix B, the original peak-scale estimate used for the force window in §5. -/
theorem inv_interactionPeak_sq_le (β : ℝ) (hβ : 0 < β) :
    (1 / interactionPeak β) ^ 2 ≤ β + 1 / 2 := by
  rw [div_pow, one_pow, div_le_iff₀ (sq_pos_of_pos (interactionPeak_properties β hβ).1)]
  exact interactionPeak_sq_lower_scale β hβ

/-- On the actual positive front semicircle, the original logarithmic force slope is strictly less than the inverse original angle.
Source: arXiv:2411.04990v2, Appendix B and §5, the original central force comparison in `thm:fixed_centers`. -/
theorem interactionLogSlope_lt_inverse (β : ℝ) (hβ : 0 < β) (x : ℝ) (hx : x ∈ Ioo 0 (Real.pi / 2)) :
    interactionLogSlope β x < 1 / x := by
  have hs : 0 < Real.sin x := Real.sin_pos_of_pos_of_lt_pi hx.1 (by linarith [hx.2, Real.pi_pos])
  have hc : 0 < Real.cos x := Real.cos_pos_of_mem_Ioo ⟨by linarith [hx.1, Real.pi_pos], hx.2⟩
  have ht := Real.le_tan hx.1.le hx.2
  rw [Real.tan_eq_sin_div_cos, le_div_iff₀ hc] at ht
  have hcot : Real.cos x / Real.sin x ≤ 1 / x := by
    rw [div_le_div_iff₀ hs hx.1]
    nlinarith only [ht]
  have hp := mul_pos hβ hs
  unfold interactionLogSlope
  linarith

/-- From half the actual original peak to that peak, the actual original logarithmic force slope is strictly less than twice the actual inverse peak.
Source: arXiv:2411.04990v2, §5, the central half of the original force-window comparison in `thm:fixed_centers`. -/
theorem interactionLogSlope_core_bound (β : ℝ) (hβ : 0 < β) (x : ℝ)
    (hx : x ∈ Icc (interactionPeak β / 2) (interactionPeak β)) : interactionLogSlope β x < 2 / interactionPeak β := by
  have ht := interactionPeak_properties β hβ
  have hx0 : 0 < x := by linarith [hx.1]
  have hfront := interactionLogSlope_lt_inverse β hβ x ⟨hx0, by linarith [hx.2]⟩
  have hinv : 1 / x ≤ 2 / interactionPeak β := by
    rw [div_le_div_iff₀ hx0 ht.1]
    nlinarith only [hx.1]
  exact hfront.trans_le hinv

/-- Actual unit temperature, an actual front angle, and the actual half-peak angle jointly realize all original logarithmic-slope hypotheses.
Source: arXiv:2411.04990v2, Appendix B and §5, the actual original force-window geometry. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ Real.sin (Real.pi / 4) ≠ 0 ∧
    Real.pi / 4 ∈ Ioo (0 : ℝ) Real.pi ∧ Real.pi / 4 ∈ Ioo (0 : ℝ) (Real.pi / 2) ∧
    interactionPeak 1 / 2 ∈ Icc (interactionPeak 1 / 2) (interactionPeak 1) := by
  have ht := (interactionPeak_properties 1 one_pos).1
  have hs : 0 < Real.sin (Real.pi / 4) := Real.sin_pos_of_pos_of_lt_pi
    (by linarith [Real.pi_pos]) (by linarith [Real.pi_pos])
  exact ⟨zero_le_one, by norm_num, one_pos, hs.ne', ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩,
    ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩, ⟨le_rfl, by linarith⟩⟩

end Transformer.Causal
