/-
# A uniform Gaussian height bound for the actual original interaction peak

ArXiv:2411.04990v2, Appendix B, the original peak and Gaussian
interaction estimates used in §5, `thm:fixed_centers`.
The original peak equation gives `beta sin²(tau)=cos(tau)=c`.
Thus the squared original peak height times beta is
`c exp(2 beta(c-1))`. The elementary inequality `c<=exp(c-1)`
and the original root equation imply that this is at most
`exp(-1)`, for every positive original temperature.

The actual original peak is consequently at most
`exp(-1/2) beta^(-1/2)`, and strictly below
`0.61 beta^(-1/2)`. The numerical constant is a proved rational
rounding of the Gaussian coefficient, rather than a changed
paper hypothesis. Original circular unimodality transports the
same bound to the magnitude of every original interaction.
-/

import Transformer.Causal.Section5_ParameterGeometry

open Real Set

namespace Transformer.Causal

/-- The original Gaussian peak coefficient has its strict rational squared upper bound, certified by the exponential series.
Source: arXiv:2411.04990v2, Appendix B, the Gaussian comparison for the original interaction in §5. -/
theorem exp_neg_one_lt_peak_height_bound_sq : Real.exp (-1) < (61 / 100 : ℝ) ^ 2 := by
  have hq := Real.quadratic_le_exp_of_nonneg (show (0 : ℝ) ≤ 1 / 4 by norm_num)
  norm_num at hq
  have hp : (41 / 32 : ℝ) ^ 4 ≤ Real.exp (1 / 4 : ℝ) ^ 4 := by gcongr
  have he : Real.exp 1 = Real.exp (1 / 4 : ℝ) ^ 4 := by rw [← Real.exp_nat_mul]; norm_num
  have hE : (10000 / 3721 : ℝ) < Real.exp 1 := by
    rw [he]
    exact lt_of_lt_of_le (by norm_num) hp
  rw [Real.exp_neg, inv_eq_one_div, div_lt_iff₀ (Real.exp_pos 1)]
  linarith

/-- The exact actual original peak equation bounds its squared height by the original Gaussian coefficient at every positive temperature.
Source: arXiv:2411.04990v2, Appendix B, the original peak equation and Gaussian interaction bound in `lemma:interaction`. -/
theorem h_pot_peak_sq_gaussian_bound (β : ℝ) (hβ : 0 < β) :
    β * h_pot β (interactionPeak β) ^ 2 ≤ Real.exp (-1) := by
  let τ := interactionPeak β
  let c := interactionPeakCos β
  have ht := interactionPeak_properties β hβ
  have hc := interactionPeakCos_properties β hβ
  have hroot : β * c ^ 2 + c - β = 0 := hc.2.2.1
  have hrel : β * Real.sin τ ^ 2 = c := by
    have hunit := Real.sin_sq_add_cos_sq τ
    rw [ht.2.2] at hunit
    linear_combination β * hunit - hroot
  have hid : β * h_pot β τ ^ 2 = c * Real.exp (2 * (β * (c - 1))) := by
    rw [h_pot, mul_pow, ht.2.2, ← Real.exp_nat_mul]
    norm_num only [Nat.cast_ofNat]
    calc
      _ = (β * Real.sin τ ^ 2) * Real.exp (2 * (β * (c - 1))) := by ring
      _ = _ := by rw [hrel]
  have hce : c ≤ Real.exp (c - 1) := by linarith [Real.add_one_le_exp (c - 1)]
  have hscore : (c - 1) + 2 * (β * (c - 1)) ≤ -1 := by
    nlinarith [mul_nonneg hβ.le (sq_nonneg (c - 1))]
  change β * h_pot β τ ^ 2 ≤ _
  calc
    _ = c * Real.exp (2 * (β * (c - 1))) := hid
    _ ≤ Real.exp (c - 1) * Real.exp (2 * (β * (c - 1))) :=
      mul_le_mul_of_nonneg_right hce (Real.exp_pos _).le
    _ = Real.exp ((c - 1) + 2 * (β * (c - 1))) := (Real.exp_add _ _).symm
    _ ≤ _ := Real.exp_le_exp.mpr hscore

/-- The exact actual original peak height is at most the Gaussian height at its original interaction scale.
Source: arXiv:2411.04990v2, Appendix B, the original peak and Gaussian interaction estimates of `lemma:interaction`. -/
theorem h_pot_peak_le_gaussian_scale (β : ℝ) (hβ : 0 < β) :
    h_pot β (interactionPeak β) ≤ Real.exp (-(1 / 2 : ℝ)) * β ^ (-(1 / 2 : ℝ)) := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have hs2 : β * (β ^ (-(1 / 2 : ℝ))) ^ 2 = 1 := by rw [interactionScale_sq β hβ, mul_inv_cancel₀ hβ.ne']
  have he : Real.exp (-(1 / 2 : ℝ)) ^ 2 = Real.exp (-1) := by
    rw [← Real.exp_nat_mul]
    congr 1
    norm_num
  have hR : β * (Real.exp (-(1 / 2 : ℝ)) * β ^ (-(1 / 2 : ℝ))) ^ 2 = Real.exp (-1) := by
    calc
      _ = Real.exp (-(1 / 2 : ℝ)) ^ 2 * (β * (β ^ (-(1 / 2 : ℝ))) ^ 2) := by ring
      _ = _ := by rw [hs2, mul_one, he]
  have hsq := h_pot_peak_sq_gaussian_bound β hβ
  rw [← hR] at hsq
  have hcompare := le_of_mul_le_mul_left hsq hβ
  have hp := mul_pos (Real.exp_pos (-(1 / 2 : ℝ))) hs
  nlinarith only [hcompare, hp]

/-- The original Gaussian coefficient is strictly less than its stated rational peak-height bound.
Source: arXiv:2411.04990v2, Appendix B, rational rounding of the original Gaussian interaction coefficient. -/
theorem exp_neg_half_lt_peak_height_bound : Real.exp (-(1 / 2 : ℝ)) < 61 / 100 := by
  have he : Real.exp (-(1 / 2 : ℝ)) ^ 2 = Real.exp (-1) := by
    rw [← Real.exp_nat_mul]
    congr 1
    norm_num
  have h := exp_neg_one_lt_peak_height_bound_sq
  rw [← he] at h
  nlinarith only [h, Real.exp_pos (-(1 / 2 : ℝ))]

/-- The actual original peak height is strictly below the rational Gaussian-height bound at every positive original temperature.
Source: arXiv:2411.04990v2, Appendix B, the original Gaussian interaction scale used in §5, `thm:fixed_centers`. -/
theorem h_pot_peak_lt_scale_bound (β : ℝ) (hβ : 0 < β) :
    h_pot β (interactionPeak β) < (61 / 100 : ℝ) * β ^ (-(1 / 2 : ℝ)) :=
  (h_pot_peak_le_gaussian_scale β hβ).trans_lt (mul_lt_mul_of_pos_right exp_neg_half_lt_peak_height_bound
    (Real.rpow_pos_of_pos hβ _))

/-- Every actual original circular interaction magnitude is bounded by the actual original peak height.
Source: arXiv:2411.04990v2, Appendix B, original circular unimodality in `lemma:interaction`, used in §5. -/
theorem abs_h_pot_le_peak (β : ℝ) (hβ : 0 < β) (x : ℝ) : |h_pot β x| ≤ h_pot β (interactionPeak β) := by
  rw [abs_h_pot_eq_h_pot_circDist]
  have ht := interactionPeak_properties β hβ
  have hm := h_pot_monotone_peak β hβ
  by_cases hx : circDist x ≤ interactionPeak β
  · exact hm.1.monotoneOn ⟨circDist_nonneg x, hx⟩ ⟨ht.1.le, le_rfl⟩ hx
  · exact hm.2.antitoneOn ⟨le_rfl, by linarith [Real.pi_pos]⟩
      ⟨le_of_not_ge hx, circDist_le_pi x⟩ (le_of_not_ge hx)

/-- Every actual original interaction magnitude obeys the same strict rational Gaussian bound at the original scale.
Source: arXiv:2411.04990v2, Appendix B and §5, the original Gaussian interaction estimate used in `thm:fixed_centers`. -/
theorem abs_h_pot_lt_scale_bound (β : ℝ) (hβ : 0 < β) (x : ℝ) :
    |h_pot β x| < (61 / 100 : ℝ) * β ^ (-(1 / 2 : ℝ)) :=
  (abs_h_pot_le_peak β hβ x).trans_lt (h_pot_peak_lt_scale_bound β hβ)

/-- The original strict frozen-center force window bounds its actual total far force by the proved Gaussian peak height.
Source: arXiv:2411.04990v2, §5, a consequence of the original `h` parameter inequality in `thm:fixed_centers`. -/
theorem frozen_force_window_lt_peak_height_bound (N β ε c : ℝ) (hβ : 0 < β)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ)))) :
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < (61 / 100 : ℝ) * β ^ (-(1 / 2 : ℝ)) :=
  hh.trans ((le_abs_self _).trans_lt (abs_h_pot_lt_scale_bound β hβ _))

/-- Original total mass at least three forces the actual original far interaction below one third of the rational peak-height bound.
Source: arXiv:2411.04990v2, §5, the original nontrivial mass regime of the force inequality in `thm:fixed_centers`. -/
theorem frozen_force_tail_lt_third_peak_height_bound (N β ε c : ℝ) (hN : 3 ≤ N) (hβ : 0 < β)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ)))) :
    h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < (61 / 300 : ℝ) * β ^ (-(1 / 2 : ℝ)) := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have hb := frozen_force_window_lt_peak_height_bound N β ε c hβ hh
  by_cases hp : 0 < h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)))
  · have hm := mul_le_mul_of_nonneg_right hN hp.le
    nlinarith only [hb, hm]
  · exact (le_of_not_gt hp).trans_lt (mul_pos (by norm_num) hs)

/-- Actual unit temperature and both original semicircle endpoints jointly realize the peak and global interaction hypotheses.
Source: arXiv:2411.04990v2, Appendix B, the original circular interaction used in §5. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ∈ Icc 0 Real.pi ∧ Real.pi ∈ Icc 0 Real.pi :=
  ⟨one_pos, ⟨le_rfl, Real.pi_pos.le⟩, ⟨Real.pi_pos.le, le_rfl⟩⟩

/-- The actual original mass-three parameter window jointly realizes both far-force height comparisons.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`, with Appendix B's original explicit parameter window. -/
example : (3 : ℝ) ≤ 3 ∧ (0 : ℝ) < 11 ∧
    3 * h_pot 11 ((142 / 25 - 1 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) <
      h_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    h_pot 11 ((142 / 25 - 1 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) <
      (61 / 300 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  exact ⟨le_rfl, hβ, hh, frozen_force_tail_lt_third_peak_height_bound 3 11 (9 / 100) (142 / 25) le_rfl hβ hh⟩

end Transformer.Causal
