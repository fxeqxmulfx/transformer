module

public import Transformer.Causal.SectionB_LogSlope

/-!
# The logarithmic force slope beyond twice the interaction peak

ArXiv:2411.04990v2, Appendix B and §5, the force comparison
in `thm:fixed_centers`.
The actual logarithmic slope of the original interaction is at
most `-1/tau` from twice its peak to the antipodal endpoint.
At twice the peak, the exact peak equation reduces this bound
to the cubic and half-angle comparison proved previously.
At temperatures at most two the slope decreases throughout
the semicircle. For larger temperatures, its front-half
monotonicity and a separate negative-cosine estimate suffice.

Together with the central bound, this proves the signed slope
comparison needed to reduce the force window to half the peak.
This reduction will preserve the original force inequality.
All bounds concern the original interaction and original peak;
no new restriction is imposed on the positive temperature.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

/-- At twice the original interaction peak, the logarithmic force slope is at most the negative inverse peak.
Source: arXiv:2411.04990v2, Appendix B and §5, an explicit comparison for the force window in `thm:fixed_centers`. -/
theorem interactionLogSlope_double_peak_le (β : ℝ) (hβ : 0 < β) :
    interactionLogSlope β (2 * interactionPeak β) ≤ -1 / interactionPeak β := by
  let τ := interactionPeak β
  have ht := interactionPeak_properties β hβ
  have hc := interactionPeakCos_properties β hβ
  have hc0 : 0 < Real.cos τ := by rw [ht.2.2]; exact hc.1
  have hc1 : Real.cos τ ≤ 1 := Real.cos_le_one τ
  have hs : 0 < Real.sin (2 * τ) := Real.sin_pos_of_pos_of_lt_pi
    (by linarith [ht.1]) (by linarith [ht.2.1])
  have hb : 0 < Real.cos (τ / 2) := Real.cos_pos_of_mem_Ioo
    ⟨by dsimp [τ]; linarith [ht.1, Real.pi_pos], by dsimp [τ]; linarith [ht.1, ht.2.1]⟩
  have hb2 : 2 * Real.cos (τ / 2) ^ 2 = 1 + Real.cos τ := by
    have h := Real.cos_two_mul (x := τ / 2)
    rw [show 2 * (τ / 2) = τ by ring] at h
    linarith
  have hp := double_peak_cosine_polynomial_lower (Real.cos τ) (Real.cos (τ / 2)) ⟨hc0.le, hc1⟩ hb2
  have hchord : Real.sin τ ≤ τ * Real.cos (τ / 2) := by
    have h := mul_le_mul_of_nonneg_right (Real.sin_le (show 0 ≤ τ / 2 by dsimp [τ]; linarith [ht.1])) hb.le
    have hi := Real.sin_two_mul (x := τ / 2)
    rw [show 2 * (τ / 2) = τ by ring] at hi
    nlinarith only [h, hi]
  have hpm := mul_le_mul_of_nonneg_left hp (show 0 ≤ τ from ht.1.le)
  have hcm := mul_le_mul_of_nonneg_left hchord (show 0 ≤ 2 * Real.cos τ by positivity)
  have hratio : Real.sin (2 * τ) / τ ≤ 4 * Real.cos τ ^ 3 - 2 * Real.cos τ ^ 2 + 1 := by
    rw [div_le_iff₀ ht.1, Real.sin_two_mul]
    nlinarith only [hpm, hcm]
  have hz := g_pot_peak_eq_zero β hβ
  rw [g_pot, mul_eq_zero] at hz
  have heq := hz.resolve_left (Real.exp_ne_zero _)
  have hfactor : Real.cos (2 * τ) - β * Real.sin (2 * τ) ^ 2 =
      -(4 * Real.cos τ ^ 3 - 2 * Real.cos τ ^ 2 + 1) := by
    rw [Real.sin_two_mul, Real.cos_two_mul]
    linear_combination 4 * Real.cos τ ^ 2 * heq
  change Real.cos (2 * τ) / Real.sin (2 * τ) - β * Real.sin (2 * τ) ≤ -1 / τ
  calc
    _ = (Real.cos (2 * τ) - β * Real.sin (2 * τ) ^ 2) / Real.sin (2 * τ) := by field_simp
    _ ≤ -1 / τ := by
      rw [div_le_iff₀ hs, hfactor]
      simp only [div_eq_mul_inv] at hratio ⊢
      nlinarith only [hratio]

/-- At temperature at least two, twice the original peak is strictly inside the front semicircle.
Source: arXiv:2411.04990v2, Appendix B, the exact peak equation used for the force comparison in §5. -/
theorem double_interactionPeak_lt_pi_div_two (β : ℝ) (hβ : 2 ≤ β) :
    2 * interactionPeak β < Real.pi / 2 := by
  have hβ0 : 0 < β := by linarith
  have ht := interactionPeak_properties β hβ0
  have hc := interactionPeakCos_properties β hβ0
  have hm := mul_le_mul_of_nonneg_right hβ
    (show 0 ≤ 1 - interactionPeakCos β ^ 2 by nlinarith [hc.1, hc.2.1])
  have hsq : 1 / 2 < interactionPeakCos β ^ 2 := by nlinarith [hc.2.2.1, hc.2.1]
  by_contra h
  have hnonpos := Real.cos_nonpos_of_pi_div_two_le_of_le (le_of_not_gt h)
    (show 2 * interactionPeak β ≤ Real.pi + Real.pi / 2 by linarith [ht.2.1, Real.pi_pos])
  rw [Real.cos_two_mul, ht.2.2] at hnonpos
  linarith

/-- Beyond twice the original peak, the logarithmic force slope is at most the negative inverse peak at every positive temperature.
Source: arXiv:2411.04990v2, Appendix B and §5, the far-force comparison for `thm:fixed_centers`. -/
theorem interactionLogSlope_far_bound (β : ℝ) (hβ : 0 < β) (x : ℝ)
    (hx : x ∈ Ico (2 * interactionPeak β) Real.pi) :
    interactionLogSlope β x ≤ -1 / interactionPeak β := by
  have ht := interactionPeak_properties β hβ
  have hx0 : 0 < x := by linarith [hx.1, ht.1]
  have hbase := interactionLogSlope_double_peak_le β hβ
  by_cases hsmall : β ≤ 2
  · exact ((interactionLogSlope_antitone_small_temperature β hβ.le hsmall)
      ⟨by linarith [ht.1], by linarith [ht.2.1]⟩ ⟨hx0, hx.2⟩ hx.1).trans hbase
  have hβ2 : 2 ≤ β := (lt_of_not_ge hsmall).le
  by_cases hfront : x < Real.pi / 2
  · exact ((interactionLogSlope_antitone_front β hβ.le)
      ⟨by linarith [ht.1], double_interactionPeak_lt_pi_div_two β hβ2⟩ ⟨hx0, hfront⟩ hx.1).trans hbase
  have hc := Real.cos_nonpos_of_pi_div_two_le_of_le (le_of_not_gt hfront)
    (show x ≤ Real.pi + Real.pi / 2 by linarith [hx.2, Real.pi_pos])
  have hs := Real.sin_pos_of_pos_of_lt_pi hx0 hx.2
  have h := negative_cosine_logSlope_lt β (1 / interactionPeak β) x hβ2 hc hs
    (inv_interactionPeak_sq_le β hβ)
  simpa only [interactionLogSlope, neg_div] using h.le

/-- The original central and far logarithmic slopes have a strictly negative combined value on the half-peak force window.
Source: arXiv:2411.04990v2, §5, the force comparison for `thm:fixed_centers`; the factor two is the actual motion of its far comparison radius. -/
theorem interactionLogSlope_window_sum_neg (β : ℝ) (hβ : 0 < β) (u v : ℝ)
    (hu : u ∈ Icc (interactionPeak β / 2) (interactionPeak β))
    (hv : v ∈ Ico (2 * interactionPeak β) Real.pi) :
    interactionLogSlope β u + 2 * interactionLogSlope β v < 0 := by
  have hc := interactionLogSlope_core_bound β hβ u hu
  have hf := interactionLogSlope_far_bound β hβ v hv
  simp only [div_eq_mul_inv] at hc hf
  linarith

/-- The original force derivatives give a strictly negative cross product on the same central and far window.
Source: arXiv:2411.04990v2, §5, the derivative numerator of the original force ratio in `thm:fixed_centers`. -/
theorem g_pot_force_window_cross_neg (β : ℝ) (hβ : 0 < β) (u v : ℝ)
    (hu : u ∈ Icc (interactionPeak β / 2) (interactionPeak β))
    (hv : v ∈ Ico (2 * interactionPeak β) Real.pi) :
    g_pot β u * h_pot β v + 2 * h_pot β u * g_pot β v < 0 := by
  have ht := interactionPeak_properties β hβ
  have hsu := Real.sin_pos_of_pos_of_lt_pi
    (show 0 < u by linarith [hu.1, ht.1]) (show u < Real.pi by linarith [hu.2, ht.2.1, Real.pi_pos])
  have hsv := Real.sin_pos_of_pos_of_lt_pi (show 0 < v by linarith [hv.1, ht.1]) hv.2
  have hpu : 0 < h_pot β u := mul_pos (Real.exp_pos _) hsu
  have hpv : 0 < h_pot β v := mul_pos (Real.exp_pos _) hsv
  have h := mul_neg_of_pos_of_neg (mul_pos hpu hpv)
    (interactionLogSlope_window_sum_neg β hβ u v hu hv)
  rw [g_pot_eq_h_pot_mul_logSlope β u hsu.ne', g_pot_eq_h_pot_mul_logSlope β v hsv.ne']
  nlinarith only [h]

/-- Unit temperature and its actual half-peak and double-peak angles jointly satisfy the central and far force-window hypotheses.
Source: arXiv:2411.04990v2, Appendix B and §5, the original interaction geometry used in `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧
    interactionPeak 1 / 2 ∈ Icc (interactionPeak 1 / 2) (interactionPeak 1) ∧
    2 * interactionPeak 1 ∈ Ico (2 * interactionPeak 1) Real.pi := by
  have ht := interactionPeak_properties 1 one_pos
  exact ⟨one_pos, ⟨le_rfl, by linarith [ht.1]⟩, ⟨le_rfl, by linarith [ht.2.1]⟩⟩

/-- Temperature two jointly realizes the large-temperature peak bound and a far angle on the back half of the original semicircle.
Source: arXiv:2411.04990v2, Appendix B and §5, the original large-temperature force comparison. -/
example : (2 : ℝ) ≤ 2 ∧ (0 : ℝ) < 2 ∧
    Real.pi / 2 ∈ Ico (2 * interactionPeak 2) Real.pi := by
  exact ⟨le_rfl, by norm_num, ⟨(double_interactionPeak_lt_pi_div_two 2 le_rfl).le, by linarith [Real.pi_pos]⟩⟩

end Transformer.Causal
