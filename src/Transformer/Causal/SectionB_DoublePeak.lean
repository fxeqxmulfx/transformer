/-
# The original force beyond twice its actual peak

ArXiv:2411.04990v2, Appendix B and §5, the original force
parameter window in `thm:fixed_centers`.
The exact original peak equation gives
`h(2 tau)/h(tau)=2c exp(-c(2c+1)/(c+1))`, with `c=cos(tau)`.
For original beta at least one, `c²+c>=1`. The displayed ratio
decreases between this actual c and one, where its value is
`2 exp(-3/2)>1/3`.

The original nontrivial force window has total mass at least
three and already forces beta above one. Its actual far-force
comparison point must therefore exceed twice the actual peak:
otherwise original unimodality and the ratio bound would make
the total far force exceed the original peak height, contradicting
the original strict force inequality. No original parameter
condition or force function is replaced.
-/

import Transformer.Causal.SectionB_PeakHeight
import Transformer.Causal.Section5_ParameterTemperature

open Real Set

namespace Transformer.Causal

/-- The exact original ratio of double-peak force to peak force, expressed in the original peak cosine.
Source: arXiv:2411.04990v2, Appendix B, the original peak equation used in §5, `thm:fixed_centers`. -/
noncomputable def doublePeakRatio (c : ℝ) : ℝ := 2 * c * Real.exp (-(c * (2 * c + 1) / (c + 1)))

/-- The actual original double-peak force ratio has its explicit signed scalar derivative.
Source: arXiv:2411.04990v2, Appendix B and §5, the original peak-force comparison for `thm:fixed_centers`. -/
theorem hasDerivAt_doublePeakRatio (x : ℝ) (hx : x + 1 ≠ 0) :
    HasDerivAt doublePeakRatio (2 * Real.exp (-(x * (2 * x + 1) / (x + 1))) *
      (2 * x + 1) * (1 - x - x ^ 2) / (x + 1) ^ 2) x := by
  have hn := (hasDerivAt_id x).fun_mul (((hasDerivAt_id x).const_mul 2).add_const 1)
  have hd := (hasDerivAt_id x).add_const 1
  have he := (hn.fun_div hd hx).fun_neg.exp
  apply (((hasDerivAt_id x).const_mul 2).fun_mul he).congr_deriv
  simp only [id_eq, one_mul]
  field_simp
  ring

/-- The actual original double-peak ratio decreases from any admissible original peak cosine to its limiting value at one.
Source: arXiv:2411.04990v2, Appendix B and §5, the beta-at-least-one original peak-force comparison. -/
theorem doublePeakRatio_antitone (c : ℝ) (hc : 0 ≤ c) (hroot : 1 ≤ c ^ 2 + c) :
    AntitoneOn doublePeakRatio (Icc c 1) := by
  have hd : ∀ x ∈ Icc c 1, HasDerivAt doublePeakRatio
      (2 * Real.exp (-(x * (2 * x + 1) / (x + 1))) * (2 * x + 1) * (1 - x - x ^ 2) / (x + 1) ^ 2) x := by
    intro x hx
    exact hasDerivAt_doublePeakRatio x (by linarith [hx.1])
  apply antitoneOn_of_deriv_nonpos (convex_Icc _ _)
    (fun x hx => (hd x hx).continuousAt.continuousWithinAt)
  · intro x hx
    rw [interior_Icc] at hx
    exact (hd x ⟨hx.1.le, hx.2.le⟩).differentiableAt.differentiableWithinAt
  · intro x hx
    rw [interior_Icc] at hx
    rw [(hd x ⟨hx.1.le, hx.2.le⟩).deriv]
    have hx0 : 0 ≤ x := hc.trans hx.1.le
    have hsq := mul_self_le_mul_self hc hx.1.le
    have hn : 1 - x - x ^ 2 ≤ 0 := by nlinarith
    exact div_nonpos_of_nonpos_of_nonneg
      (mul_nonpos_of_nonneg_of_nonpos (by positivity) hn) (sq_nonneg _)

/-- The actual original interaction at twice its actual original peak has the exact ratio given by its original peak cosine.
Source: arXiv:2411.04990v2, Appendix B, the original `h` and original peak equation used in §5, `thm:fixed_centers`. -/
theorem h_pot_double_peak_eq_ratio (β : ℝ) (hβ : 0 < β) :
    h_pot β (2 * interactionPeak β) = h_pot β (interactionPeak β) * doublePeakRatio (interactionPeakCos β) := by
  let τ := interactionPeak β
  let c := interactionPeakCos β
  have ht := interactionPeak_properties β hβ
  have hc := interactionPeakCos_properties β hβ
  have hroot : β * c ^ 2 + c - β = 0 := hc.2.2.1
  have hbc : β * (c - 1) * (c + 1) = -c := by nlinarith
  have hd : β * (2 * c ^ 2 - 1 - c) = -(c * (2 * c + 1) / (c + 1)) := by
    rw [← neg_div]
    apply (eq_div_iff (show c + 1 ≠ 0 by linarith [hc.1])).mpr
    linear_combination (2 * c + 1) * hbc
  have hscore : β * (Real.cos (2 * τ) - 1) = β * (Real.cos τ - 1) - c * (2 * c + 1) / (c + 1) := by
    rw [Real.cos_two_mul, ht.2.2]
    nlinarith only [hd]
  change h_pot β (2 * τ) = h_pot β τ * doublePeakRatio c
  rw [h_pot, h_pot, hscore, Real.exp_sub, Real.sin_two_mul, ht.2.2, doublePeakRatio]
  rw [Real.exp_neg]
  ring

/-- The limiting original double-peak force ratio is strictly above one third.
Source: arXiv:2411.04990v2, Appendix B and §5, the original peak-force comparison in `thm:fixed_centers`. -/
theorem doublePeakRatio_one_gt_third : (1 / 3 : ℝ) < doublePeakRatio 1 := by
  have hh := Real.exp_bound_div_one_sub_of_interval' (x := 1 / 2) (by norm_num) (by norm_num)
  norm_num at hh
  have hm := mul_lt_mul_of_pos_right Real.exp_one_lt_three (Real.exp_pos (1 / 2 : ℝ))
  have he : Real.exp (3 / 2 : ℝ) < 6 := by
    rw [show (3 / 2 : ℝ) = 1 + 1 / 2 by ring, Real.exp_add]
    linarith
  have hinv : (1 / 6 : ℝ) < Real.exp (-(3 / 2 : ℝ)) := by
    rw [Real.exp_neg, inv_eq_one_div, lt_div_iff₀ (Real.exp_pos (3 / 2))]
    linarith
  norm_num [doublePeakRatio]
  linarith

/-- For original temperatures at least one, the actual original force at twice the original peak exceeds one third of its actual original maximum.
Source: arXiv:2411.04990v2, Appendix B and §5, the original force-window separation in `thm:fixed_centers`. -/
theorem h_pot_double_peak_gt_third_height (β : ℝ) (hβ1 : 1 ≤ β) :
    h_pot β (interactionPeak β) / 3 < h_pot β (2 * interactionPeak β) := by
  have hβ : 0 < β := by linarith
  let c := interactionPeakCos β
  have hc := interactionPeakCos_properties β hβ
  have hroot : 1 ≤ c ^ 2 + c := by
    have hnonneg : 0 ≤ 1 - c ^ 2 := by nlinarith [hc.1, hc.2.1]
    have hm := mul_le_mul_of_nonneg_right hβ1 hnonneg
    have hr : β * c ^ 2 + c - β = 0 := hc.2.2.1
    nlinarith
  have hratio := doublePeakRatio_antitone c hc.1.le hroot
    (show c ∈ Icc c 1 from ⟨le_rfl, hc.2.1.le⟩) (show (1 : ℝ) ∈ Icc c 1 from ⟨hc.2.1.le, le_rfl⟩) hc.2.1.le
  have ht := interactionPeak_properties β hβ
  have hp : 0 < h_pot β (interactionPeak β) := mul_pos (Real.exp_pos _)
    (Real.sin_pos_of_pos_of_lt_pi ht.1 (by linarith [Real.pi_pos]))
  have hm := mul_lt_mul_of_pos_right (doublePeakRatio_one_gt_third.trans_le hratio) hp
  rw [h_pot_double_peak_eq_ratio β hβ]
  change _ < h_pot β (interactionPeak β) * doublePeakRatio c
  nlinarith only [hm]

/-- The original separated force window with actual total mass at least three places its actual original far comparison point beyond twice the actual original peak.
Source: arXiv:2411.04990v2, §5, a derived consequence of the original force hypothesis of `thm:fixed_centers`. -/
theorem frozen_force_radius_gt_double_peak (N β ε c : ℝ) (hN : 3 ≤ N) (hβ : 0 < β)
    (hε : 0 < ε) (hc : 2 + 2 * ε < c) (hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ)))) :
    2 * interactionPeak β < (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) := by
  have hβ1 := (one_lt_temperature_of_frozen_parameters N β ε c hN hβ hε hc hcpi hh).le
  have ht := interactionPeak_properties β hβ
  have hr := frozen_parameter_radii β ε c hβ hε hc
  have hrpi := (frozen_parameter_radii_lt_pi β ε c hβ hε hcpi).1
  by_contra h
  have hle := le_of_not_gt h
  have hm := (h_pot_monotone_peak β hβ).2.antitoneOn ⟨hr.2.1.le, hrpi.le⟩
    ⟨by linarith, by linarith [Real.pi_pos]⟩ hle
  have hp := h_pot_nonneg β (show (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ∈ Icc 0 Real.pi from
    ⟨ht.1.le.trans hr.2.1.le, hrpi.le⟩)
  have hmass := mul_le_mul_of_nonneg_right hN hp
  have hupper := (le_abs_self (h_pot β (ε * β ^ (-(1 / 2 : ℝ))))).trans (abs_h_pot_le_peak β hβ _)
  have hthird := h_pot_double_peak_gt_third_height β hβ1
  nlinarith only [hh, hm, hmass, hupper, hthird]

/-- Actual admissible original ratio inputs realize the derivative and nondegenerate comparison-interval hypotheses jointly.
Source: arXiv:2411.04990v2, Appendix B, the original peak-force ratio used in §5. -/
example : (1 : ℝ) + 1 ≠ 0 ∧ (0 : ℝ) ≤ 3 / 4 ∧ (3 / 4 : ℝ) ≤ 1 ∧
    (1 : ℝ) ≤ (3 / 4 : ℝ) ^ 2 + 3 / 4 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 := by norm_num

/-- The actual original mass-three frozen-center window jointly realizes every force-radius premise and its strictly improved separation conclusion.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`, with Appendix B's original explicit parameter window. -/
example : ∃ N β ε c : ℝ, 3 ≤ N ∧ 0 < β ∧ 0 < ε ∧ 2 + 2 * ε < c ∧
    c * β ^ (-(1 / 2 : ℝ)) < Real.pi ∧
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
    2 * interactionPeak β < (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  exact ⟨3, 11, 9 / 100, 142 / 25, le_rfl, hβ, hε, hc, hcpi, hh,
    frozen_force_radius_gt_double_peak 3 11 (9 / 100) (142 / 25) le_rfl hβ hε hc hcpi hh⟩

end Transformer.Causal
