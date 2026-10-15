/-
# Original narrow peak windows and curvature-parameter preservation

ArXiv:2411.04990v2, Appendix B and §5, the force lower bound
used in `thm:fixed_centers`. The original interaction can decrease
after its peak, so controlling a window extending beyond the peak
requires an actual comparison of its two endpoints.

We prove `h(3 tau/2)>=h(tau/2)` at the actual original peak for
every positive beta. With `c=cos(tau)` and `b=cos(tau/2)`, the
original root equation gives the exact interaction ratio
`(1+2c)*exp(-c/b)`. The logarithm bound
`2x/(x+2)<=log(1+x)` and `1+c=2b²<=2b` show this ratio is
at least one. Original unimodality then gives `h(x)>=h(eta)`
throughout `[eta,tau+eta]` when `0<=eta<=tau/2`.
The narrow-radius premise is explicit, and no original larger
radius is silently substituted into the source's force estimate.
The original signed curvature parameter inequality is also
preserved when the neighborhood radius is decreased.
-/

import Transformer.Causal.Section5_CircularInteraction
import Transformer.Causal.Section5_ParameterGeometry
import Mathlib.Analysis.SpecialFunctions.Log.Basic

open Real Set

namespace Transformer.Causal

/-- The actual half-angle relation controls the original exponential ratio through a proved logarithm bound.
Source: arXiv:2411.04990v2, Appendix B and §5, auxiliary calculation for the force comparison in `thm:fixed_centers`. -/
theorem exp_peak_half_factor (c b : ℝ) (hc : 0 < c) (hc1 : c ≤ 1) (hb : 0 < b)
    (hb2 : 2 * b ^ 2 = 1 + c) : Real.exp (c / b) ≤ 1 + 2 * c := by
  have hb1 : b ≤ 1 := by nlinarith
  have hm := mul_le_mul_of_nonneg_left hb1 hb.le
  have hcb : 1 + c ≤ 2 * b := by nlinarith
  have hquot : c / b ≤ 2 * (2 * c) / (2 * c + 2) := by
    rw [div_le_div_iff₀ hb (by positivity)]
    nlinarith [mul_nonneg hc.le (show 0 ≤ 2 * b - 1 - c by linarith)]
  exact (Real.le_log_iff_exp_le (by positivity)).mp
    (hquot.trans (Real.le_log_one_add_of_nonneg (show 0 ≤ 2 * c by positivity)))

/-- The actual original interaction at three half-peak distances is at least its value at one half-peak distance.
Source: arXiv:2411.04990v2, Appendix B and §5, the force endpoint comparison used to justify `thm:fixed_centers` on narrow neighborhoods. -/
theorem h_pot_half_peak_le_three_half_peak (β : ℝ) (hβ : 0 < β) :
    h_pot β (interactionPeak β / 2) ≤ h_pot β (3 * interactionPeak β / 2) := by
  let τ := interactionPeak β
  let a := τ / 2
  let C := interactionPeakCos β
  let B := Real.cos a
  have ht := interactionPeak_properties β hβ
  have hc := interactionPeakCos_properties β hβ
  have h2a : 2 * a = τ := by dsimp [a]; ring
  have hB : 0 < B := Real.cos_pos_of_mem_Ioo ⟨by dsimp [a, τ]; linarith [Real.pi_pos],
    by dsimp [a, τ]; linarith [Real.pi_pos]⟩
  have hS : 0 < Real.sin a := Real.sin_pos_of_pos_of_lt_pi
    (by dsimp [a, τ]; linarith) (by dsimp [a, τ]; linarith [Real.pi_pos])
  have hcosB : C = 2 * B ^ 2 - 1 := by
    have h := Real.cos_two_mul a
    rw [h2a, ht.2.2] at h
    exact h
  have hcosS : C = 1 - 2 * Real.sin a ^ 2 := by
    have h := Real.cos_two_mul_eq_one_sub a
    rw [h2a, ht.2.2] at h
    exact h
  have hsin2 : Real.sin τ = 2 * Real.sin a * B := by
    have h := Real.sin_two_mul a
    rw [h2a] at h
    exact h
  have hroot : β * C ^ 2 + C - β = 0 := hc.2.2.1
  have hrel : β * Real.sin τ ^ 2 = C := by
    have hunit := Real.sin_sq_add_cos_sq τ
    rw [ht.2.2] at hunit
    linear_combination β * hunit - hroot
  have hrel' : 4 * β * Real.sin a ^ 2 * B ^ 2 = C := by rw [hsin2] at hrel; nlinarith only [hrel]
  have hsin3 : Real.sin (3 * a) = (1 + 2 * C) * Real.sin a := by
    rw [Real.sin_three_mul]
    linear_combination (-2 * Real.sin a) * hcosS
  have hcos3 : Real.cos (3 * a) - B = -4 * B * Real.sin a ^ 2 := by
    rw [Real.cos_three_mul]
    change 4 * B ^ 3 - 3 * B - B = -4 * B * Real.sin a ^ 2
    linear_combination (4 * B) * Real.sin_sq_add_cos_sq a
  have hdiff : β * (Real.cos (3 * a) - B) = -C / B := by
    apply (eq_div_iff hB.ne').mpr
    rw [hcos3]
    nlinarith only [hrel']
  have hscore : β * (Real.cos (3 * a) - 1) = β * (Real.cos a - 1) + (-C / B) := by
    change β * (Real.cos (3 * a) - 1) = β * (B - 1) + (-C / B)
    nlinarith only [hdiff]
  have hExp := exp_peak_half_factor C B hc.1 hc.2.1.le hB (by linarith [hcosB])
  have hfactor : 1 ≤ (1 + 2 * C) * Real.exp (-C / B) := by
    have hm := mul_le_mul_of_nonneg_right hExp (Real.exp_pos (-C / B)).le
    have hcancel : Real.exp (C / B) * Real.exp (-C / B) = 1 := by
      rw [neg_div, ← Real.exp_add, add_neg_cancel, Real.exp_zero]
    rw [hcancel] at hm
    exact hm
  have hbound : h_pot β a ≤ h_pot β (3 * a) := by
    calc
      _ = (Real.exp (β * (Real.cos a - 1)) * Real.sin a) * 1 := by rw [h_pot, mul_one]
      _ ≤ (Real.exp (β * (Real.cos a - 1)) * Real.sin a) * ((1 + 2 * C) * Real.exp (-C / B)) :=
        mul_le_mul_of_nonneg_left hfactor (mul_pos (Real.exp_pos _) hS).le
      _ = _ := by rw [h_pot, hscore, Real.exp_add, hsin3]; ring
  simpa only [a, τ, mul_div_assoc] using hbound

/-- Shifting from an original narrow radius by the actual original peak cannot lower the actual original force below its value at that radius.
Source: arXiv:2411.04990v2, §5, the endpoint lower bound needed for the force estimate in `thm:fixed_centers`. -/
theorem h_pot_peak_shift_lower (β : ℝ) (hβ : 0 < β) (η : ℝ) (hη : 0 ≤ η)
    (hnarrow : η ≤ interactionPeak β / 2) : h_pot β η ≤ h_pot β (interactionPeak β + η) := by
  have ht := interactionPeak_properties β hβ
  have hm := h_pot_monotone_peak β hβ
  have hhead := hm.1.monotoneOn (show η ∈ Icc 0 (interactionPeak β) from ⟨hη, by linarith⟩)
    (show interactionPeak β / 2 ∈ Icc 0 (interactionPeak β) from ⟨by linarith, by linarith⟩) hnarrow
  have htail := hm.2.antitoneOn
    (show interactionPeak β + η ∈ Icc (interactionPeak β) Real.pi from ⟨by linarith, by linarith [Real.pi_pos]⟩)
    (show 3 * interactionPeak β / 2 ∈ Icc (interactionPeak β) Real.pi from ⟨by linarith, by linarith [Real.pi_pos]⟩)
    (show interactionPeak β + η ≤ 3 * interactionPeak β / 2 by linarith)
  exact hhead.trans ((h_pot_half_peak_le_three_half_peak β hβ).trans htail)

/-- The actual original force stays above its original narrow-edge value throughout the whole peak-plus-neighborhood comparison interval.
Source: arXiv:2411.04990v2, §5, the force lower bound in `thm:fixed_centers`, with the sufficient narrow-radius condition explicit. -/
theorem h_pot_narrow_window_lower (β : ℝ) (hβ : 0 < β) (η : ℝ) (hη : 0 ≤ η)
    (hnarrow : η ≤ interactionPeak β / 2) (x : ℝ) (hx : x ∈ Icc η (interactionPeak β + η)) :
    h_pot β η ≤ h_pot β x := by
  have ht := interactionPeak_properties β hβ
  have hm := h_pot_monotone_peak β hβ
  by_cases hcore : x ≤ interactionPeak β
  · exact hm.1.monotoneOn ⟨hη, by linarith⟩ ⟨hη.trans hx.1, hcore⟩ hx.1
  · have htail := hm.2.antitoneOn ⟨(le_of_not_ge hcore), by linarith [hx.2, Real.pi_pos]⟩
      ⟨by linarith, by linarith [Real.pi_pos]⟩ hx.2
    exact (h_pot_peak_shift_lower β hβ η hη hnarrow).trans htail

/-- Two actual original tokens in a half-peak neighborhood have nonnegative actual original pair curvature.
Source: arXiv:2411.04990v2, §5, the same-cluster sign step in `thm:fixed_centers`, with its sufficient radius bound explicit. -/
theorem g_pot_nonneg_in_narrow_cluster (β : ℝ) (hβ : 0 < β) (η θ x y : ℝ)
    (hnarrow : η ≤ interactionPeak β / 2) (hx : circDist (x - θ) ≤ η) (hy : circDist (y - θ) ≤ η) :
    0 ≤ g_pot β (x - y) := by
  have ht := circDist_triangle x θ y
  rw [show θ - y = -(y - θ) by ring, circDist_neg] at ht
  apply (g_pot_nonneg_iff_circDist_le_peak β hβ _).mpr
  linarith

/-- Decreasing the actual original neighborhood radius preserves the original signed curvature parameter inequality.
Source: arXiv:2411.04990v2, §5, the original signed `g` condition in `thm:fixed_centers`. -/
theorem frozen_parameter_curvature_window_mono (N β ε η c : ℝ) (hN : 0 < N) (hβ : 0 < β)
    (hε : 0 < ε) (hη : 0 ≤ η) (hηε : η ≤ ε) (hc : 2 + 2 * ε < c)
    (hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi)
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ)))) :
    -(N * g_pot β ((c - 2 * η) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (η * β ^ (-(1 / 2 : ℝ))) := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have he := frozen_parameter_core_lt_peak N β ε c hN hβ hε hc hcpi hg
  have hr := frozen_parameter_radii β ε c hβ hε hc
  have hcore := g_pot_core_lower β hβ _ _ (mul_nonneg hη hs.le)
    (mul_le_mul_of_nonneg_right hηε hs.le) he.le
  have hfar := g_pot_far_lower β hβ _ _ hr.2.2.2.le
    (show (c - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ≤ (c - 2 * η) * β ^ (-(1 / 2 : ℝ)) by
      nlinarith [mul_nonneg (sub_nonneg.mpr hηε) hs.le])
    (show (c - 2 * η) * β ^ (-(1 / 2 : ℝ)) ≤ Real.pi by nlinarith [mul_nonneg hη hs.le])
  have hm := mul_le_mul_of_nonneg_left hfar hN.le
  linarith

/-- The actual unit-temperature peak has genuine narrow points on both sides of the peak inside the proved original force window.
Source: arXiv:2411.04990v2, §5, the original peak-plus-neighborhood interval used in `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ interactionPeak 1 / 4 ∧
    interactionPeak 1 / 4 ≤ interactionPeak 1 / 2 ∧
    interactionPeak 1 / 2 ∈ Icc (interactionPeak 1 / 4) (interactionPeak 1 + interactionPeak 1 / 4) ∧
    5 * interactionPeak 1 / 4 ∈ Icc (interactionPeak 1 / 4) (interactionPeak 1 + interactionPeak 1 / 4) := by
  have ht := (interactionPeak_properties 1 one_pos).1
  exact ⟨one_pos, by linarith, by linarith, ⟨by linarith, by linarith⟩, ⟨by linarith, by linarith⟩⟩

/-- The positive auxiliary half-angle premises hold at their actual limiting endpoint.
Source: arXiv:2411.04990v2, Appendix B, the original peak half-angle calculation used in §5. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) ^ 2 = 1 + 1 := by norm_num

/-- Distinct original tokens on opposite sides of a genuine narrow original neighborhood jointly satisfy the pair-curvature premises.
Source: arXiv:2411.04990v2, §5, the same-cluster pair estimate used in `thm:fixed_centers`. -/
example : interactionPeak 1 / 4 ≤ interactionPeak 1 / 2 ∧
    circDist (interactionPeak 1 / 4 - 0) ≤ interactionPeak 1 / 4 ∧
    circDist (-interactionPeak 1 / 4 - 0) ≤ interactionPeak 1 / 4 := by
  have ht := interactionPeak_properties 1 one_pos
  have hdist : circDist (interactionPeak 1 / 4) = interactionPeak 1 / 4 :=
    circDist_eq_self _ ⟨by linarith, by linarith [Real.pi_pos]⟩
  rw [sub_zero, sub_zero, neg_div, circDist_neg, hdist]
  exact ⟨by linarith, le_rfl, le_rfl⟩

/-- The original nontrivial numerical parameter window jointly realizes curvature-window preservation at a strictly smaller positive radius.
Source: arXiv:2411.04990v2, §5, the original parameter conditions in `thm:fixed_centers`. -/
example : (0 : ℝ) < 3 ∧ (0 : ℝ) < 11 ∧ (0 : ℝ) < 9 / 100 ∧ (0 : ℝ) ≤ 9 / 200 ∧
    (9 / 200 : ℝ) ≤ 9 / 100 ∧ 2 + 2 * (9 / 100 : ℝ) < 142 / 25 ∧
    (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) < Real.pi ∧
    -(3 * g_pot 11 ((142 / 25 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ)))) <
      g_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    -(3 * g_pot 11 ((142 / 25 - 2 * (9 / 200)) * (11 : ℝ) ^ (-(1 / 2 : ℝ)))) <
      g_pot 11 ((9 / 200) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  exact ⟨hN, hβ, hε, by norm_num, by norm_num, hc, hcpi, hg,
    frozen_parameter_curvature_window_mono 3 11 (9 / 100) (9 / 200) (142 / 25)
      hN hβ hε (by norm_num) (by norm_num) hc hcpi hg⟩

end Transformer.Causal
