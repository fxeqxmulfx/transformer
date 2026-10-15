/-
# Consequences of the original frozen-center parameter hypotheses

ArXiv:2411.04990v2, §5, `thm:fixed_centers`.
For two or more actual separated centers, their circular separation
puts `c beta^(-1/2)` below pi. The original hypothesis
`c > 2+2 epsilon` then puts the force comparison beyond the actual
interaction peak, and the curvature comparison beyond twice the
original interaction scale. Both remain inside the first semicircle.

The corrected original curvature inequality consequently forces
`epsilon beta^(-1/2)` strictly below the actual peak. This is a
derived consequence of the original assumptions, rather than an
additional restriction on epsilon or beta. We also retain the
actual original mass bound and the uniqueness of a center within
the peak-plus-neighborhood radius used in the manuscript's induction.
-/

import Transformer.Causal.Section5_CircularInteraction

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {m : ℕ}

/-- The original frozen-center total mass dominates the actual number of moving and frozen tokens.
Source: arXiv:2411.04990v2, §5, `N=n+sum a_j` and `a_j>=1` in `thm:fixed_centers`. -/
theorem frozen_total_mass_lower (n : ℕ) (a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j) :
    (n : ℝ) + m ≤ (n : ℝ) + ∑ j, a j := by
  have h := Finset.sum_le_sum (s := Finset.univ) (fun j _ => ha j)
  have hs : (∑ j : Idx m, (1 : ℝ)) = m := by simp [Idx]
  rw [hs] at h
  linarith

/-- Any actual original frozen center of mass at least one makes the original total mass strictly positive.
Source: arXiv:2411.04990v2, §5, the positive total mass in `thm:fixed_centers`. -/
theorem frozen_total_mass_pos (n : ℕ) (a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j) (hm : 0 < m) :
    0 < (n : ℝ) + ∑ j, a j := by
  have h := frozen_total_mass_lower n a ha
  have hmreal : (0 : ℝ) < m := by exact_mod_cast hm
  linarith [Nat.cast_nonneg (α := ℝ) n]

/-- The original parameter inequality places both far comparison radii beyond the exact original interaction thresholds.
Source: arXiv:2411.04990v2, §5, the original `c>2+2epsilon` hypothesis of `thm:fixed_centers`. -/
theorem frozen_parameter_radii (β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε) (hc : 2 + 2 * ε < c) :
    0 < ε * β ^ (-(1 / 2 : ℝ)) ∧
      interactionPeak β < (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ∧
      (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) < (c - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ∧
      2 * β ^ (-(1 / 2 : ℝ)) < (c - 2 * ε) * β ^ (-(1 / 2 : ℝ)) := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have h1 := mul_pos (show 0 < c - 2 - 2 * ε by linarith) hs
  have ht := interactionPeak_lt_scale β hβ
  exact ⟨mul_pos hε hs, by nlinarith, by nlinarith, by nlinarith⟩

/-- One actual separated pair bounds the original separation scale strictly below the antipodal distance.
Source: arXiv:2411.04990v2, §5, actual circle separation in `thm:fixed_centers`. -/
theorem frozen_separation_scale_lt_pi (β c θ ψ : ℝ)
    (hsep : c * β ^ (-(1 / 2 : ℝ)) < circDist (θ - ψ)) :
    c * β ^ (-(1 / 2 : ℝ)) < Real.pi := hsep.trans_le (circDist_le_pi _)

/-- The original positive neighborhood radius keeps both actual far comparison points inside the first semicircle.
Source: arXiv:2411.04990v2, §5, the circular regime of the original `h` and `g` hypotheses in `thm:fixed_centers`. -/
theorem frozen_parameter_radii_lt_pi (β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi) :
    (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) < Real.pi ∧
      (c - 2 * ε) * β ^ (-(1 / 2 : ℝ)) < Real.pi := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have he := mul_pos hε hs
  constructor <;> nlinarith

/-- The original signed curvature inequality forces the original neighborhood strictly below the actual interaction peak.
Source: arXiv:2411.04990v2, §5, the corrected-sign `g` hypothesis of `thm:fixed_centers`; subpeak size is derived here. -/
theorem frozen_parameter_core_lt_peak (N β ε c : ℝ) (hN : 0 < N) (hβ : 0 < β)
    (hε : 0 < ε) (hc : 2 + 2 * ε < c) (hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi)
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) <
      g_pot β (ε * β ^ (-(1 / 2 : ℝ)))) : ε * β ^ (-(1 / 2 : ℝ)) < interactionPeak β := by
  have hr := frozen_parameter_radii β ε c hβ hε hc
  have hpi := frozen_parameter_radii_lt_pi β ε c hβ hε hcpi
  have hneg := g_pot_far_neg β hβ _ hpi.2.le hr.2.2.2.le
  have hp : 0 < g_pot β (ε * β ^ (-(1 / 2 : ℝ))) := by
    have hn := mul_neg_of_pos_of_neg hN hneg
    linarith
  have hemem : ε * β ^ (-(1 / 2 : ℝ)) ∈ Icc 0 Real.pi := by
    have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
    have h := mul_pos (show 0 < c - ε by linarith) hs
    exact ⟨hr.1.le, by nlinarith⟩
  exact (g_pot_pos_iff_lt_peak β hβ _ hemem).mp hp

/-- Actual original center separation exceeds twice the actual peak-plus-neighborhood radius used to localize the next token.
Source: arXiv:2411.04990v2, §5, the unique-cluster geometry in `thm:fixed_centers`. -/
theorem frozen_parameter_peak_separation (β ε c θ ψ : ℝ) (hβ : 0 < β)
    (hc : 2 + 2 * ε < c) (hsep : c * β ^ (-(1 / 2 : ℝ)) < circDist (θ - ψ)) :
    2 * (interactionPeak β + ε * β ^ (-(1 / 2 : ℝ))) < circDist (θ - ψ) := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have hm := mul_pos (show 0 < c - 2 - 2 * ε by linarith) hs
  nlinarith [interactionPeak_lt_scale β hβ]

/-- A token cannot lie in two original peak-plus-neighborhood balls around the original separated frozen centers.
Source: arXiv:2411.04990v2, §5, the unambiguous center assignment in the proof of `thm:fixed_centers`. -/
theorem frozen_parameter_center_unique (β ε c : ℝ) (hβ : 0 < β) (hc : 2 + 2 * ε < c)
    (θ : Idx m → ℝ)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (u : ℝ) (i j : Idx m)
    (hi : circDist (u - θ i) ≤ interactionPeak β + ε * β ^ (-(1 / 2 : ℝ)))
    (hj : circDist (u - θ j) ≤ interactionPeak β + ε * β ^ (-(1 / 2 : ℝ))) : i = j := by
  by_contra hij
  have hs := frozen_parameter_peak_separation β ε c (θ i) (θ j) hβ hc (hsep i j hij)
  have ht := circDist_triangle (θ i) u (θ j)
  rw [show θ i - u = -(u - θ i) by ring, circDist_neg] at ht
  linarith

/-- The original displayed numerical parameters jointly satisfy the actual signed curvature inequality, positive masses, and semicircle regime.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`, with the Appendix B `lem:interaction` parameter window. -/
theorem frozen_sample_parameters : (0 : ℝ) < 3 ∧ (0 : ℝ) < 11 ∧ (0 : ℝ) < 9 / 100 ∧
    2 + 2 * (9 / 100 : ℝ) < 142 / 25 ∧ (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) < Real.pi ∧
    3 * h_pot 11 ((142 / 25 - 1 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) <
      h_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    -(3 * g_pot 11 ((142 / 25 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ)))) <
      g_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) := by
  have hex : (156 : ℝ) < Real.exp (3 * (142 / 25 - 1 - 2 * (9 / 100)) ^ 2 / 8) := by
    have h1 := Real.quadratic_le_exp_of_nonneg (show (0 : ℝ) ≤ 81 / 32 by norm_num)
    have h3 : Real.exp (3 * (142 / 25 - 1 - 2 * (9 / 100)) ^ 2 / 8) = Real.exp (81 / 32) ^ 3 := by
      rw [← Real.exp_nat_mul]
      norm_num
    rw [h3]
    calc (156 : ℝ) < (1 + 81 / 32 + (81 / 32) ^ 2 / 2) ^ 3 := by norm_num
      _ ≤ Real.exp (81 / 32) ^ 3 := by gcongr
  have hw := interaction_inequalities 3 (142 / 25) (9 / 100) 11 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by nlinarith)
  have hs : (11 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ 1 / 3 := by
    rw [Real.rpow_neg (by norm_num), ← Real.sqrt_eq_rpow, one_div]
    have hsq : (3 : ℝ) ≤ Real.sqrt 11 := by
      have h := Real.sq_sqrt (show (0 : ℝ) ≤ 11 by norm_num)
      nlinarith [Real.sqrt_nonneg 11]
    exact inv_anti₀ (by norm_num) hsq
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, ?_, hw.1, hw.2⟩
  nlinarith [Real.pi_gt_three]

/-- Actual unit frozen masses and two actual distinct centers jointly realize the original total-mass and unique-center hypotheses.
Source: arXiv:2411.04990v2, §5, the original antipodal frozen configuration in `thm:fixed_centers`. -/
example : (∀ j : Idx 2, (1 : ℝ) ≤ (fun _ : Idx 2 => (1 : ℝ)) j) ∧ (0 : ℕ) < 2 ∧
    (∀ i j : Idx 2, i ≠ j → (3 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) <
      circDist ((![0, Real.pi] : Idx 2 → ℝ) i - (![0, Real.pi] : Idx 2 → ℝ) j)) ∧
    circDist (0 - 0) ≤ interactionPeak 1 + (1 / 4 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ∧
    2 + 2 * (1 / 4 : ℝ) < 3 := by
  have ht := (interactionPeak_properties 1 one_pos).1
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  refine ⟨fun _ => le_rfl, by norm_num, ?_, ?_, by norm_num⟩
  · intro i j hij
    have hlt : (3 : ℝ) < Real.pi := Real.pi_gt_three
    fin_cases i <;> fin_cases j <;> simp_all [circDist_neg]
  · rw [sub_self, circDist_zero]
    positivity

end Transformer.Causal
