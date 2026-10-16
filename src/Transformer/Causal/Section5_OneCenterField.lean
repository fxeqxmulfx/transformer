/-
# Exact limiting rows for one original frozen center

ArXiv:2411.04990v2, §5 and Appendix A, the single-center
case of `thm:fixed_centers`.
Separation of frozen centers is vacuous with only one center,
so the two-center parameter-window reduction does not apply.
Instead, once the actual predecessors have converged to that
center on the circle, each of their actual interactions is
the same circular interaction with the center. Their real
lifts are retained and need not equal the center's real lift.

The actual limiting force and curvature are exactly the
single-center interaction multiplied by the positive incoming
mass. Every actual limiting force zero is simple. A critical
angle with nonnegative curvature is exactly at the frozen
center on the circle and has strictly positive curvature.
These identities use the original actual force, masses, and
circular distance, without any modified field or extra
separation premise. They hold at every real temperature.
-/

import Transformer.Causal.Section5_RowDerivative
import Transformer.Causal.Section5_CriticalSets

open Real Set Filter

namespace Transformer.Causal

variable {n : ℕ}

/-- Circular coincidence preserves both actual sine and cosine interactions with every original last angle, even when real lifts differ.
Source: arXiv:2411.04990v2, §5, the single-center limiting interaction in `thm:fixed_centers`. -/
theorem frozen_trig_sub_of_circDist_zero (θ p : ℝ) (hp : circDist (p - θ) = 0) (u : ℝ) :
    Real.cos (u - p) = Real.cos (u - θ) ∧ Real.sin (u - p) = Real.sin (u - θ) := by
  have hc : Real.cos (p - θ) = 1 := by
    have h := cos_circDist (p - θ)
    rw [hp, Real.cos_zero] at h
    exact h.symm
  have hs : Real.sin (p - θ) = 0 := by
    have h := abs_sin_eq_sin_circDist (p - θ)
    rw [hp, Real.sin_zero] at h
    exact abs_eq_zero.mp h
  have he : u - p = (u - θ) - (p - θ) := by ring
  constructor
  · rw [he, Real.cos_sub, hc, hs]
    ring
  · rw [he, Real.sin_sub, hc, hs]
    ring

/-- Every actual predecessor interaction equals the original single-center force and curvature when that predecessor is circularly at the original frozen center.
Source: arXiv:2411.04990v2, §5, the original single-center force and curvature terms of `thm:fixed_centers`. -/
theorem frozen_interactions_sub_of_circDist_zero (β θ p : ℝ) (hp : circDist (p - θ) = 0) (u : ℝ) :
    h_pot β (u - p) = h_pot β (u - θ) ∧ g_pot β (u - p) = g_pot β (u - θ) := by
  obtain ⟨hc, hs⟩ := frozen_trig_sub_of_circDist_zero θ p hp u
  simp only [h_pot, g_pot, hc, hs, and_self]

/-- The exact original limiting last force for one frozen center is its actual total incoming mass times the original single-center interaction.
Source: arXiv:2411.04990v2, §5, the actual single-center specialization of the force in `thm:fixed_centers`. -/
theorem frozenOneCenter_force (β θ a : ℝ) (p : Idx n → ℝ) (hp : ∀ j, circDist (p j - θ) = 0) (u : ℝ) :
    frozenLastForce β (fun _ : Idx 1 => θ) (fun _ => a) p u = ((n : ℝ) + a) * h_pot β (u - θ) := by
  unfold frozenLastForce
  simp_rw [(frozen_interactions_sub_of_circDist_zero β θ _ (hp _) u).1]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Idx]
  ring

/-- The exact original limiting last curvature for one frozen center is its actual total incoming mass times the original single-center curvature.
Source: arXiv:2411.04990v2, §5, the actual single-center diagonal curvature in `thm:fixed_centers`. -/
theorem frozenOneCenter_curvature (β θ a : ℝ) (p : Idx n → ℝ) (hp : ∀ j, circDist (p j - θ) = 0) (u : ℝ) :
    frozenLastCurvature β (fun _ : Idx 1 => θ) (fun _ => a) p u = ((n : ℝ) + a) * g_pot β (u - θ) := by
  unfold frozenLastCurvature
  simp_rw [(frozen_interactions_sub_of_circDist_zero β θ _ (hp _) u).2]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Idx]
  ring

/-- Every zero of the original single interaction has nonzero actual curvature, at every real temperature.
Source: arXiv:2411.04990v2, §5 and Appendix B, the original one-center critical-angle nondegeneracy used for `thm:fixed_centers`. -/
theorem frozen_h_pot_zero_curvature_ne (β u : ℝ) (hu : h_pot β u = 0) : g_pot β u ≠ 0 := by
  have hs : Real.sin u = 0 := (mul_eq_zero.mp hu).resolve_left (Real.exp_ne_zero _)
  have hc2 := Real.sin_sq_add_cos_sq u
  rw [hs, zero_pow (by decide : (2 : ℕ) ≠ 0), zero_add] at hc2
  have hc : Real.cos u ≠ 0 := by intro hz; rw [hz] at hc2; norm_num at hc2
  simpa only [g_pot, hs, zero_pow (by decide : (2 : ℕ) ≠ 0), mul_zero, sub_zero] using
    mul_ne_zero (Real.exp_ne_zero _) hc

/-- A nonrepelling zero of the original single interaction lies exactly at the original frozen center on the circle.
Source: arXiv:2411.04990v2, §5 and Appendix B, the attracting single-center critical angle in `thm:fixed_centers`. -/
theorem frozen_h_pot_zero_circDist_zero (β u : ℝ) (hu : h_pot β u = 0) (hg : 0 ≤ g_pot β u) : circDist u = 0 := by
  have hs : Real.sin u = 0 := (mul_eq_zero.mp hu).resolve_left (Real.exp_ne_zero _)
  have hc2 := Real.sin_sq_add_cos_sq u
  rw [hs, zero_pow (by decide : (2 : ℕ) ≠ 0), zero_add] at hc2
  have hc : Real.cos u = 1 := by
    rcases sq_eq_one_iff.mp hc2 with hpos | hneg
    · exact hpos
    · have h := Real.exp_pos (β * (Real.cos u - 1))
      simp only [g_pot, hs, zero_pow (by decide : (2 : ℕ) ≠ 0), mul_zero, sub_zero] at hg
      rw [hneg, mul_neg_one] at hg
      rw [hneg] at h
      linarith
  apply Real.strictAntiOn_cos.injOn ⟨circDist_nonneg u, circDist_le_pi u⟩ ⟨le_rfl, Real.pi_pos.le⟩
  rw [cos_circDist, hc, Real.cos_zero]

/-- The original center is an actual simple attracting zero of the exact original limiting row with a positive original center mass.
Source: arXiv:2411.04990v2, §5, the actual single-center limiting equilibrium in `thm:fixed_centers`. -/
theorem frozenOneCenter_center_critical (β θ a : ℝ) (ha : 0 < a)
    (p : Idx n → ℝ) (hp : ∀ j, circDist (p j - θ) = 0) :
    frozenLastForce β (fun _ : Idx 1 => θ) (fun _ => a) p θ = 0 ∧
      0 < frozenLastCurvature β (fun _ : Idx 1 => θ) (fun _ => a) p θ := by
  rw [frozenOneCenter_force β θ a p hp, frozenOneCenter_curvature β θ a p hp]
  simp only [sub_self, h_pot, g_pot, Real.sin_zero, Real.cos_zero, sub_self, mul_zero, Real.exp_zero,
    mul_one, zero_pow (by decide : (2 : ℕ) ≠ 0), sub_zero, true_and]
  linarith [Nat.cast_nonneg (α := ℝ) n]

/-- All actual limiting last-angle critical points are simple for one original frozen center of positive mass once every original predecessor is circularly at that center.
Source: arXiv:2411.04990v2, §5 and Appendix A, the single-center critical-point assumption needed for the actual scalar convergence step in `thm:fixed_centers`. -/
theorem frozenOneCenter_critical_curvature_ne (β θ a : ℝ) (ha : 0 < a)
    (p : Idx n → ℝ) (hp : ∀ j, circDist (p j - θ) = 0) (u : ℝ)
    (hF : frozenLastForce β (fun _ : Idx 1 => θ) (fun _ => a) p u = 0) :
    frozenLastCurvature β (fun _ : Idx 1 => θ) (fun _ => a) p u ≠ 0 := by
  have hW : 0 < (n : ℝ) + a := by linarith [Nat.cast_nonneg (α := ℝ) n]
  rw [frozenOneCenter_force β θ a p hp] at hF
  have hh := (mul_eq_zero.mp hF).resolve_left hW.ne'
  rw [frozenOneCenter_curvature β θ a p hp]
  exact mul_ne_zero hW.ne' (frozen_h_pot_zero_curvature_ne β (u - θ) hh)

/-- Every actual nonrepelling critical angle over a one-center consensus prefix lies exactly at the original center on the circle and has strictly positive original curvature.
Source: arXiv:2411.04990v2, §5 and Appendix A, the attracting alternative for the actual single-center row of `thm:fixed_centers`. -/
theorem frozenOneCenter_nonrepelling_critical (β θ a : ℝ) (ha : 0 < a)
    (p : Idx n → ℝ) (hp : ∀ j, circDist (p j - θ) = 0) (u : ℝ)
    (hF : frozenLastForce β (fun _ : Idx 1 => θ) (fun _ => a) p u = 0)
    (hG : 0 ≤ frozenLastCurvature β (fun _ : Idx 1 => θ) (fun _ => a) p u) :
    0 < frozenLastCurvature β (fun _ : Idx 1 => θ) (fun _ => a) p u ∧ circDist (u - θ) = 0 := by
  have hW : 0 < (n : ℝ) + a := by linarith [Nat.cast_nonneg (α := ℝ) n]
  have hh : h_pot β (u - θ) = 0 := by
    rw [frozenOneCenter_force β θ a p hp] at hF
    exact (mul_eq_zero.mp hF).resolve_left hW.ne'
  have hg : 0 ≤ g_pot β (u - θ) := by
    have h := hG
    rw [frozenOneCenter_curvature β θ a p hp] at h
    exact nonneg_of_mul_nonneg_right h hW
  exact ⟨lt_of_le_of_ne hG (frozenOneCenter_critical_curvature_ne β θ a ha p hp u hF).symm,
    frozen_h_pot_zero_circDist_zero β (u - θ) hh hg⟩

/-- A genuine original frozen center and actual preceding token jointly realize positive mass, circular coincidence, criticality, and strictly positive original curvature.
Source: arXiv:2411.04990v2, §5, the actual one-center equilibrium used for `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (∀ j : Idx 1, circDist ((fun _ : Idx 1 => (0 : ℝ)) j - 0) = 0) ∧
    frozenLastForce 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) 0 = 0 ∧
    0 < frozenLastCurvature 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) 0 := by
  have hp : ∀ j : Idx 1, circDist ((fun _ : Idx 1 => (0 : ℝ)) j - 0) = 0 := fun _ => by rw [sub_self, circDist_zero]
  exact ⟨zero_lt_one, hp, frozenOneCenter_center_critical 1 0 1 zero_lt_one _ hp⟩

/-- The original zero angle jointly realizes the interaction-zero, nonrepelling-curvature, and nonzero-curvature premises without any vacuous critical-point hypothesis.
Source: arXiv:2411.04990v2, §5 and Appendix B, the original single-center attracting interaction in `thm:fixed_centers`. -/
example : h_pot 1 0 = 0 ∧ 0 ≤ g_pot 1 0 ∧ circDist 0 = 0 ∧ g_pot 1 0 ≠ 0 := by
  have hh : h_pot 1 0 = 0 := by norm_num [h_pot]
  have hg : 0 ≤ g_pot 1 0 := by norm_num [g_pot]
  exact ⟨hh, hg, frozen_h_pot_zero_circDist_zero 1 0 hh hg, frozen_h_pot_zero_curvature_ne 1 0 hh⟩

end Transformer.Causal
