/-
# Three causal tokens with a resting pair

ArXiv:2605.09213v1, §1, `eq:gpt-dyn-alibi`, for `N = 3`. The first token
never moves. The second follows the first alone, so it rests whenever the
two are at an equilibrium distance, `sin(θ₂ - θ₁) = 0`. The third token then
follows an autonomous field of its own angle. That field vanishes at `0` and
`π`, so `GlobalFlow.exists_global_Icc` gives it a complete trajectory from
`π/2`.

Two such resting pairs, at distance `0` and at distance `π`, give the third
token different velocities at `π/2` when `β ≠ 0`. Moving the second token by
whole turns changes nothing, by the periodicity of `w_β'`.
These flows are used for the counterexample in
`Section3_AperiodicCovariance`.
-/

import Transformer.Kinetic.Defs
import Transformer.GlobalFlow.Interval

open Real

namespace Transformer
namespace Kinetic

/-- `w_β'` vanishes wherever `sin` does. -/
theorem wBetaDeriv_eq_zero_of_sin_eq_zero (β : ℝ) {z : ℝ} (hz : Real.sin z = 0) :
    wBetaDeriv β z = 0 := by
  simp [wBetaDeriv, hz]

/-- The first of three tokens has no predecessor and does not move.
Source: arXiv:2605.09213v1, §1, `eq:gpt-dyn-alibi`. -/
theorem gptField_three_zero (lam β : ℝ) (θ : Idx 3 → ℝ) : gptField lam β 3 θ 0 = 0 := by
  simp [gptField]

/-- The second of three tokens follows its single predecessor and does not
move when the two are at an equilibrium distance.
Source: arXiv:2605.09213v1, §1, `eq:gpt-dyn-alibi`. -/
theorem gptField_three_one_eq_zero (lam β : ℝ) (θ : Idx 3 → ℝ)
    (h : Real.sin (θ 1 - θ 0) = 0) : gptField lam β 3 θ 1 = 0 := by
  simp [gptField, Finset.sum_filter, wBetaDeriv_eq_zero_of_sin_eq_zero β h]

/-- The third of three tokens averages its two predecessors with the ALiBi
weights. Source: arXiv:2605.09213v1, §1, `eq:gpt-dyn-alibi` and
`eq:partition-function`. -/
theorem gptField_three_two (lam β : ℝ) (θ : Idx 3 → ℝ) :
    gptField lam β 3 θ 2 = (Real.exp (-(lam / 3) * 2) + Real.exp (-(lam / 3)))⁻¹ *
      (Real.exp (-(lam / 3) * 2) * wBetaDeriv β (θ 2 - θ 0) +
        Real.exp (-(lam / 3)) * wBetaDeriv β (θ 2 - θ 1)) := by
  simp [gptField, alibiZ, Finset.sum_filter, Fin.sum_univ_three]
  norm_num

/-- `w_β'` is smooth. -/
theorem contDiff_wBetaDeriv (β : ℝ) : ContDiff ℝ 1 (wBetaDeriv β) :=
  (contDiff_const.mul contDiff_sin).mul (contDiff_const.mul contDiff_cos).exp

/-- Three tokens whose first two sit at an equilibrium distance, and whose
third follows its field, form a GPT flow.
Source: arXiv:2605.09213v1, §1, `eq:gpt-dyn-alibi`. -/
theorem isGPTFlow_three (lam β y : ℝ) (hy : Real.sin y = 0) (x : ℝ → ℝ)
    (hx : ∀ t, HasDerivAt x (gptField lam β 3 ![0, y, x t] 2) t) :
    IsGPTFlow lam β 3 (fun t => ![0, y, x t]) := by
  intro t j
  fin_cases j
  · rw [show ((⟨0, by norm_num⟩ : Fin 3)) = 0 from rfl, gptField_three_zero]
    exact hasDerivAt_const t 0
  · rw [show ((⟨1, by norm_num⟩ : Fin 3)) = 1 from rfl,
      gptField_three_one_eq_zero lam β _ (by simpa using hy)]
    exact hasDerivAt_const t y
  · exact hx t

/-- The third token has a complete trajectory from `π/2` whenever the first two
sit at an equilibrium distance. Source: arXiv:2605.09213v1, §1,
`eq:gpt-dyn-alibi`, with `GlobalFlow.exists_global_Icc`. -/
theorem exists_three_flow (lam β y : ℝ) (hy : Real.sin y = 0) :
    ∃ x : ℝ → ℝ, x 0 = π / 2 ∧ ∀ t, HasDerivAt x (gptField lam β 3 ![0, y, x t] 2) t := by
  let F : ℝ → ℝ := fun z => gptField lam β 3 ![0, y, z] 2
  have hF : F = fun z => (Real.exp (-(lam / 3) * 2) + Real.exp (-(lam / 3)))⁻¹ *
      (Real.exp (-(lam / 3) * 2) * wBetaDeriv β (z - 0) +
        Real.exp (-(lam / 3)) * wBetaDeriv β (z - y)) := by
    funext z
    simp only [F, gptField_three_two]
    simp
  have hlip : LocallyLipschitz F := by
    rw [hF]
    exact (contDiff_const.mul ((contDiff_const.mul ((contDiff_wBetaDeriv β).comp
      (contDiff_id.sub contDiff_const))).add (contDiff_const.mul ((contDiff_wBetaDeriv β).comp
      (contDiff_id.sub contDiff_const))))).locallyLipschitz
  have hzero : ∀ z, Real.sin z = 0 → F z = 0 := by
    intro z hz
    have h1 : Real.sin (z - y) = 0 := by rw [Real.sin_sub, hz, hy]; ring
    rw [hF]
    simp only [sub_zero, wBetaDeriv_eq_zero_of_sin_eq_zero β hz,
      wBetaDeriv_eq_zero_of_sin_eq_zero β h1, mul_zero, add_zero]
  obtain ⟨x, hx0, hx⟩ := GlobalFlow.exists_global_Icc (x₀ := π / 2) hlip
    (hzero 0 Real.sin_zero) (hzero π Real.sin_pi) ⟨by positivity, by linarith [Real.pi_pos]⟩
  exact ⟨x, hx0, fun t => (hx t).1⟩

/-- At `π/2` the third token's field changes when the second token moves from
`0` to `π`, whenever `β ≠ 0`. -/
theorem gptField_three_pi_sub_zero (lam β : ℝ) (hβ : β ≠ 0) :
    gptField lam β 3 ![0, π, π / 2] 2 - gptField lam β 3 ![0, 0, π / 2] 2 ≠ 0 := by
  have hA : Real.exp (-(lam / 3) * 2) + Real.exp (-(lam / 3)) ≠ 0 := by positivity
  have key : gptField lam β 3 ![0, π, π / 2] 2 - gptField lam β 3 ![0, 0, π / 2] 2 =
      (Real.exp (-(lam / 3) * 2) + Real.exp (-(lam / 3)))⁻¹ *
        (2 * β * Real.exp (-(lam / 3))) := by
    rw [gptField_three_two, gptField_three_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.head_cons, Matrix.tail_cons, sub_zero]
    rw [show π / 2 - π = -(π / 2) by ring]
    simp only [wBetaDeriv, Real.sin_neg, Real.cos_neg, Real.sin_pi_div_two,
      Real.cos_pi_div_two, mul_zero, Real.exp_zero]
    ring
  rw [key]
  exact mul_ne_zero (inv_ne_zero hA)
    (mul_ne_zero (mul_ne_zero two_ne_zero hβ) (Real.exp_pos _).ne')

/-- Moving the second token by whole turns does not change the third token's
field. -/
theorem gptField_three_two_add_turns (lam β z : ℝ) (M : ℕ) :
    gptField lam β 3 ![0, π + 2 * π * M, z] 2 = gptField lam β 3 ![0, π, z] 2 := by
  rw [gptField_three_two, gptField_three_two]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.head_cons, Matrix.tail_cons]
  rw [show z - (π + 2 * π * M) = z - π - M * (2 * π) by ring,
    (wBetaDeriv_periodic β).sub_nat_mul_eq M]

/-- The hypotheses of the three-token lemmas hold at the resting pairs at
distance `0` and `π`, with the third token resting at `0`, and at `β = 1`. -/
example (lam β : ℝ) : Real.sin 0 = 0 ∧ Real.sin π = 0 ∧
    Real.sin ((![0, 0, 0] : Idx 3 → ℝ) 1 - ![0, 0, 0] 0) = 0 ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ))
      (gptField lam β 3 ![0, 0, (fun _ : ℝ => (0 : ℝ)) t] 2) t) ∧ (1 : ℝ) ≠ 0 := by
  refine ⟨Real.sin_zero, Real.sin_pi, by simp, fun t => ?_, one_ne_zero⟩
  rw [gptField_three_two]
  simpa [wBetaDeriv] using hasDerivAt_const t (0 : ℝ)

end Kinetic
end Transformer
