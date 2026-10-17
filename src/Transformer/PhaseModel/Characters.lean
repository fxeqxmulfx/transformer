/-
# Phase model — critical points of `V_full` near a character

The phases `(2πjx/n)_x` of a character leave the energy of every set of rows
of the division table invariant (`divisionEnergy_add_characterPhase`), so
the estimates near the zero `0` hold near every character.  For `n ≥ 4`, a
local minimizer of `V_full` with `Σ_x (1 - cos (ψ_x - 2πjx/n)) < n/8` is the
character `j` up to periods
(`sub_characterPhase_mem_periods_of_isLocalMin`), and a critical point there
that is not is a strict saddle (`isStrictSaddle_of_sum_one_sub_cos_lt`):
the Hessian has a negative diagonal entry.  The last example exhibits such
a saddle for `n = 17` and `j = 0`: the phases `π e_0` keep every relation in
`πℤ`, so they are critical, at `Σ_x (1 - cos ψ_x) = 2 < 17/8`.
-/

import Transformer.PhaseModel.Basin

open Real

namespace Transformer.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- **The phases of a character leave the energy of every set of rows
invariant**: `V_S(ψ + (2πjx/n)_x) = V_S(ψ)`. -/
theorem divisionEnergy_add_characterPhase (S : Finset (Fin n × Fin n)) (j : ℤ)
    (ψ : EucSpace n) : divisionEnergy S (ψ + characterPhase n j) = divisionEnergy S ψ :=
  phaseEnergy_add (fun p : S => intVec (divisionRow p.1))
    (fun r => exists_inner_intVec_divisionRow_characterPhase j r.1) ψ

/-- **A local minimizer of `V_full` near a character is that character** up
to periods: for `n ≥ 4`, if `ψ` is a local minimizer of `V_full` with
`Σ_x (1 - cos (ψ_x - 2πjx/n)) < n/8`, then `ψ - (2πjx/n)_x ∈ 2πℤⁿ`. -/
theorem sub_characterPhase_mem_periods_of_isLocalMin (hn : 4 ≤ n) {ψ : EucSpace n}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ) {j : ℤ}
    (hE : ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) < n / 8) :
    ψ - characterPhase n j ∈ periods n := by
  have hcomp : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ∘ (· + characterPhase n j) =
      divisionEnergy Finset.univ := funext (divisionEnergy_add_characterPhase _ j)
  have hmin' : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n)))
      (ψ - characterPhase n j) := by
    have h : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n)))
        ((· + characterPhase n j) (ψ - characterPhase n j)) := by
      simpa only [sub_add_cancel] using hmin
    have h2 := IsLocalMin.comp_continuous (g := fun x => x + characterPhase n j) h
      (by fun_prop : Continuous fun x : EucSpace n => x + characterPhase n j).continuousAt
    rwa [hcomp] at h2
  have hg := SaddleAvoidance.gradient_eq_zero_of_isLocalMin hmin'
  refine mem_periods_of_forall_half_lt_cos hg (half_lt_cos_of_gradient_eq_zero hn hg
    (fun t => SaddleAvoidance.inner_nonneg_of_isLocalMin
      ((contDiff_phaseEnergy _ (k := 1)).differentiable one_ne_zero)
      (hasFDerivAt_gradient_phaseEnergy _ _).differentiableAt.hasFDerivAt hmin' _) ?_)
  simpa only [WithLp.ofLp_sub, Pi.sub_apply] using hE

/-- **A critical point of `V_full` near a character that is not that
character is a strict saddle**: for `n ≥ 4`, if `∇V_full(ψ) = 0`,
`Σ_x (1 - cos (ψ_x - 2πjx/n)) < n/8` and `ψ - (2πjx/n)_x ∉ 2πℤⁿ`, then the
Hessian at `ψ` has a negative eigenvalue: a negative diagonal entry. -/
theorem isStrictSaddle_of_sum_one_sub_cos_lt (hn : 4 ≤ n) {ψ : EucSpace n}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ = 0) {j : ℤ}
    (hE : ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) < n / 8)
    (hψ : ψ - characterPhase n j ∉ periods n) :
    SaddleAvoidance.IsStrictSaddle (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ := by
  have hV := divisionEnergy_add_characterPhase (Finset.univ : Finset (Fin n × Fin n)) j
  have hgrad := SaddleAvoidance.gradient_add_eq hV (ψ - characterPhase n j)
  rw [sub_add_cancel] at hgrad
  have hH : fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n)))) ψ =
      fderiv ℝ (gradient (divisionEnergy Finset.univ)) (ψ - characterPhase n j) := by
    have h1 := fderiv_comp_add_right (𝕜 := ℝ)
      (f := gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))))
      (x := ψ - characterPhase n j) (characterPhase n j)
    rw [sub_add_cancel, funext (SaddleAvoidance.gradient_add_eq hV)] at h1
    exact h1.symm
  by_contra hs
  refine hψ (mem_periods_of_forall_half_lt_cos (hgrad ▸ hg)
    (half_lt_cos_of_gradient_eq_zero hn (hgrad ▸ hg) (fun t => ?_) ?_))
  · by_contra ht
    push Not at ht
    exact hs (SaddleAvoidance.isStrictSaddle_of_inner_neg
      (f := divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) (contDiff_phaseEnergy _) hg
      (h := PiLp.single 2 t 1) (by rw [hH]; exact ht))
  · simpa only [WithLp.ofLp_sub, Pi.sub_apply] using hE

/-- The hypotheses of `sub_characterPhase_mem_periods_of_isLocalMin` hold
for `n = 4` at the phases of the character `j = 1`, a global minimizer. -/
example : let V := divisionEnergy (Finset.univ : Finset (Fin 4 × Fin 4))
    4 ≤ 4 ∧ IsLocalMin V (characterPhase 4 1) ∧
      ∑ x, (1 - cos ((characterPhase 4 1).ofLp x - (characterPhase 4 1).ofLp x)) <
        ((4 : ℕ) : ℝ) / 8 := by
  intro V
  refine ⟨le_rfl, Filter.Eventually.of_forall fun y => ?_, by norm_num⟩
  rw [show V (characterPhase 4 1) = 0 from divisionEnergy_characterPhase _ 1]
  exact phaseEnergy_nonneg _ y

/-- The hypotheses of `isStrictSaddle_of_sum_one_sub_cos_lt` hold for
`n = 17`, `j = 0` and `ψ = π e_0`: every relation of `ψ` is a multiple of
`π`, so `ψ` is critical; `Σ_x (1 - cos ψ_x) = 2 < 17/8`; and `π ∉ 2πℤ`. -/
example : let V := divisionEnergy (Finset.univ : Finset (Fin 17 × Fin 17))
    let ψ : EucSpace 17 := PiLp.single 2 0 π
    4 ≤ 17 ∧ gradient V ψ = 0 ∧
      ∑ x, (1 - cos (ψ.ofLp x - (characterPhase 17 0).ofLp x)) < ((17 : ℕ) : ℝ) / 8 ∧
      ψ - characterPhase 17 0 ∉ periods 17 := by
  intro V ψ
  have hc : characterPhase 17 0 = 0 := PiLp.ext fun x => by simp [characterPhase]
  have hrel : ∀ a : Fin 17 → ℤ, ⟪intVec a, ψ⟫ = (a 0 : ℤ) * π := fun a => by
    simp [ψ, inner_intVec]
  refine ⟨by norm_num, ?_, ?_, ?_⟩
  · simp only [V, divisionEnergy]
    rw [gradient_phaseEnergy]
    exact Finset.sum_eq_zero fun r _ => by rw [hrel, sin_int_mul_pi, zero_smul]
  · have h : ∀ x : Fin 17, 1 - cos (ψ.ofLp x - (characterPhase 17 0).ofLp x) =
        if x = 0 then 2 else 0 := fun x => by
      by_cases hx : x = 0
      · simp only [ψ, hx, hc, PiLp.ofLp_single, Pi.single_eq_same, WithLp.ofLp_zero,
          Pi.zero_apply, sub_zero, cos_pi, ite_true]
        norm_num
      · simp [ψ, hx, hc]
    simp only [h, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    norm_num
  · rw [hc, sub_zero]
    intro h
    obtain ⟨k, hk⟩ := h 0
    simp only [ψ, PiLp.ofLp_single, Pi.single_eq_same] at hk
    have h1 : (1 : ℝ) = 2 * k := by nlinarith [pi_pos]
    have h2 : (1 : ℤ) = 2 * k := by exact_mod_cast h1
    omega

end Transformer.PhaseModel
