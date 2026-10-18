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

Each theorem is that of `Landscape.PhaseModel.Abelian` at `G = Fin n` for the
character `cyclicChar n j`, whose phases are `2πjx/n` up to periods
(`characterPhase_sub_mem_periods`); for `n = 0` there is only one vector.

Source: item M6.8 of `landscape_plan.md`.
-/

import Transformer.Landscape.PhaseModel.Abelian.Characters
import Transformer.Landscape.PhaseModel.Basin
import Transformer.Landscape.PhaseModel.Orthogonality

open Real

namespace Transformer.Landscape.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- **The phases of a character leave the energy of every set of rows
invariant**: `V_S(ψ + (2πjx/n)_x) = V_S(ψ)`. -/
theorem divisionEnergy_add_characterPhase (S : Finset (Fin n × Fin n)) (j : ℤ)
    (ψ : EucSpace n) : divisionEnergy S (ψ + characterPhase n j) = divisionEnergy S ψ := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact congrArg _ (PiLp.ext fun t => t.elim0)
  have : NeZero n := ⟨hn.ne'⟩
  have h1 := Abelian.divisionEnergy_add_characterPhase S (cyclicChar n j)
    (ψ + (characterPhase n j - Abelian.characterPhase (cyclicChar n j)))
  have h2 := Abelian.phaseEnergy_add_of_mem_periods (fun p : S => divisionRow p.1)
    (characterPhase_sub_mem_periods j) ψ
  rw [add_assoc, sub_add_cancel] at h1
  exact h1.trans h2

/-- **A local minimizer of `V_full` near a character is that character** up
to periods: for `n ≥ 4`, if `ψ` is a local minimizer of `V_full` with
`Σ_x (1 - cos (ψ_x - 2πjx/n)) < n/8`, then `ψ - (2πjx/n)_x ∈ 2πℤⁿ`. -/
theorem sub_characterPhase_mem_periods_of_isLocalMin (hn : 4 ≤ n) {ψ : EucSpace n}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ) {j : ℤ}
    (hE : ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) < n / 8) :
    ψ - characterPhase n j ∈ periods n := by
  have : NeZero n := ⟨by omega⟩
  simp only [cos_sub_characterPhase] at hE
  exact (sub_characterPhase_mem_periods_iff ψ j).2
    (Abelian.sub_characterPhase_mem_periods_of_isLocalMin (G := Fin n)
      (by rwa [Fintype.card_fin]) hmin (by rwa [Fintype.card_fin]))

/-- **A critical point of `V_full` near a character that is not that
character is a strict saddle**: for `n ≥ 4`, if `∇V_full(ψ) = 0`,
`Σ_x (1 - cos (ψ_x - 2πjx/n)) < n/8` and `ψ - (2πjx/n)_x ∉ 2πℤⁿ`, then the
Hessian at `ψ` has a negative eigenvalue: a negative diagonal entry. -/
theorem isStrictSaddle_of_sum_one_sub_cos_lt (hn : 4 ≤ n) {ψ : EucSpace n}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ = 0) {j : ℤ}
    (hE : ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) < n / 8)
    (hψ : ψ - characterPhase n j ∉ periods n) :
    Landscape.SaddleAvoidance.IsStrictSaddle (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ := by
  have : NeZero n := ⟨by omega⟩
  simp only [cos_sub_characterPhase] at hE
  exact Abelian.isStrictSaddle_of_sum_one_sub_cos_lt (G := Fin n) (by rwa [Fintype.card_fin]) hg
    (by rwa [Fintype.card_fin]) fun h => hψ ((sub_characterPhase_mem_periods_iff ψ j).2 h)

/-- The hypotheses of `sub_characterPhase_mem_periods_of_isLocalMin` hold
for `n = 4` at the phases of the character `j = 1`, a global minimizer. -/
example : characterPhase 4 1 - characterPhase 4 1 ∈ periods 4 :=
  sub_characterPhase_mem_periods_of_isLocalMin le_rfl
    (Filter.Eventually.of_forall fun y => by
      rw [divisionEnergy_characterPhase]
      exact phaseEnergy_nonneg _ y)
    (by norm_num)

/-- The hypotheses of `isStrictSaddle_of_sum_one_sub_cos_lt` hold for
`n = 17`, `j = 0` and `ψ = π e_0`: every relation of `ψ` is a multiple of
`π`, so `ψ` is critical; `Σ_x (1 - cos ψ_x) = 2 < 17/8`; and `π ∉ 2πℤ`. -/
example : Landscape.SaddleAvoidance.IsStrictSaddle
    (divisionEnergy (Finset.univ : Finset (Fin 17 × Fin 17))) (PiLp.single 2 0 π) := by
  have hc : characterPhase 17 0 = 0 := PiLp.ext fun x => by simp [characterPhase]
  have hrel : ∀ a : Fin 17 → ℤ, ⟪intVec a, (PiLp.single 2 0 π : EucSpace 17)⟫ = (a 0 : ℤ) * π :=
    fun a => by simp [inner_intVec]
  refine isStrictSaddle_of_sum_one_sub_cos_lt (by norm_num) ?_ (j := 0) ?_ ?_
  · simp only [divisionEnergy]
    rw [gradient_phaseEnergy]
    exact Finset.sum_eq_zero fun r _ => by rw [hrel, sin_int_mul_pi, zero_smul]
  · have h : ∀ x : Fin 17, 1 - cos ((PiLp.single 2 0 π : EucSpace 17).ofLp x -
        (characterPhase 17 0).ofLp x) = if x = 0 then 2 else 0 := fun x => by
      by_cases hx : x = 0
      · simp only [hx, hc, PiLp.ofLp_single, Pi.single_eq_same, WithLp.ofLp_zero,
          Pi.zero_apply, sub_zero, cos_pi, ite_true]
        norm_num
      · simp [hx, hc]
    simp only [h, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    norm_num
  · rw [hc, sub_zero]
    intro h
    obtain ⟨k, hk⟩ := h 0
    simp only [PiLp.ofLp_single, Pi.single_eq_same] at hk
    have h1 : (1 : ℝ) = 2 * k := by nlinarith [pi_pos]
    have h2 : (1 : ℤ) = 2 * k := by exact_mod_cast h1
    omega

end Transformer.Landscape.PhaseModel
