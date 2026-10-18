/-
# Phase model — the landscape of `V_full` below `n²/8`

For every phases `ψ`, some character `j` has
`n Σ_x (1 - cos (ψ_x - 2πjx/n)) ≤ V_full(ψ)` (`exists_mul_sum_one_sub_cos_le`),
so below `n²/8` the phases lie within `n/8` of a character
(`exists_sum_one_sub_cos_lt`), where for `n ≥ 4` every critical point is the
character up to periods or a strict saddle (`Characters`).  Hence, for
`n ≥ 4`, every local minimizer of `V_full` below `n²/8` is the phases of a
character up to periods (`exists_sub_characterPhase_mem_periods_of_isLocalMin`),
a global minimizer (`divisionEnergy_eq_zero_of_isLocalMin`); every critical
point with `0 < V_full < n²/8` is a strict saddle
(`isStrictSaddle_of_divisionEnergy_lt`).  So gradient descent started below
`n²/8` converges to a character for almost every start
(`Landscape.PhaseModel.Convergence`).  The critical points above `n²/8` are not
settled here.

Each theorem is that of `Landscape.PhaseModel.Abelian.Landscape` at `G = Fin n`, the
character `χ` found there being `cyclicChar n j` for some `j < n`
(`bijective_cyclicChar`).

Source: item M6.8 of `landscape_plan.md`.
-/

import Transformer.Landscape.PhaseModel.Abelian.Landscape
import Transformer.Landscape.PhaseModel.Characters
import Transformer.Landscape.PhaseModel.Fourier

open Real

namespace Transformer.Landscape.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- **Below `n²/8`, the phases lie within `n/8` of a character**: if
`V_full(ψ) < n²/8`, some `j` has `Σ_x (1 - cos (ψ_x - 2πjx/n)) < n/8`. -/
theorem exists_sum_one_sub_cos_lt [NeZero n] {ψ : EucSpace n}
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    ∃ j : ℤ, ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) < n / 8 := by
  obtain ⟨χ, hχ⟩ := Abelian.exists_sum_one_sub_cos_lt (G := Fin n) (ψ := ψ)
    (by rw [Fintype.card_fin]; exact hV)
  obtain ⟨j, rfl⟩ := bijective_cyclicChar.2 χ
  refine ⟨(j : ℕ), ?_⟩
  simp only [Fintype.card_fin] at hχ
  simp only [cos_sub_characterPhase]
  exact hχ

/-- **Below `n²/8`, every local minimizer of `V_full` is a character**: for
`n ≥ 4`, a local minimizer `ψ` of `V_full` with `V_full(ψ) < n²/8` has
`ψ - (2πjx/n)_x ∈ 2πℤⁿ` for some `j`. -/
theorem exists_sub_characterPhase_mem_periods_of_isLocalMin (hn : 4 ≤ n) {ψ : EucSpace n}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ)
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    ∃ j : ℤ, ψ - characterPhase n j ∈ periods n := by
  have : NeZero n := ⟨by omega⟩
  obtain ⟨χ, hχ⟩ := Abelian.exists_sub_characterPhase_mem_periods_of_isLocalMin (G := Fin n)
    (by rwa [Fintype.card_fin]) hmin (by rw [Fintype.card_fin]; exact hV)
  obtain ⟨j, rfl⟩ := bijective_cyclicChar.2 χ
  exact ⟨(j : ℕ), (sub_characterPhase_mem_periods_iff ψ _).2 hχ⟩

/-- **Below `n²/8`, `V_full` has no spurious local minimizer**: for `n ≥ 4`,
a local minimizer `ψ` of `V_full` with `V_full(ψ) < n²/8` is a zero, a
global minimizer. -/
theorem divisionEnergy_eq_zero_of_isLocalMin (hn : 4 ≤ n) {ψ : EucSpace n}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ)
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ = 0 := by
  have : NeZero n := ⟨by omega⟩
  exact Abelian.divisionEnergy_eq_zero_of_isLocalMin (G := Fin n) (by rwa [Fintype.card_fin]) hmin
    (by rw [Fintype.card_fin]; exact hV)

/-- **Below `n²/8`, every critical point of `V_full` that is not a zero is a
strict saddle**: for `n ≥ 4`, at a critical point `ψ` with
`0 < V_full(ψ) < n²/8` the Hessian has a negative eigenvalue. -/
theorem isStrictSaddle_of_divisionEnergy_lt (hn : 4 ≤ n) {ψ : EucSpace n}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ = 0)
    (hV0 : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ ≠ 0)
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    Landscape.SaddleAvoidance.IsStrictSaddle (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ := by
  have : NeZero n := ⟨by omega⟩
  exact Abelian.isStrictSaddle_of_divisionEnergy_lt (G := Fin n) (by rwa [Fintype.card_fin]) hg
    hV0 (by rw [Fintype.card_fin]; exact hV)

/-- The hypotheses of the first three theorems hold for `n = 4` at the
phases of the character `j = 1`, a global minimizer with
`V_full = 0 < 4²/8`. -/
example : (∃ j : ℤ, ∑ x, (1 - cos ((characterPhase 4 1).ofLp x - (characterPhase 4 j).ofLp x)) <
      ((4 : ℕ) : ℝ) / 8) ∧ (∃ j : ℤ, characterPhase 4 1 - characterPhase 4 j ∈ periods 4) ∧
    divisionEnergy (Finset.univ : Finset (Fin 4 × Fin 4)) (characterPhase 4 1) = 0 := by
  have hV : divisionEnergy (Finset.univ : Finset (Fin 4 × Fin 4)) (characterPhase 4 1) <
      ((4 : ℕ) : ℝ) ^ 2 / 8 := by
    rw [divisionEnergy_characterPhase]
    norm_num
  have hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin 4 × Fin 4)))
      (characterPhase 4 1) := Filter.Eventually.of_forall fun y => by
    rw [divisionEnergy_characterPhase]
    exact phaseEnergy_nonneg _ y
  exact ⟨exists_sum_one_sub_cos_lt hV, exists_sub_characterPhase_mem_periods_of_isLocalMin le_rfl
    hmin hV, divisionEnergy_eq_zero_of_isLocalMin le_rfl hmin hV⟩

/-- The hypotheses of `isStrictSaddle_of_divisionEnergy_lt` hold for
`n ≥ 49` at `ψ = π e_0`: every relation of `ψ` is a multiple of `π`, so `ψ`
is critical; the relation of the pair `(0, 0)` is `-π`, so
`V_full(ψ) ≥ 2`; and only the pairs `(x, y)` with `x = 0`, `y = 0` or
`x = y` have a nonzero relation, each adding at most `2`, so
`V_full(ψ) ≤ 6n < n²/8`. -/
example (n : ℕ) [NeZero n] (hn : 49 ≤ n) : Landscape.SaddleAvoidance.IsStrictSaddle
    (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) (PiLp.single 2 0 π) := by
  set V := divisionEnergy (Finset.univ : Finset (Fin n × Fin n))
  set ψ : EucSpace n := PiLp.single 2 0 π with hψdef
  have hψ : ∀ x : Fin n, ψ.ofLp x = if x = 0 then π else 0 := fun x => by
    simp [ψ]
  have hrel : ∀ a : Fin n → ℤ, ⟪intVec a, ψ⟫ = (a 0 : ℤ) * π := fun a => by
    simp [ψ, inner_intVec]
  have hV : V ψ = ∑ p : Fin n × Fin n,
      (1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2))) := divisionEnergy_apply _ ψ
  refine isStrictSaddle_of_divisionEnergy_lt (by omega) ?_ ?_ ?_
  · simp only [divisionEnergy]
    rw [gradient_phaseEnergy]
    exact Finset.sum_eq_zero fun r _ => by rw [hrel, sin_int_mul_pi, zero_smul]
  · have hnn : ∀ p : Fin n × Fin n,
        0 ≤ 1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2)) :=
      fun p => sub_nonneg.2 (cos_le_one _)
    have h0 := Finset.single_le_sum (fun p _ => hnn p)
      (Finset.mem_univ ((0 : Fin n), (0 : Fin n)))
    rw [← hV] at h0
    simp only [sub_self, zero_sub, cos_neg, hψ, ite_true, cos_pi] at h0
    intro h
    linarith
  · have hb : ∀ p : Fin n × Fin n, 1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2)) ≤
        2 * ((if p.1 = 0 then 1 else 0) + (if p.2 = 0 then 1 else 0) +
          (if p.1 = p.2 then 1 else 0)) := by
      rintro ⟨x, y⟩
      dsimp only
      have hc := neg_one_le_cos (ψ.ofLp x - ψ.ofLp y - ψ.ofLp (x - y))
      by_cases hx : x = 0
      · rw [ite_eq_left hx]
        split_ifs <;> linarith
      by_cases hy : y = 0
      · rw [ite_eq_left hy]
        split_ifs <;> linarith
      by_cases hxy : x = y
      · rw [ite_eq_left hxy]
        split_ifs
        linarith
      simp [hψ, hx, hy, hxy, sub_eq_zero]
    have hsum : ∑ p : Fin n × Fin n, 2 * ((if p.1 = 0 then (1 : ℝ) else 0) +
        (if p.2 = 0 then 1 else 0) + (if p.1 = p.2 then 1 else 0)) = 6 * n := by
      rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_add_distrib, Fintype.sum_prod_type,
        Finset.sum_comm, Fintype.sum_prod_type, Fintype.sum_prod_type]
      simp only [Finset.sum_ite_eq, Finset.sum_ite_eq', Finset.mem_univ, ite_true,
        Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
      ring
    have hle : V ψ ≤ 6 * n := by
      rw [hV, ← hsum]
      exact Finset.sum_le_sum fun p _ => hb p
    have hn' : (49 : ℝ) ≤ n := by exact_mod_cast hn
    nlinarith

end Transformer.Landscape.PhaseModel
