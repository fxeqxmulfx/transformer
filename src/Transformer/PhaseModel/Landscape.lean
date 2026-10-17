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
(`isStrictSaddle_of_divisionEnergy_lt`); and gradient descent with `0 < η`,
`9 n² η < 1`, started below `n²/8`, almost surely converges to the phases of
a character up to periods whenever it converges
(`ae_tendsto_characterPhase`).  Whether the orbits converge, and the critical
points above `n²/8`, are not settled here.
-/

import Transformer.PhaseModel.Characters
import Transformer.PhaseModel.Fourier
import Transformer.SaddleAvoidance.Minimizers

open Filter MeasureTheory Real
open scoped Topology

namespace Transformer.PhaseModel

open SaddleAvoidance (gradientStep)

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {n : ℕ}

/-- **Below `n²/8`, the phases lie within `n/8` of a character**: if
`V_full(ψ) < n²/8`, some `j` has `Σ_x (1 - cos (ψ_x - 2πjx/n)) < n/8`. -/
theorem exists_sum_one_sub_cos_lt [NeZero n] {ψ : EucSpace n}
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    ∃ j : ℤ, ∑ x, (1 - cos (ψ.ofLp x - (characterPhase n j).ofLp x)) < n / 8 := by
  obtain ⟨j, hj⟩ := exists_mul_sum_one_sub_cos_le ψ
  have hn : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne n)
  refine ⟨j, ?_⟩
  by_contra h
  push Not at h
  nlinarith

/-- **Below `n²/8`, every local minimizer of `V_full` is a character**: for
`n ≥ 4`, a local minimizer `ψ` of `V_full` with `V_full(ψ) < n²/8` has
`ψ - (2πjx/n)_x ∈ 2πℤⁿ` for some `j`. -/
theorem exists_sub_characterPhase_mem_periods_of_isLocalMin (hn : 4 ≤ n) {ψ : EucSpace n}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ)
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    ∃ j : ℤ, ψ - characterPhase n j ∈ periods n := by
  have : NeZero n := ⟨by omega⟩
  obtain ⟨j, hj⟩ := exists_sum_one_sub_cos_lt hV
  exact ⟨j, sub_characterPhase_mem_periods_of_isLocalMin hn hmin hj⟩

/-- **Below `n²/8`, `V_full` has no spurious local minimizer**: for `n ≥ 4`,
a local minimizer `ψ` of `V_full` with `V_full(ψ) < n²/8` is a zero, a
global minimizer. -/
theorem divisionEnergy_eq_zero_of_isLocalMin (hn : 4 ≤ n) {ψ : EucSpace n}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ)
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ = 0 :=
  (divisionEnergy_univ_eq_zero_iff ψ).2
    (exists_sub_characterPhase_mem_periods_of_isLocalMin hn hmin hV)

/-- **Below `n²/8`, every critical point of `V_full` that is not a zero is a
strict saddle**: for `n ≥ 4`, at a critical point `ψ` with
`0 < V_full(ψ) < n²/8` the Hessian has a negative eigenvalue. -/
theorem isStrictSaddle_of_divisionEnergy_lt (hn : 4 ≤ n) {ψ : EucSpace n}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ = 0)
    (hV0 : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ ≠ 0)
    (hV : divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) ψ < n ^ 2 / 8) :
    SaddleAvoidance.IsStrictSaddle (divisionEnergy (Finset.univ : Finset (Fin n × Fin n))) ψ := by
  have : NeZero n := ⟨by omega⟩
  obtain ⟨j, hj⟩ := exists_sum_one_sub_cos_lt hV
  exact isStrictSaddle_of_sum_one_sub_cos_lt hn hg hj fun h =>
    hV0 ((divisionEnergy_univ_eq_zero_iff ψ).2 ⟨j, h⟩)

/-- **Gradient descent on `V_full` started below `n²/8` converges, if at
all, to the phases of a character**, for almost every start.  Let `n ≥ 4`
and `0 < η` with `9 n² η < 1`.  For a start `x` drawn from a law `ν`
absolutely continuous with respect to a Haar measure, almost surely: if
`V_full(x) < n²/8` and the orbit converges to `w`, then `w` is the phases of
a character up to periods.  The orbit decreases `V_full`, so
`V_full(w) < n²/8`; `w` is critical, and almost surely not a strict saddle
(`SaddleAvoidance.ae_not_tendsto_of_isStrictSaddle`). -/
theorem ae_tendsto_characterPhase (hn : 4 ≤ n) (μ : Measure (EucSpace n)) [μ.IsAddHaarMeasure]
    {ν : Measure (EucSpace n)} (hν : ν ≪ μ) {η : ℝ} (hη : 0 < η) (hηn : 9 * n ^ 2 * η < 1) :
    ∀ᵐ x ∂ν, divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) x < n ^ 2 / 8 →
      ∀ w, Tendsto (fun k => (gradientStep (divisionEnergy (Finset.univ :
        Finset (Fin n × Fin n))) η)^[k] x) atTop (𝓝 w) →
        ∃ j : ℤ, w - characterPhase n j ∈ periods n := by
  set V := divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) with hVdef
  have hL := lipschitzWith_gradient_phaseEnergy
    (fun p : (Finset.univ : Finset (Fin n × Fin n)) => intVec (divisionRow p.1))
  have hηL : η * ((∑ p : (Finset.univ : Finset (Fin n × Fin n)),
      ‖intVec (divisionRow p.1)‖₊ ^ 2 : NNReal) : ℝ) < 1 := by
    push_cast
    have h := sum_norm_intVec_divisionRow_sq_le (Finset.univ : Finset (Fin n × Fin n))
    have hcard : ((Finset.univ : Finset (Fin n × Fin n)).card : ℝ) = n ^ 2 := by simp [sq]
    rw [hcard] at h
    nlinarith
  have hfd : Differentiable ℝ V := (contDiff_phaseEnergy _ (k := 1)).differentiable one_ne_zero
  filter_upwards [SaddleAvoidance.ae_not_tendsto_of_isStrictSaddle μ hν (contDiff_phaseEnergy _)
    hL hη hηL] with x hx hx0 w hw
  have hg : gradient V w = 0 := SaddleAvoidance.gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw
  have hanti := SaddleAvoidance.antitone_iterate hfd hL hη.le hηL.le x
  have hle : V w ≤ V x :=
    le_of_tendsto' ((hfd.continuous.tendsto w).comp hw) fun k => by
      simpa using hanti (Nat.zero_le k)
  by_contra hne
  exact hx w (isStrictSaddle_of_divisionEnergy_lt hn hg
    (fun h => hne ((divisionEnergy_univ_eq_zero_iff w).1 h)) (hle.trans_lt hx0)) hw

/-- The hypotheses of the first four theorems hold for `n = 4` at the phases
of the character `j = 1`, a global minimizer with `V_full = 0 < 4²/8`. -/
example : let V := divisionEnergy (Finset.univ : Finset (Fin 4 × Fin 4))
    4 ≤ 4 ∧ IsLocalMin V (characterPhase 4 1) ∧
      V (characterPhase 4 1) < ((4 : ℕ) : ℝ) ^ 2 / 8 := by
  intro V
  have h0 : V (characterPhase 4 1) = 0 := divisionEnergy_characterPhase _ 1
  refine ⟨le_rfl, Filter.Eventually.of_forall fun y => ?_, by rw [h0]; norm_num⟩
  rw [h0]
  exact phaseEnergy_nonneg _ y

/-- The hypotheses of `isStrictSaddle_of_divisionEnergy_lt` hold for
`n ≥ 49` at `ψ = π e_0`: every relation of `ψ` is a multiple of `π`, so `ψ`
is critical; the relation of the pair `(0, 0)` is `-π`, so
`V_full(ψ) ≥ 2`; and only the pairs `(x, y)` with `x = 0`, `y = 0` or
`x = y` have a nonzero relation, each adding at most `2`, so
`V_full(ψ) ≤ 6n < n²/8`. -/
example (n : ℕ) [NeZero n] (hn : 49 ≤ n) :
    let V := divisionEnergy (Finset.univ : Finset (Fin n × Fin n))
    let ψ : EucSpace n := PiLp.single 2 0 π
    4 ≤ n ∧ gradient V ψ = 0 ∧ V ψ ≠ 0 ∧ V ψ < n ^ 2 / 8 := by
  intro V ψ
  have hψ : ∀ x : Fin n, ψ.ofLp x = if x = 0 then π else 0 := fun x => by
    simp [ψ]
  have hrel : ∀ a : Fin n → ℤ, ⟪intVec a, ψ⟫ = (a 0 : ℤ) * π := fun a => by
    simp [ψ, inner_intVec]
  have hV : V ψ = ∑ p : Fin n × Fin n,
      (1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2))) := divisionEnergy_apply _ ψ
  refine ⟨by omega, ?_, ?_, ?_⟩
  · simp only [V, divisionEnergy]
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

/-- The hypotheses of `ae_tendsto_characterPhase` hold for `n = 4` with
Lebesgue measure and `η = 1/145`, as `9 · 4² = 144`. -/
example : 4 ≤ 4 ∧ (volume : Measure (EucSpace 4)).IsAddHaarMeasure ∧
    (volume : Measure (EucSpace 4)) ≪ volume ∧ (0 : ℝ) < 1 / 145 ∧
    9 * ((4 : ℕ) : ℝ) ^ 2 * (1 / 145) < 1 :=
  ⟨le_rfl, inferInstance, Measure.AbsolutelyContinuous.rfl, by norm_num, by norm_num⟩

end Transformer.PhaseModel
