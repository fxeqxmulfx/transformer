/-
# Phase model — convergence to a character from below `n²/8`

For `n ≥ 4`, every critical point of `V_full` below `n²/8` is a zero, a
global minimizer, or a strict saddle (`PhaseModel.Landscape`), and `V_full`
is analytic.  So gradient descent with `0 < η`, `9 n² η < 1`, started below
`n²/8`, converges for almost every start, and its limit is the phases of a
character up to periods (`ae_exists_tendsto_characterPhase`):
`SaddleAvoidance.ae_exists_tendsto_isMinOn_of_lt_of_analyticOnNhd` applies at
the level `n²/8`, with the periods as translations, every orbit having a
limit point up to periods in the cube.  In particular a convergent orbit ends
at a character (`ae_tendsto_characterPhase`).

Source: item M6.4 of `landscape_plan.md`.
-/

import Transformer.PhaseModel.Descent
import Transformer.PhaseModel.Landscape
import Transformer.SaddleAvoidance.AnalyticLevelLandscape

open Filter MeasureTheory
open scoped Topology

namespace Transformer.PhaseModel

open SaddleAvoidance (gradientStep)

variable {n : ℕ}

/-- **Gradient descent on `V_full` started below `n²/8` converges to the
phases of a character**, for almost every start.  Let `n ≥ 4` and `0 < η`
with `9 n² η < 1`.  For a start `x` drawn from a law `ν` absolutely
continuous with respect to a Haar measure, almost surely: if
`V_full(x) < n²/8`, the orbit converges, and its limit is the phases of a
character up to periods.  Below `n²/8` every critical point of `V_full` is a
zero, a global minimizer, or a strict saddle
(`isStrictSaddle_of_divisionEnergy_lt`), and the zeros are the characters up
to periods (`divisionEnergy_univ_eq_zero_iff`). -/
theorem ae_exists_tendsto_characterPhase (hn : 4 ≤ n) (μ : Measure (EucSpace n))
    [μ.IsAddHaarMeasure] {ν : Measure (EucSpace n)} (hν : ν ≪ μ) {η : ℝ} (hη : 0 < η)
    (hηn : 9 * n ^ 2 * η < 1) :
    ∀ᵐ x ∂ν, divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) x < n ^ 2 / 8 →
      ∃ w, Tendsto (fun k => (gradientStep (divisionEnergy (Finset.univ :
        Finset (Fin n × Fin n))) η)^[k] x) atTop (𝓝 w) ∧
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
  have hcrit : ∀ z, gradient V z = 0 → V z < n ^ 2 / 8 →
      IsMinOn V Set.univ z ∨ SaddleAvoidance.IsStrictSaddle V z := fun z hz hzc => by
    by_cases h0 : V z = 0
    · exact Or.inl (isMinOn_univ_iff.2 fun y => by rw [h0]; exact phaseEnergy_nonneg _ y)
    · exact Or.inr (isStrictSaddle_of_divisionEnergy_lt hn hz h0 hzc)
  have hcl : ∀ᵐ x ∂ν, V x < n ^ 2 / 8 → ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ periods n,
      ‖(gradientStep V η)^[k] x + v - z‖ < ε := ae_of_all _ fun x _ =>
    (exists_frequently_periods (fun p : (Finset.univ : Finset (Fin n × Fin n)) =>
      divisionRow p.1) η x).imp fun _ h => h.2
  filter_upwards [SaddleAvoidance.ae_exists_tendsto_isMinOn_of_lt_of_analyticOnNhd μ hν
    (analyticOnNhd_phaseEnergy _) hL hη hηL
    (fun _ hv ψ => phaseEnergy_add_of_mem_periods _ hv ψ) hcrit hcl] with x hx hx0
  obtain ⟨w, hmin, hw⟩ := hx hx0
  refine ⟨w, hw, (divisionEnergy_univ_eq_zero_iff w).1
    (le_antisymm ?_ (phaseEnergy_nonneg _ w))⟩
  exact (isMinOn_univ_iff.1 hmin (characterPhase n 1)).trans_eq
    (divisionEnergy_characterPhase _ 1)

/-- **Gradient descent on `V_full` started below `n²/8` converges, if at
all, to the phases of a character**, for almost every start.  Let `n ≥ 4`
and `0 < η` with `9 n² η < 1`.  For a start `x` drawn from a law `ν`
absolutely continuous with respect to a Haar measure, almost surely: if
`V_full(x) < n²/8` and the orbit converges to `w`, then `w` is the phases of
a character up to periods.  The orbit converges anyway
(`ae_exists_tendsto_characterPhase`), and limits are unique. -/
theorem ae_tendsto_characterPhase (hn : 4 ≤ n) (μ : Measure (EucSpace n)) [μ.IsAddHaarMeasure]
    {ν : Measure (EucSpace n)} (hν : ν ≪ μ) {η : ℝ} (hη : 0 < η) (hηn : 9 * n ^ 2 * η < 1) :
    ∀ᵐ x ∂ν, divisionEnergy (Finset.univ : Finset (Fin n × Fin n)) x < n ^ 2 / 8 →
      ∀ w, Tendsto (fun k => (gradientStep (divisionEnergy (Finset.univ :
        Finset (Fin n × Fin n))) η)^[k] x) atTop (𝓝 w) →
        ∃ j : ℤ, w - characterPhase n j ∈ periods n := by
  filter_upwards [ae_exists_tendsto_characterPhase hn μ hν hη hηn] with x hx hx0 w hw
  obtain ⟨w', hw', hj⟩ := hx hx0
  rwa [tendsto_nhds_unique hw hw']

/-- `ae_exists_tendsto_characterPhase` applies for `n = 4` with Lebesgue
measure, for starts drawn from it, and `η = 1/145`, as `9 · 4² = 144`. -/
example := ae_exists_tendsto_characterPhase (n := 4) le_rfl volume
  Measure.AbsolutelyContinuous.rfl (η := 1 / 145) (by norm_num) (by norm_num)

/-- `ae_tendsto_characterPhase` applies for `n = 4` with Lebesgue measure,
for starts drawn from it, and `η = 1/145`, as `9 · 4² = 144`. -/
example := ae_tendsto_characterPhase (n := 4) le_rfl volume
  Measure.AbsolutelyContinuous.rfl (η := 1 / 145) (by norm_num) (by norm_num)

end Transformer.PhaseModel
