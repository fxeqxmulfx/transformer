/-
# Phase model over a finite abelian group — convergence to a character

As for `ℤ/n` (`PhaseModel.Convergence`), with `n = |G|`.  The periods span
`ℝ^G` (`span_periods`), and the phase energy is analytic, so gradient descent
on a phase energy with integer rows, with `0 < η` and `η Σ_r ‖a_r‖² < 1`,
converges from every start, to a critical point
(`exists_tendsto_gradient_phaseEnergy_eq_zero`).  For `n ≥ 4`, every
critical point of `V_full` below `n²/8` is a zero, a global minimizer, or a
strict saddle (`PhaseModel.Abelian.Landscape`).  So gradient descent with
`0 < η`, `9 n² η < 1`, started below `n²/8`, converges for almost every
start, and its limit is the phases of a character up to periods
(`ae_exists_tendsto_characterPhase`), by
`SaddleAvoidance.ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top_of_analyticOnNhd`
at the level `n²/8`.  In particular a convergent orbit ends at a character
(`ae_tendsto_characterPhase`).

Source: item M6.4 of `landscape_plan.md`.
-/

import Transformer.PhaseModel.Abelian.Landscape
import Transformer.SaddleAvoidance.Analytic
import Transformer.SaddleAvoidance.AnalyticLevelLandscape

open Filter MeasureTheory Real
open scoped Topology

namespace Transformer.PhaseModel.Abelian

open SaddleAvoidance (gradientStep)

/-- **The periods span `ℝ^G`**: they contain `2π e_x` for every `x`. -/
theorem span_periods {G : Type*} [Fintype G] : Submodule.span ℝ (periods G) = ⊤ := by
  classical
  rw [eq_top_iff, ← (EuclideanSpace.basisFun G ℝ).toBasis.span_eq, Submodule.span_le]
  rintro _ ⟨x, rfl⟩
  have h : EuclideanSpace.single x (2 * π) ∈ periods G := fun y => by
    by_cases hy : y = x
    · exact ⟨1, by simp [hy]⟩
    · exact ⟨0, by simp [hy]⟩
  have he : (EuclideanSpace.basisFun G ℝ).toBasis x =
      (2 * π)⁻¹ • EuclideanSpace.single x (2 * π) := by
    ext y
    by_cases hy : y = x
    · simp [hy]
      field_simp
    · simp [hy]
  rw [SetLike.mem_coe, he]
  exact Submodule.smul_mem _ _ (Submodule.subset_span h)

/-- **Gradient descent on a phase energy with integer rows converges from
every start, to a critical point**, for `0 < η` with `η Σ_r ‖a_r‖² < 1`: the
periods span `ℝ^G` and leave the energy invariant, and the energy is
analytic. -/
theorem exists_tendsto_gradient_phaseEnergy_eq_zero {G R : Type*} [Fintype G] [Fintype R]
    (a : R → G → ℤ) {η : ℝ} (hη : 0 < η) (hηL : η * ∑ r, ‖intVec (a r)‖ ^ 2 < 1)
    (x : EuclideanSpace ℝ G) :
    ∃ w, Tendsto (fun k => (gradientStep (phaseEnergy fun r => intVec (a r)) η)^[k] x) atTop
      (𝓝 w) ∧ gradient (phaseEnergy fun r => intVec (a r)) w = 0 := by
  obtain ⟨w, hw, hwx⟩ := SaddleAvoidance.exists_tendsto_of_span_eq_top_of_analyticOnNhd
    (analyticOnNhd_phaseEnergy _) (lipschitzWith_gradient_phaseEnergy _) hη
    (by push_cast; exact hηL) span_periods
    (fun _ hv ψ => phaseEnergy_add_of_mem_periods a hv ψ) x
  exact ⟨w, hwx, hw⟩

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]

/-- **Gradient descent on `V_full` started below `|G|²/8` converges to the
phases of a character**, for almost every start.  Let `|G| ≥ 4` and `0 < η`
with `9 |G|² η < 1`.  For a start `x` drawn from a law `ν` absolutely
continuous with respect to a Haar measure, almost surely: if
`V_full(x) < |G|²/8`, the orbit converges, and its limit is the phases of a
character up to periods.  Below `|G|²/8` every critical point of `V_full` is
a zero, a global minimizer, or a strict saddle
(`isStrictSaddle_of_divisionEnergy_lt`), and the zeros are the characters up
to periods (`divisionEnergy_univ_eq_zero_iff`). -/
theorem ae_exists_tendsto_characterPhase (hn : 4 ≤ Fintype.card G)
    (μ : Measure (EuclideanSpace ℝ G)) [μ.IsAddHaarMeasure] {ν : Measure (EuclideanSpace ℝ G)}
    (hν : ν ≪ μ) {η : ℝ} (hη : 0 < η) (hηn : 9 * Fintype.card G ^ 2 * η < 1) :
    ∀ᵐ x ∂ν, divisionEnergy (Finset.univ : Finset (G × G)) x < Fintype.card G ^ 2 / 8 →
      ∃ w, Tendsto (fun k => (gradientStep (divisionEnergy (Finset.univ :
        Finset (G × G))) η)^[k] x) atTop (𝓝 w) ∧
        ∃ χ : AddChar G ℂ, w - characterPhase χ ∈ periods G := by
  set V := divisionEnergy (Finset.univ : Finset (G × G)) with hVdef
  have hL := lipschitzWith_gradient_phaseEnergy
    (fun p : (Finset.univ : Finset (G × G)) => intVec (divisionRow p.1))
  have hηL : η * ((∑ p : (Finset.univ : Finset (G × G)),
      ‖intVec (divisionRow p.1)‖₊ ^ 2 : NNReal) : ℝ) < 1 := by
    push_cast
    have h := sum_norm_intVec_divisionRow_sq_le (Finset.univ : Finset (G × G))
    have hcard : ((Finset.univ : Finset (G × G)).card : ℝ) = Fintype.card G ^ 2 := by
      simp [sq]
    rw [hcard] at h
    nlinarith
  have hcrit : ∀ z, gradient V z = 0 → V z < Fintype.card G ^ 2 / 8 →
      IsMinOn V Set.univ z ∨ SaddleAvoidance.IsStrictSaddle V z := fun z hz hzc => by
    by_cases h0 : V z = 0
    · exact Or.inl (isMinOn_univ_iff.2 fun y => by rw [h0]; exact phaseEnergy_nonneg _ y)
    · exact Or.inr (isStrictSaddle_of_divisionEnergy_lt hn hz h0 hzc)
  filter_upwards [SaddleAvoidance.ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top_of_analyticOnNhd
    μ hν (analyticOnNhd_phaseEnergy _) hL hη hηL span_periods
    (fun _ hv ψ => phaseEnergy_add_of_mem_periods _ hv ψ) hcrit] with x hx hx0
  obtain ⟨w, hmin, hw⟩ := hx hx0
  refine ⟨w, hw, (divisionEnergy_univ_eq_zero_iff w).1
    (le_antisymm ?_ (phaseEnergy_nonneg _ w))⟩
  exact (isMinOn_univ_iff.1 hmin (characterPhase 0)).trans_eq
    (divisionEnergy_characterPhase _ 0)

/-- **Gradient descent on `V_full` started below `|G|²/8` converges, if at
all, to the phases of a character**, for almost every start.  Let
`|G| ≥ 4` and `0 < η` with `9 |G|² η < 1`.  For a start `x` drawn from a law
`ν` absolutely continuous with respect to a Haar measure, almost surely: if
`V_full(x) < |G|²/8` and the orbit converges to `w`, then `w` is the phases
of a character up to periods.  The orbit converges anyway
(`ae_exists_tendsto_characterPhase`), and limits are unique. -/
theorem ae_tendsto_characterPhase (hn : 4 ≤ Fintype.card G) (μ : Measure (EuclideanSpace ℝ G))
    [μ.IsAddHaarMeasure] {ν : Measure (EuclideanSpace ℝ G)} (hν : ν ≪ μ) {η : ℝ} (hη : 0 < η)
    (hηn : 9 * Fintype.card G ^ 2 * η < 1) :
    ∀ᵐ x ∂ν, divisionEnergy (Finset.univ : Finset (G × G)) x < Fintype.card G ^ 2 / 8 →
      ∀ w, Tendsto (fun k => (gradientStep (divisionEnergy (Finset.univ :
        Finset (G × G))) η)^[k] x) atTop (𝓝 w) →
        ∃ χ : AddChar G ℂ, w - characterPhase χ ∈ periods G := by
  filter_upwards [ae_exists_tendsto_characterPhase hn μ hν hη hηn] with x hx hx0 w hw
  obtain ⟨w', hw', hχ⟩ := hx hx0
  rwa [tendsto_nhds_unique hw hw']

/-- `exists_tendsto_gradient_phaseEnergy_eq_zero` applies to the rows of the
full division table of the Klein four-group `ℤ/2 × ℤ/2` with `η = 1/145`,
from the start `0`: the `16` rows have squared norms at most `9`. -/
example := exists_tendsto_gradient_phaseEnergy_eq_zero
  (fun p : (Finset.univ : Finset ((ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2))) => divisionRow p.1)
  (η := 1 / 145) (by norm_num) (by
    have h := sum_norm_intVec_divisionRow_sq_le
      (Finset.univ : Finset ((ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2)))
    have hcard : ((Finset.univ : Finset ((ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2))).card : ℝ) =
        16 := by
      simp [Fintype.card_prod, ZMod.card]
    rw [hcard] at h
    linarith) 0

/-- `ae_exists_tendsto_characterPhase` applies in the Klein four-group
`ℤ/2 × ℤ/2` with Lebesgue measure, for starts drawn from it, and
`η = 1/145`, as `9 · 4² = 144`. -/
example := ae_exists_tendsto_characterPhase (G := ZMod 2 × ZMod 2) (by simp) volume
  Measure.AbsolutelyContinuous.rfl (η := 1 / 145) (by norm_num)
  (by norm_num [Fintype.card_prod, ZMod.card])

/-- `ae_tendsto_characterPhase` applies in the Klein four-group `ℤ/2 × ℤ/2`
with Lebesgue measure, for starts drawn from it, and `η = 1/145`, as
`9 · 4² = 144`. -/
example := ae_tendsto_characterPhase (G := ZMod 2 × ZMod 2) (by simp) volume
  Measure.AbsolutelyContinuous.rfl (η := 1 / 145) (by norm_num)
  (by norm_num [Fintype.card_prod, ZMod.card])

end Transformer.PhaseModel.Abelian
