/-
# AMSGrad — summable displacement energy and vanishing velocities

Training extension of arXiv:1904.03590v4, Algorithm 1 and §4.
The estimates concern the entire actual learning trajectory. A lower
bound on the fixed objective suffices; boundedness of the weights is
not assumed, and no inverse-root iteration limit is involved.
-/

import Transformer.AMSGrad.Section4_TrainingGradientBound
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Algebra.InfiniteSum.NatInt

open scoped BigOperators Topology
open Filter

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- All actual weighted squared parameter displacements are summable.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
theorem trainingRun_kinetic_summable (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x) :
    Summable (fun t : ℕ => trainingKinetic η ε (trainingRun η ε β β₂ f initial (t + 1))) := by
  apply summable_of_sum_range_le
    (fun t => trainingKinetic_nonneg η ε _ hε.le) (c := 2 * η * (f initial - lower))
  intro T
  have hbudget := trainingRun_kinetic_budget η ε β β₂ L f initial hf hη hε hL hβ hβ' hstep T
  have henergy := loss_le_trainingEnergy η ε β f
    (trainingRun η ε β β₂ f initial T) hη hε.le hβ hβ'
  have hl := hlower (trainingRun η ε β β₂ f initial T).position
  have hsum := (div_le_iff₀ (show 0 < 2 * η by positivity)).mp
    (show (∑ t ∈ Finset.range T,
      trainingKinetic η ε (trainingRun η ε β β₂ f initial (t + 1))) / (2 * η) ≤
        f initial - lower by linarith)
  simpa only [mul_comm] using hsum

/-- Summability assumptions hold for a nonconstant quadratic and nonzero
momentum, arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- The entire actual sequence of weighted displacement energies tends
to zero. Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
theorem trainingRun_kinetic_tendsto_zero (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => trainingKinetic η ε (trainingRun η ε β β₂ f initial t))
      atTop (𝓝 0) := by
  have h := (trainingRun_kinetic_summable η ε β β₂ L lower f initial hf
    hη hε hL hβ hβ' hstep hlower).tendsto_atTop_zero
  exact (Filter.tendsto_add_atTop_iff_nat 1).mp h

/-- Vanishing-energy hypotheses hold on a nonconstant quadratic,
arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- Norms of actual consecutive parameter displacements vanish.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
theorem trainingRun_velocity_tendsto_zero (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => ‖trainingVelocity η ε (trainingRun η ε β β₂ f initial t)‖)
      atTop (𝓝 0) := by
  have hK := trainingRun_kinetic_tendsto_zero η ε β β₂ L lower f initial hf
    hη hε hL hβ hβ' hstep hlower
  have hsq : Tendsto (fun t : ℕ =>
      ‖trainingVelocity η ε (trainingRun η ε β β₂ f initial t)‖ ^ 2) atTop (𝓝 0) := by
    apply squeeze_zero (fun t => sq_nonneg _) (g := fun t =>
      trainingKinetic η ε (trainingRun η ε β β₂ f initial t) / ε)
    · intro t
      apply (le_div_iff₀ hε).mpr
      simpa only [mul_comm] using
        trainingVelocity_energy_le η ε (trainingRun η ε β β₂ f initial t)
    · simpa only [zero_div] using hK.div_const ε
  have h := (Real.continuous_sqrt.tendsto 0).comp hsq
  simpa only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] using h

/-- Actual-displacement convergence assumptions have a nonconstant
quadratic witness, arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.AMSGrad
