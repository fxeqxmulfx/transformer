/-
# AdaFisher — vanishing actual displacement and momentum error

arXiv:2405.16397v3, §3.4 and Appendix A.2, deterministic extension.
A lower loss bound makes the proved dissipation budget finite. Its two
strictly positive weights give convergence along the entire trajectory.
-/

import Transformer.AdaFisher.Section3_TrainingBudget
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Topology.Algebra.InfiniteSum.NatInt

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AdaFisher

open Optimization

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- The actual full training trajectory has summable weighted squared
displacements and momentum errors. Source: arXiv:2405.16397v3, §3.4,
explicit constant-step/full-gradient extension of Algorithm 1. -/
theorem trainingRun_dissipation_summable (η β γ δ L lower : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (hL : 0 < L) (hstep : 2 * L * η ≤ δ * (1 - β))
    (hlower : ∀ x, lower ≤ f x) :
    Summable (fun t : ℕ => trainingDissipation η β δ f
      (trainingRun η β γ δ f factors initial t)
      (trainingRun η β γ δ f factors initial (t + 1))) := by
  apply summable_of_sum_range_le
    (fun t => trainingDissipation_nonneg η β δ f _ _ hη hβ hβ' hδ)
    (c := f initial + η / (δ * (1 - β)) * ‖gradient f initial‖ ^ 2 - lower)
  intro T
  have h := trainingRun_dissipation_budget η β γ δ L f factors initial hf hgrad
    hη hβ hβ' hδ hL hstep T
  have he := loss_le_trainingEnergy η β δ f
    (trainingRun η β γ δ f factors initial T) hη hβ' hδ
  have hl := hlower (trainingPosition (trainingRun η β γ δ f factors initial T))
  linarith

/-- Summability assumptions hold for a nonconstant lower-bounded loss
with nonzero momentum, arXiv:2405.16397v3, §3.4, deterministic extension. -/
example : SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) ∧
    (∀ x : TrainingSpace 2 2, 0 ≤ energy x) := by
  exact ⟨energy_models.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- Actual parameter displacements and corrected-momentum errors both
vanish, with no assumption of bounded weights or gradients. Source:
arXiv:2405.16397v3, §3.4, deterministic full Algorithm 1 extension. -/
theorem trainingRun_velocity_error_tendsto_zero (η β γ δ L lower : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (hL : 0 < L) (hstep : 2 * L * η ≤ δ * (1 - β))
    (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => ‖trainingVelocity η β δ (trainingRun η β γ δ f factors initial t)‖)
      atTop (𝓝 0) ∧
    Tendsto (fun t : ℕ => ‖trainingError β f (trainingRun η β γ δ f factors initial t)‖)
      atTop (𝓝 0) := by
  let run := trainingRun η β γ δ f factors initial
  let D := fun t => trainingDissipation η β δ f (run t) (run (t + 1))
  have hD : Tendsto D atTop (𝓝 0) :=
    (trainingRun_dissipation_summable η β γ δ L lower f factors initial hf hgrad
      hη hβ hβ' hδ hL hstep hlower).tendsto_atTop_zero
  obtain ⟨hc, he⟩ := training_dissipation_coefficients η β δ hη hβ hβ' hδ
  have hV : Tendsto (fun t : ℕ => ‖trainingVelocity η β δ (run (t + 1))‖ ^ 2)
      atTop (𝓝 0) := by
    apply squeeze_zero (fun t => sq_nonneg _)
      (g := fun t => D t / (δ / (4 * η)))
    · intro t
      apply (le_div_iff₀ hc).mpr
      have h := mul_nonneg he.le (sq_nonneg ‖trainingError β f (run t)‖)
      dsimp only [D, trainingDissipation]
      nlinarith only [h]
    · simpa only [zero_div] using hD.div_const (δ / (4 * η))
  have hE : Tendsto (fun t : ℕ => ‖trainingError β f (run t)‖ ^ 2) atTop (𝓝 0) := by
    apply squeeze_zero (fun t => sq_nonneg _)
      (g := fun t => D t / (η * (1 - β ^ 2) / δ))
    · intro t
      apply (le_div_iff₀ he).mpr
      have h := mul_nonneg hc.le (sq_nonneg ‖trainingVelocity η β δ (run (t + 1))‖)
      dsimp only [D, trainingDissipation]
      nlinarith only [h]
    · simpa only [zero_div] using hD.div_const (η * (1 - β ^ 2) / δ)
  have hv := (Real.continuous_sqrt.tendsto 0).comp hV
  have herr := (Real.continuous_sqrt.tendsto 0).comp hE
  simp only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] at hv herr
  exact ⟨(Filter.tendsto_add_atTop_iff_nat 1).mp hv, herr⟩

/-- The full vanishing-error domain has a nonconstant quadratic witness,
arXiv:2405.16397v3, §3.4, deterministic training extension. -/
example : SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) ∧
    (∀ x : TrainingSpace 2 2, 0 ≤ energy x) := by
  exact ⟨energy_models.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.AdaFisher
