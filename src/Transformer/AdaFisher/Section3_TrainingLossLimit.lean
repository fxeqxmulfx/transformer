/-
# AdaFisher — a finite limit for the actual training loss

arXiv:2405.16397v3, §3.4, deterministic full-gradient extension.
Loss need not decrease at every momentum step. Its Lyapunov energy
decreases, and its squared corrected-momentum error tends to zero.
-/

import Transformer.AdaFisher.Section3_TrainingErrorLimit
import Mathlib.Topology.Order.MonotoneConvergence

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AdaFisher

open Optimization

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- The actual full training run has nonincreasing Lyapunov energy.
Source: arXiv:2405.16397v3, §3.4 and Appendix A.2, explicit deterministic
constant-step extension of Algorithm 1. -/
theorem trainingRun_energy_antitone (η β γ δ L : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (hL : 0 < L) (hstep : 2 * L * η ≤ δ * (1 - β)) :
    Antitone (fun t : ℕ => trainingEnergy η β δ f (trainingRun η β γ δ f factors initial t)) := by
  apply antitone_nat_of_succ_le
  intro t
  have h := trainingRun_energy_descent η β γ δ L f factors initial hf hgrad
    hη hβ hβ' hδ hL hstep t
  obtain ⟨hc, he⟩ := training_dissipation_coefficients η β δ hη hβ hβ' hδ
  have hnV := mul_nonneg hc.le
    (sq_nonneg ‖trainingVelocity η β δ (trainingRun η β γ δ f factors initial (t + 1))‖)
  have hnE := mul_nonneg he.le
    (sq_nonneg ‖trainingError β f (trainingRun η β γ δ f factors initial t)‖)
  linarith

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) := by
  exact ⟨energy_models.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- Actual training losses converge to a finite value above their lower
bound. Global optimality is not asserted for a nonconvex objective.
Source: arXiv:2405.16397v3, §3.4, deterministic constant-step extension
of the complete Algorithm 1 with bias correction and computed KF metric. -/
theorem trainingRun_loss_tendsto (η β γ δ L lower : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (hL : 0 < L) (hstep : 2 * L * η ≤ δ * (1 - β))
    (hlower : ∀ x, lower ≤ f x) :
    ∃ value : ℝ, lower ≤ value ∧
      Tendsto (fun t : ℕ => f (trainingPosition (trainingRun η β γ δ f factors initial t)))
        atTop (𝓝 value) := by
  let E := fun t => trainingEnergy η β δ f (trainingRun η β γ δ f factors initial t)
  let c := η / (δ * (1 - β))
  have hE : ∀ t, lower ≤ E t := fun t =>
    (hlower _).trans (loss_le_trainingEnergy η β δ f
      (trainingRun η β γ δ f factors initial t) hη hβ' hδ)
  have hbdd : BddBelow (Set.range E) := by
    refine ⟨lower, ?_⟩
    rintro _ ⟨t, rfl⟩
    exact hE t
  have hlim := tendsto_atTop_ciInf (trainingRun_energy_antitone η β γ δ L f factors initial
    hf hgrad hη hβ hβ' hδ hL hstep) hbdd
  have herr := (trainingRun_velocity_error_tendsto_zero η β γ δ L lower f factors initial
    hf hgrad hη hβ hβ' hδ hL hstep hlower).2
  refine ⟨⨅ t, E t, le_ciInf hE, ?_⟩
  have h := hlim.sub ((herr.pow 2).const_mul c)
  simpa only [E, c, zero_pow (by decide : 2 ≠ 0), mul_zero, sub_zero,
    trainingEnergy, add_sub_cancel_right] using h

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) ∧
    (∀ x : TrainingSpace 2 2, 0 ≤ energy x) := by
  exact ⟨energy_models.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.AdaFisher
