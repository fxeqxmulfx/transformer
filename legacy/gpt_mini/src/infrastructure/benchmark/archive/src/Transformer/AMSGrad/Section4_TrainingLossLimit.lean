/-
# AMSGrad — finite limit of the actual training loss

Training extension of arXiv:1904.03590v4, Algorithm 1 and §4,
with the denominator regularizer of §6. The loss itself need not
decrease at each momentum step. Its Lyapunov energy is monotone,
and their difference tends to zero.
-/

import Transformer.AMSGrad.Section4_TrainingVelocityLimit
import Mathlib.Topology.Order.MonotoneConvergence

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- The actual run's Lyapunov energy is nonincreasing under the sufficient
constant-step condition. Source: arXiv:1904.03590v4, Algorithm 1 and §4,
fixed-objective/epsilon extension. -/
theorem trainingRun_energy_antitone (η ε β β₂ L : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) :
    Antitone (fun t : ℕ => trainingEnergy η ε β f (trainingRun η ε β β₂ f initial t)) := by
  apply antitone_nat_of_succ_le
  intro t
  have h := trainingStep_energy_descent η ε β β₂ L f
    (trainingRun η ε β β₂ f initial t) hf hη hε hL hβ hβ' hstep
  exact h.trans (sub_le_self _ (div_nonneg
    (trainingKinetic_nonneg η ε _ hε.le) (by positivity)))

/-- Monotone-energy hypotheses hold with a nonconstant loss and nonzero
momentum, arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num⟩

/-- The actual fixed-objective losses converge to a finite value above
their lower bound. This does not assert that a nonconvex loss reaches a
global minimum. Source: arXiv:1904.03590v4, Algorithm 1 and §4, explicit
constant-momentum, constant-step, unprojected extension with §6 epsilon. -/
theorem trainingRun_loss_tendsto (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x) :
    ∃ value : ℝ, lower ≤ value ∧
      Tendsto (fun t : ℕ => f (trainingRun η ε β β₂ f initial t).position)
        atTop (𝓝 value) := by
  let E := fun t => trainingEnergy η ε β f (trainingRun η ε β β₂ f initial t)
  let c := β / (2 * η * (1 - β))
  have hE : ∀ t, lower ≤ E t := fun t =>
    (hlower _).trans (loss_le_trainingEnergy η ε β f
      (trainingRun η ε β β₂ f initial t) hη hε.le hβ hβ')
  have hbdd : BddBelow (Set.range E) := by
    refine ⟨lower, ?_⟩
    rintro _ ⟨t, rfl⟩
    exact hE t
  have hlim := tendsto_atTop_ciInf
    (trainingRun_energy_antitone η ε β β₂ L f initial hf hη hε hL hβ hβ' hstep) hbdd
  have hK := trainingRun_kinetic_tendsto_zero η ε β β₂ L lower f initial hf
    hη hε hL hβ hβ' hstep hlower
  refine ⟨⨅ t, E t, le_ciInf hE, ?_⟩
  have h := hlim.sub (hK.const_mul c)
  simpa only [E, c, mul_zero, sub_zero, trainingEnergy, add_sub_cancel_right] using h

/-- Finite-loss-limit assumptions hold on a nonconstant lower-bounded
quadratic, arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.AMSGrad
