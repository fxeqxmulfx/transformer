/-
# AdaFisher — decrease of the full momentum-error Lyapunov energy

arXiv:2405.16397v3, §3.4 and Appendix A.2, deterministic extension.
Bias correction is retained. Damping and Lipschitz continuity control
both the actual displacement and the error of the corrected moment.
-/

import Transformer.AdaFisher.Section3_TrainingError
import Transformer.AdaFisher.Section3_TrainingLossDescent

noncomputable section

namespace Transformer.AdaFisher

open Optimization

variable {a b : ℕ}

/-- The initial energy contains the true initial gradient, since the
initial corrected moment is zero. Source: arXiv:2405.16397v3, Algorithm 1,
§3.4, deterministic training extension. -/
theorem trainingEnergy_initial [NeZero a] [NeZero b] (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ)
    (factors : TrainingFactors a b) (initial : TrainingSpace a b) :
    trainingEnergy η β δ f (trainingRun η β γ δ f factors initial 0) =
      f initial + η / (δ * (1 - β)) * ‖gradient f initial‖ ^ 2 := by
  have hz : (WithLp.toLp 2 (fun _ : Fin (a * b) => (0 : ℝ)) : TrainingSpace a b) = 0 := rfl
  simp [trainingEnergy, trainingRun, initializeOptimizer, trainingMoment,
    trainingError, trainingPosition, hz]

/-- The actual loss is below its Lyapunov energy. Source:
arXiv:2405.16397v3, §3.4, deterministic training extension. -/
theorem loss_le_trainingEnergy (η β δ : ℝ) (f : TrainingSpace a b → ℝ)
    (s : OptimizerState a b) (hη : 0 < η) (hβ' : β < 1) (hδ : 0 < δ) :
    f (trainingPosition s) ≤ trainingEnergy η β δ f s := by
  have hc : 0 ≤ η / (δ * (1 - β)) := by positivity
  exact le_add_of_nonneg_right (mul_nonneg hc (sq_nonneg _))

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) < 1 / 40 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 := by norm_num

/-- The actual run's Lyapunov energy pays for both parameter motion and
momentum error at every iteration. Source: arXiv:2405.16397v3, §3.4 and
Appendix A.2, explicit constant-step/full-gradient extension of Algorithm 1.
The sufficient condition is `2*L*eta <= damping*(1-beta)`; neither an
energy budget nor metric monotonicity is assumed. -/
theorem trainingRun_energy_descent [NeZero a] [NeZero b] (η β γ δ L : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (hL : 0 < L) (hstep : 2 * L * η ≤ δ * (1 - β)) (t : ℕ) :
    trainingEnergy η β δ f (trainingRun η β γ δ f factors initial (t + 1)) ≤
      trainingEnergy η β δ f (trainingRun η β γ δ f factors initial t) -
        δ / (4 * η) *
          ‖trainingVelocity η β δ (trainingRun η β γ δ f factors initial (t + 1))‖ ^ 2 -
        η * (1 - β ^ 2) / δ *
          ‖trainingError β f (trainingRun η β γ δ f factors initial t)‖ ^ 2 := by
  let s := trainingRun η β γ δ f factors initial t
  let next := trainingRun η β γ δ f factors initial (t + 1)
  let c := η / (δ * (1 - β))
  have hc : 0 ≤ c := by dsimp only [c]; positivity
  have hloss := trainingRun_loss_descent η β γ δ L f factors initial hf
    hη hβ hβ' hδ hL hstep t
  have herr := mul_le_mul_of_nonneg_left
    (trainingRun_error_bound η β γ δ L f factors initial hβ hβ' hgrad t) hc
  have hkin := mul_le_mul_of_nonneg_right
    (training_coefficient_bounds η β δ L hη hβ hβ' hδ hL hstep).2
    (sq_nonneg ‖trainingVelocity η β δ next‖)
  have hid : η * β ^ 2 / δ + c * β = c - η * (1 - β ^ 2) / δ := by
    dsimp only [c]
    field_simp [(sub_pos.mpr hβ').ne']
    ring
  have hidE := congrArg (fun z : ℝ => z * ‖trainingError β f s‖ ^ 2) hid
  have hV : δ / (2 * η) = 2 * (δ / (4 * η)) := by ring
  rw [hV] at hloss
  change f (trainingPosition next) + c * ‖trainingError β f next‖ ^ 2 ≤
    f (trainingPosition s) + c * ‖trainingError β f s‖ ^ 2 -
      δ / (4 * η) * ‖trainingVelocity η β δ next‖ ^ 2 -
      η * (1 - β ^ 2) / δ * ‖trainingError β f s‖ ^ 2
  nlinarith only [hloss, herr, hkin, hidE]

/-- The full energy-descent hypotheses hold on a nonconstant quadratic
with beta=0.9. Source: arXiv:2405.16397v3, §3.4, deterministic extension. -/
example : SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) := by
  exact ⟨energy_models.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num⟩

end Transformer.AdaFisher
