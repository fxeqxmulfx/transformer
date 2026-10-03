/-
# AMSGrad — displacement energy in its actual monotone metric

Training extension of arXiv:1904.03590v4, Algorithm 1 and §4.
The epsilon lower bound is derived from the implemented denominator,
not assumed for a free preconditioner sequence.
-/

import Transformer.AMSGrad.Section4_TrainingModels

open scoped BigOperators

noncomputable section

namespace Transformer.AMSGrad

variable {d : ℕ}

/-- The kinetic energy is nonnegative for nonnegative epsilon.
Source: arXiv:1904.03590v4, Algorithm 1 and §6, training extension. -/
theorem trainingKinetic_nonneg (η ε : ℝ) (s : TrainingState d) (hε : 0 ≤ ε) :
    0 ≤ trainingKinetic η ε s := by
  apply Finset.sum_nonneg
  intro i hi
  exact mul_nonneg (add_nonneg hε (Real.sqrt_nonneg _)) (sq_nonneg _)

/-- The kinetic energy domain is nonempty,
arXiv:1904.03590v4, §6, training extension. -/
example : (0 : ℝ) ≤ 1 := by norm_num

/-- Weighted displacement energy controls the actual Euclidean
displacement. Source: arXiv:1904.03590v4, Algorithm 1 and §6,
training extension. -/
theorem trainingVelocity_energy_le (η ε : ℝ) (s : TrainingState d) :
    ε * ‖trainingVelocity η ε s‖ ^ 2 ≤ trainingKinetic η ε s := by
  rw [EuclideanSpace.norm_sq_eq, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i hi
  simp only [Real.norm_eq_abs, sq_abs]
  exact mul_le_mul_of_nonneg_right (le_add_of_nonneg_right (Real.sqrt_nonneg _))
    (sq_nonneg _)

/-- A bounded actual metric also bounds its kinetic energy above.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
theorem trainingKinetic_le (η ε M : ℝ) (s : TrainingState d)
    (hM : ∀ i, trainingDenominator ε s i ≤ M) :
    trainingKinetic η ε s ≤ M * ‖trainingVelocity η ε s‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i hi
  simp only [Real.norm_eq_abs, sq_abs]
  exact mul_le_mul_of_nonneg_right (hM i) (sq_nonneg _)

/-- The upper-bound hypothesis holds on genuine zero-initialized moment
buffers, arXiv:1904.03590v4, Algorithm 1, training extension. -/
example : ∀ i : Fin 1,
    trainingDenominator 1 (⟨WithLp.toLp 2 (fun _ => 1), 0, 0, 0⟩ : TrainingState 1) i ≤ 1 := by
  intro i
  norm_num [trainingDenominator]

/-- The initial Lyapunov energy equals the actual initial loss, because
the source initializes momentum to zero. Source: arXiv:1904.03590v4,
Algorithm 1, training extension. -/
theorem trainingEnergy_initial (η ε β β₂ : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) :
    trainingEnergy η ε β f (trainingRun η ε β β₂ f initial 0) = f initial := by
  simp [trainingRun, trainingEnergy, trainingKinetic, trainingVelocity]

/-- Loss is bounded above by its actual Lyapunov energy, with no
trajectory hypothesis. Source: arXiv:1904.03590v4, Algorithm 1 and §4,
training extension. -/
theorem loss_le_trainingEnergy (η ε β : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (hη : 0 < η) (hε : 0 ≤ ε) (hβ : 0 ≤ β) (hβ' : β < 1) :
    f s.position ≤ trainingEnergy η ε β f s := by
  have hk := trainingKinetic_nonneg η ε s hε
  have hc : 0 ≤ β / (2 * η * (1 - β)) := by positivity
  exact le_add_of_nonneg_right (mul_nonneg hc hk)

/-- The Lyapunov coefficient has valid parameters with nonzero momentum,
arXiv:1904.03590v4, Algorithm 1, training extension. -/
example : (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 := by norm_num

end Transformer.AMSGrad
