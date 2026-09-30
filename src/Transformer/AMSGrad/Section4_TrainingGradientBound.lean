/-
# AMSGrad — derive gradient bounds from the objective

Training extension of arXiv:1904.03590v4, Algorithm 1 and §4.
No global bounded-gradient premise is imposed on a strongly convex
objective on all of Euclidean space. Smoothness and the lower loss
bound instead bound gradients on the actual Lyapunov-controlled run.
-/

import Transformer.AMSGrad.Section4_TrainingDescent
import Transformer.Optimization.Descent

open scoped BigOperators

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- Smoothness and a global lower loss bound control the genuine gradient
energy at any point. This is derived from the objective, not assumed of
its optimizer trajectory. Source: training extension of
arXiv:1904.03590v4, Algorithm 1 and §4. -/
theorem smooth_gradient_energy_bound (f : TrainingSpace d → ℝ) (L lower : ℝ)
    (hf : SmoothObjective f L) (hL : 0 < L) (hlower : ∀ x, lower ≤ f x)
    (x : TrainingSpace d) : ‖gradient f x‖ ^ 2 ≤ 2 * L * (f x - lower) := by
  have h := safeguardedStep_descent f 1 L x (gradient f x) hf hL (by norm_num) le_rfl
  have hl := hlower (safeguardedStep 1 L x (gradient f x) (gradient f x))
  have hden : 0 < 2 * L := by positivity
  have hm := mul_le_mul_of_nonneg_left (show ‖gradient f x‖ ^ 2 / (2 * L) ≤ f x - lower by
    norm_num only [one_pow, one_div, inv_mul_eq_div] at h
    linarith) hden.le
  have heq : 2 * L * (‖gradient f x‖ ^ 2 / (2 * L)) = ‖gradient f x‖ ^ 2 := by
    field_simp
  rwa [heq] at hm

/-- The gradient-bound assumptions hold for a nonconstant strongly convex
loss, arXiv:1904.03590v4, §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧ (0 : ℝ) < 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

/-- The actual Lyapunov energy never exceeds the initial loss.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
theorem trainingRun_energy_le_initial (η ε β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) (t : ℕ) :
    trainingEnergy η ε β f (trainingRun η ε β β₂ f initial t) ≤ f initial := by
  have hbudget := trainingRun_kinetic_budget η ε β β₂ L f initial hf hη hε hL hβ hβ' hstep t
  have hsum : 0 ≤ ∑ k ∈ Finset.range t,
      trainingKinetic η ε (trainingRun η ε β β₂ f initial (k + 1)) :=
    Finset.sum_nonneg fun k hk => trainingKinetic_nonneg η ε _ hε.le
  have hdiv := div_nonneg hsum (show 0 ≤ 2 * η by positivity)
  linarith

/-- Initial-loss control has a nonconstant witness with nonzero momentum,
arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num⟩

/-- Every coordinate of the actual full gradient is bounded by the
initial loss gap. No boundedness or convergence premise about iterates
is supplied. Source: arXiv:1904.03590v4, Algorithm 1 and §4,
fixed-objective training extension. -/
theorem trainingRun_gradient_coordinate_bound (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x) (t : ℕ) (i : Fin d) :
    (gradient f (trainingRun η ε β β₂ f initial t).position i) ^ 2 ≤
      2 * L * (f initial - lower) := by
  let s := trainingRun η ε β β₂ f initial t
  have henergy := trainingRun_energy_le_initial η ε β β₂ L f initial hf hη hε hL hβ hβ' hstep t
  have hloss := loss_le_trainingEnergy η ε β f s hη hε.le hβ hβ'
  have hg := smooth_gradient_energy_bound f L lower hf hL hlower s.position
  have hcoord : (gradient f s.position i) ^ 2 ≤ ‖gradient f s.position‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq]
    simpa only [Real.norm_eq_abs, sq_abs] using
      (Finset.single_le_sum (fun j (_ : j ∈ Finset.univ) => sq_nonneg ‖gradient f s.position j‖)
        (Finset.mem_univ i))
  have hmul := mul_le_mul_of_nonneg_left (show f s.position - lower ≤ f initial - lower by
    linarith) (show 0 ≤ 2 * L by positivity)
  linarith

/-- All trajectory-bound hypotheses hold on a nonconstant quadratic,
arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.AMSGrad
