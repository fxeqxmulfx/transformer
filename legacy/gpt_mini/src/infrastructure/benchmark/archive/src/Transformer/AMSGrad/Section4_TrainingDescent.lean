/-
# AMSGrad — sufficient decrease of the actual Lyapunov energy

Training extension of arXiv:1904.03590v4, Algorithm 1, §4 and §6.
Constant momentum is allowed. The explicit step condition is
`0 < eta` and `L * eta <= epsilon`, with positive smoothness and epsilon.
Loss itself need not decrease at every momentum step; its Lyapunov
energy does, and the resulting displacement budget is proved here.
-/

import Transformer.AMSGrad.Section4_TrainingScalarEnergy
import Transformer.AMSGrad.Section4_TrainingKinetic
import Transformer.Optimization.Quadratic

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- Full actual AMSGrad momentum/preconditioner step decreases its
Lyapunov energy by `kinetic/(2*eta)`. Monotonicity of the maximum history
is used to derive the estimate; no candidate-alignment assumption or
direction guard is needed. Source: training extension of
arXiv:1904.03590v4, Algorithm 1, §4 and §6. -/
theorem trainingStep_energy_descent (η ε β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (s : TrainingState d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) :
    trainingEnergy η ε β f (trainingStep η ε β β₂ f s) ≤
      trainingEnergy η ε β f s - trainingKinetic η ε (trainingStep η ε β β₂ f s) / (2 * η) := by
  let next := trainingStep η ε β β₂ f s
  let u := trainingVelocity η ε s
  let v := trainingVelocity η ε next
  let c := β / (2 * η * (1 - β))
  have hcoordinate : ∀ i : Fin d,
      gradient f s.position i * v i + c * trainingDenominator ε next i * v i ^ 2 -
        c * trainingDenominator ε s i * u i ^ 2 ≤
          -trainingDenominator ε next i * v i ^ 2 / η := by
    intro i
    exact scalar_momentum_energy η β _ _ _ _ _ hη hβ hβ'
      (trainingDenominator_pos ε s hε i).le
      (trainingDenominator_mono η ε β β₂ f s i)
      (trainingVelocity_balance η ε β β₂ f s hε i)
  have hsum := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hcoordinate i)
  have hinner : ⟪gradient f s.position, v⟫_ℝ = ∑ i, gradient f s.position i * v i := by
    simp only [PiLp.inner_apply, RCLike.inner_apply, conj_trivial, mul_comm]
  have hkin : ⟪gradient f s.position, v⟫_ℝ + c * trainingKinetic η ε next -
      c * trainingKinetic η ε s ≤ -trainingKinetic η ε next / η := by
    dsimp only [u, v] at hsum
    simpa only [hinner, trainingKinetic, Finset.sum_add_distrib, Finset.sum_sub_distrib,
      mul_assoc, ← Finset.mul_sum, neg_mul, ← Finset.sum_div, ← Finset.sum_neg_distrib] using hsum
  have hposition : next.position - s.position = v := by
    rw [trainingStep_position]
    exact add_sub_cancel_left _ _
  have hsmooth := hf.2 s.position next.position
  rw [hposition] at hsmooth
  have hlower := trainingVelocity_energy_le η ε next
  have hbound : L / 2 * ‖v‖ ^ 2 ≤ trainingKinetic η ε next / (2 * η) := by
    have hc : 0 ≤ L / (2 * ε) := by positivity
    have hscaled := mul_le_mul_of_nonneg_left hlower hc
    have hcoef : L / (2 * ε) ≤ 1 / (2 * η) := by
      apply (div_le_div_iff₀ (by positivity : 0 < 2 * ε) (by positivity : 0 < 2 * η)).mpr
      nlinarith
    have hk := mul_le_mul_of_nonneg_right hcoef (trainingKinetic_nonneg η ε next hε.le)
    have heq : L / (2 * ε) * (ε * ‖v‖ ^ 2) = L / 2 * ‖v‖ ^ 2 := by field_simp
    rw [heq] at hscaled
    have heq' : 1 / (2 * η) * trainingKinetic η ε next =
        trainingKinetic η ε next / (2 * η) := by ring
    rw [heq'] at hk
    exact hscaled.trans hk
  have hdivide : trainingKinetic η ε next / η =
      2 * (trainingKinetic η ε next / (2 * η)) := by ring
  rw [neg_div, hdivide] at hkin
  change f next.position + c * trainingKinetic η ε next ≤
    f s.position + c * trainingKinetic η ε s - trainingKinetic η ε next / (2 * η)
  linarith

/-- All descent hypotheses hold on a nonconstant quadratic with momentum
`0.9`, arXiv:1904.03590v4, Algorithm 1, §4 and §6, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num⟩

/-- Telescoping controls every finite sum of actual weighted squared
displacements. Source: arXiv:1904.03590v4, Algorithm 1, §4,
fixed-objective training extension. -/
theorem trainingRun_kinetic_budget (η ε β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hstep : L * η ≤ ε) (T : ℕ) :
    (∑ t ∈ Finset.range T, trainingKinetic η ε (trainingRun η ε β β₂ f initial (t + 1))) /
        (2 * η) ≤ f initial - trainingEnergy η ε β f (trainingRun η ε β β₂ f initial T) := by
  induction T with
  | zero => simp [trainingEnergy_initial]
  | succ T ih =>
    rw [Finset.sum_range_succ, add_div]
    have h := trainingStep_energy_descent η ε β β₂ L f
      (trainingRun η ε β β₂ f initial T) hf hη hε hL hβ hβ' hstep
    change trainingEnergy η ε β f (trainingRun η ε β β₂ f initial (T + 1)) ≤
      trainingEnergy η ε β f (trainingRun η ε β β₂ f initial T) -
        trainingKinetic η ε (trainingRun η ε β β₂ f initial (T + 1)) / (2 * η) at h
    linarith

/-- The finite-budget domain is witnessed by a nonconstant quadratic and
nonzero momentum, arXiv:1904.03590v4, Algorithm 1, §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num⟩

end Transformer.AMSGrad
