/-
# AMSGrad — gradient energy from consecutive actual displacements

Training extension of arXiv:1904.03590v4, Algorithm 1 and §4.
The actual bounded adaptive metric and exact momentum balance prevent
vanishing updates from hiding a nonzero objective gradient.
-/

import Transformer.AMSGrad.Section4_TrainingMetricBound

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- A bounded positive metric converts the exact momentum balance into
a squared-gradient bound. Source: arXiv:1904.03590v4, Algorithm 1 and §4,
fixed-objective extension. The final run theorem derives these metric
bounds, rather than assuming them about an arbitrary trajectory. -/
theorem scalar_balance_gradient_bound (η β a b u v g M : ℝ)
    (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (ha : 0 ≤ a) (haM : a ≤ M) (hb : 0 ≤ b) (hbM : b ≤ M)
    (hbalance : b * v = β * a * u - η * (1 - β) * g) :
    g ^ 2 ≤ 2 * M ^ 2 / (η ^ 2 * (1 - β) ^ 2) * (u ^ 2 + v ^ 2) := by
  have hM : 0 ≤ M := ha.trans haM
  have hβa : 0 ≤ β * a := mul_nonneg hβ ha
  have hβaM : β * a ≤ M := by
    have h := mul_nonneg ha (sub_nonneg.mpr hβ'.le)
    nlinarith
  have hA : (β * a) ^ 2 ≤ M ^ 2 := by nlinarith
  have hB : b ^ 2 ≤ M ^ 2 := by nlinarith
  have hu := mul_le_mul_of_nonneg_right hA (sq_nonneg u)
  have hv := mul_le_mul_of_nonneg_right hB (sq_nonneg v)
  have heq : η * (1 - β) * g = β * a * u - b * v := by linarith
  have hbal := congrArg (fun z : ℝ => z ^ 2) heq
  have hpoly : (η ^ 2 * (1 - β) ^ 2) * g ^ 2 ≤
      2 * M ^ 2 * (u ^ 2 + v ^ 2) := by
    nlinarith only [hu, hv, hbal, sq_nonneg (β * a * u + b * v)]
  have hden : 0 < η ^ 2 * (1 - β) ^ 2 := by positivity
  calc
    g ^ 2 ≤ (2 * M ^ 2 * (u ^ 2 + v ^ 2)) / (η ^ 2 * (1 - β) ^ 2) :=
      (le_div_iff₀ hden).mpr (by nlinarith [hpoly])
    _ = _ := by ring

/-- A nonzero gradient and displacement satisfy the scalar assumptions,
arXiv:1904.03590v4, Algorithm 1, training extension. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    (1 : ℝ) * (-1 / 2) = (1 / 2) * 1 * 0 - 1 * (1 - 1 / 2) * 1 := by norm_num

/-- The true gradient energy is bounded by consecutive actual parameter
displacements. The constant uses only the initial loss gap, objective
smoothness and algorithm parameters. Source: arXiv:1904.03590v4,
Algorithm 1 and §4, fixed-objective/epsilon training extension. -/
theorem trainingRun_gradient_energy_bound (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1) (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x)
    (t : ℕ) :
    ‖gradient f (trainingRun η ε β β₂ f initial t).position‖ ^ 2 ≤
      2 * (ε + Real.sqrt (2 * L * (f initial - lower))) ^ 2 /
          (η ^ 2 * (1 - β) ^ 2) *
        (‖trainingVelocity η ε (trainingRun η ε β β₂ f initial t)‖ ^ 2 +
          ‖trainingVelocity η ε (trainingRun η ε β β₂ f initial (t + 1))‖ ^ 2) := by
  let s := trainingRun η ε β β₂ f initial t
  let next := trainingRun η ε β β₂ f initial (t + 1)
  let M := ε + Real.sqrt (2 * L * (f initial - lower))
  have hsum := Finset.sum_le_sum (s := Finset.univ) (fun i _ =>
    scalar_balance_gradient_bound η β (trainingDenominator ε s i)
      (trainingDenominator ε next i) (trainingVelocity η ε s i)
      (trainingVelocity η ε next i) (gradient f s.position i) M hη hβ hβ'
      (trainingDenominator_pos ε s hε i).le
      (trainingRun_denominator_bound η ε β β₂ L lower f initial hf
        hη hε hL hβ hβ' hβ₂ hβ₂' hstep hlower t i)
      (trainingDenominator_pos ε next hε i).le
      (trainingRun_denominator_bound η ε β β₂ L lower f initial hf
        hη hε hL hβ hβ' hβ₂ hβ₂' hstep hlower (t + 1) i)
      (trainingVelocity_balance η ε β β₂ f s hε i))
  have hnorm (w : TrainingSpace d) : (∑ i, (w i) ^ 2) = ‖w‖ ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq]
    simp only [Real.norm_eq_abs, sq_abs]
  rw [← Finset.mul_sum, Finset.sum_add_distrib, hnorm, hnorm, hnorm] at hsum
  exact hsum

/-- Full gradient-energy hypotheses hold for a nonconstant loss with
nonzero momentum, arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.AMSGrad
