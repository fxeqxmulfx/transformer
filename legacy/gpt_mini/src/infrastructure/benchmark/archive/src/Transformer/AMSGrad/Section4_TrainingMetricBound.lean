/-
# AMSGrad — boundedness of the actual adaptive metric

Training extension of arXiv:1904.03590v4, Algorithm 1 and §4.
The metric upper bound follows from the objective's initial loss gap;
it is not a free preconditioner or bounded-iterate hypothesis.
-/

import Transformer.AMSGrad.Section4_TrainingGradientBound

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- Positive second-moment EMA and its maximum inherit a squared-gradient
bound along the actual run. Source: arXiv:1904.03590v4, Algorithm 1,
fixed-objective extension. This helper's gradient premise is subsequently
derived from smoothness and the lower loss bound. -/
theorem trainingRun_moments_bound (η ε β β₂ C : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hC : 0 ≤ C) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1)
    (hgrad : ∀ t i, (gradient f (trainingRun η ε β β₂ f initial t).position i) ^ 2 ≤ C)
    (t : ℕ) (i : Fin d) :
    0 ≤ (trainingRun η ε β β₂ f initial t).second i ∧
      0 ≤ (trainingRun η ε β β₂ f initial t).maximum i ∧
      (trainingRun η ε β β₂ f initial t).second i ≤ C ∧
      (trainingRun η ε β β₂ f initial t).maximum i ≤ C := by
  induction t with
  | zero => exact ⟨le_rfl, le_rfl, hC, hC⟩
  | succ t ih =>
    let s := trainingRun η ε β β₂ f initial t
    let g := gradient f s.position i
    have hv0 : 0 ≤ β₂ * s.second i + (1 - β₂) * g ^ 2 :=
      add_nonneg (mul_nonneg hβ₂ ih.1) (mul_nonneg (sub_nonneg.mpr hβ₂') (sq_nonneg g))
    have hv : β₂ * s.second i + (1 - β₂) * g ^ 2 ≤ C := by
      have h1 := mul_le_mul_of_nonneg_left ih.2.2.1 hβ₂
      have h2 := mul_le_mul_of_nonneg_left (hgrad t i) (sub_nonneg.mpr hβ₂')
      dsimp only [s, g] at *
      nlinarith
    exact ⟨hv0, le_trans ih.2.1 (le_max_left _ _), hv, max_le ih.2.2.2 hv⟩

/-- Moment-bound hypotheses are satisfiable with a nonconstant quadratic,
nonzero initial gradient and nonzero momentum. Source: arXiv:1904.03590v4,
Algorithm 1 and §4, training extension. -/
example :
    let initial : TrainingSpace 1 := WithLp.toLp 2 (fun _ => 1)
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
      (∀ t i, (gradient energy
        (trainingRun (1 / 2) 1 (9 / 10) (1 / 2) energy initial t).position i) ^ 2 ≤ 1) := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro t i
  have h := trainingRun_gradient_coordinate_bound (1 / 2) 1 (9 / 10) (1 / 2) 1 0
    energy (WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) energy_models.1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (fun x => div_nonneg (sq_nonneg _) (by norm_num)) t i
  have hC : 2 * (1 : ℝ) *
      (energy (WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) - 0) = 1 := by
    norm_num [energy, EuclideanSpace.norm_sq_eq]
  rwa [hC] at h

/-- The actual denominator is uniformly bounded above by epsilon plus
the square root of the initial gradient-energy bound. All metric
boundedness is proved from the actual EMA recurrence and objective.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
theorem trainingRun_denominator_bound (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1) (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x)
    (t : ℕ) (i : Fin d) :
    trainingDenominator ε (trainingRun η ε β β₂ f initial t) i ≤
      ε + Real.sqrt (2 * L * (f initial - lower)) := by
  have hC : 0 ≤ 2 * L * (f initial - lower) :=
    mul_nonneg (by positivity) (sub_nonneg.mpr (hlower initial))
  have h := (trainingRun_moments_bound η ε β β₂ _ f initial hC hβ₂ hβ₂'
    (fun t i => trainingRun_gradient_coordinate_bound η ε β β₂ L lower f initial
      hf hη hε hL hβ hβ' hstep hlower t i) t i).2.2.2
  exact add_le_add le_rfl (Real.sqrt_le_sqrt h)

/-- All metric-bound assumptions hold with a nonconstant loss and nonzero
momentum, arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example : SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
    (∀ x : TrainingSpace 1, 0 ≤ energy x) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩

end Transformer.AMSGrad
