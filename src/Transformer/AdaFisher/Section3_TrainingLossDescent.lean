/-
# AdaFisher — actual loss descent up to the controlled momentum error

arXiv:2405.16397v3, §3.4 and Appendix A.2, full-gradient specialization.
The KF metric is allowed to change arbitrarily within its derived bounds.
The explicit error term replaces an assumed gradient-alignment premise.
-/

import Transformer.AdaFisher.Section3_TrainingCoefficients
import Transformer.AdaFisher.Section3_TrainingInner

open scoped InnerProductSpace

noncomputable section

namespace Transformer.AdaFisher

open Optimization

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- A complete actual AdaFisher update satisfies a loss decrease estimate
with its current momentum error. Source: arXiv:2405.16397v3, §3.4 and
Appendix A.2, explicit constant-step/full-gradient extension of Algorithm 1.
No Fisher/Hessian identification or monotone-metric assumption is made. -/
theorem trainingRun_loss_descent (η β γ δ L : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1) (hδ : 0 < δ) (hL : 0 < L)
    (hstep : 2 * L * η ≤ δ * (1 - β)) (t : ℕ) :
    f (trainingPosition (trainingRun η β γ δ f factors initial (t + 1))) ≤
      f (trainingPosition (trainingRun η β γ δ f factors initial t)) -
        δ / (2 * η) *
          ‖trainingVelocity η β δ (trainingRun η β γ δ f factors initial (t + 1))‖ ^ 2 +
        η * β ^ 2 / δ * ‖trainingError β f (trainingRun η β γ δ f factors initial t)‖ ^ 2 := by
  let s := trainingRun η β γ δ f factors initial t
  let next := trainingRun η β γ δ f factors initial (t + 1)
  let e := trainingError β f s
  let v := trainingVelocity η β δ next
  have hpos : trainingPosition next - trainingPosition s = v := by
    have h := trainingStep_position η β γ δ f factors s
    change trainingPosition next = trainingPosition s + v at h
    rw [h]
    exact add_sub_cancel_left _ _
  have hsmooth := hf.2 (trainingPosition s) (trainingPosition next)
  rw [hpos] at hsmooth
  have hinner := trainingRun_inner_balance η β γ δ f factors initial hη hβ hβ' hδ t
  change ⟪gradient f (trainingPosition s), v⟫_ℝ =
    -(∑ i, trainingMetric δ next i * (v i) ^ 2) / η - trainingBias β t * ⟪e, v⟫_ℝ at hinner
  rw [neg_div] at hinner
  obtain ⟨hb0, hb⟩ := trainingBias_bounds β hβ hβ' t
  have hcross : -trainingBias β t * ⟪e, v⟫_ℝ ≤ β * ‖e‖ * ‖v‖ := by
    have hc := real_inner_le_norm e (-v)
    rw [inner_neg_right, norm_neg] at hc
    have h1 := mul_le_mul_of_nonneg_left hc hb0
    have h2 := mul_le_mul_of_nonneg_right hb (mul_nonneg (norm_nonneg e) (norm_nonneg v))
    nlinarith only [h1, h2]
  have hYoung := training_young η δ β ‖e‖ ‖v‖ hη hδ
  have hcoef := (training_coefficient_bounds η β δ L hη hβ hβ' hδ hL hstep).1
  have hsmoothterm := mul_le_mul_of_nonneg_right hcoef (sq_nonneg ‖v‖)
  have hmetric := div_le_div_of_nonneg_right
    (training_metric_energy_lower δ next v) hη.le
  have heq : δ * ‖v‖ ^ 2 / η = 4 * (δ / (4 * η) * ‖v‖ ^ 2) := by ring
  rw [heq] at hmetric
  have heq' : δ / (2 * η) * ‖v‖ ^ 2 = 2 * (δ / (4 * η) * ‖v‖ ^ 2) := by ring
  change f (trainingPosition next) ≤ f (trainingPosition s) -
    δ / (2 * η) * ‖v‖ ^ 2 + η * β ^ 2 / δ * ‖e‖ ^ 2
  rw [heq']
  linarith

/-- A nonconstant quadratic and nonzero momentum satisfy every loss
descent assumption, arXiv:2405.16397v3, §3.4, deterministic extension. -/
example : SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
    (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) := by
  exact ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num⟩

end Transformer.AdaFisher
