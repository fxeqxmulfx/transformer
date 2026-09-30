/-
# AdaFisher — the genuine gradient and the actual Fisher solve

arXiv:2405.16397v3, Algorithm 1, §3.4 and Appendix A.2.
Exact momentum and Fisher balances supply the inner-product identity
used by the loss decrease; sufficient alignment is not assumed.
-/

import Transformer.AdaFisher.Section3_TrainingBias

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.AdaFisher

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- Damping controls Euclidean energy in the actual computed metric.
Source: arXiv:2405.16397v3, Proposition 3.2, equation (4). -/
theorem training_metric_energy_lower (δ : ℝ) (s : OptimizerState a b)
    (v : TrainingSpace a b) :
    δ * ‖v‖ ^ 2 ≤ ∑ i, trainingMetric δ s i * (v i) ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  simp only [Real.norm_eq_abs, sq_abs]
  exact mul_le_mul_of_nonneg_right (trainingMetric_bounds δ s i).1 (sq_nonneg _)

/-- The actual gradient-displacement pairing is the negative metric
energy plus the explicitly controlled momentum-error term. Source:
arXiv:2405.16397v3, §3.4 and Appendix A.2, full Algorithm 1 specialization. -/
theorem trainingRun_inner_balance (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (t : ℕ) :
    let s := trainingRun η β γ δ f factors initial t
    let next := trainingRun η β γ δ f factors initial (t + 1)
    let v := trainingVelocity η β δ next
    ⟪gradient f (trainingPosition s), v⟫_ℝ =
      -(∑ i, trainingMetric δ next i * (v i) ^ 2) / η -
        trainingBias β t * ⟪trainingError β f s, v⟫_ℝ := by
  dsimp only
  let s := trainingRun η β γ δ f factors initial t
  let next := trainingRun η β γ δ f factors initial (t + 1)
  let v := trainingVelocity η β δ next
  have hm := trainingRun_moment_balance η β γ δ f factors initial hβ hβ' t
  have hpoint (i : Fin (a * b)) :
      gradient f (trainingPosition s) i * v i =
        -(trainingMetric δ next i * (v i) ^ 2) / η -
          trainingBias β t * (trainingError β f s i * v i) := by
    have hmi : trainingMoment β next i = gradient f (trainingPosition s) i +
        trainingBias β t * trainingError β f s i :=
      congrArg (fun w : TrainingSpace a b => w i) hm
    have hv := congrArg (fun z : ℝ => z * v i)
      (trainingVelocity_balance η β δ next hδ i)
    rw [hmi] at hv
    calc
      _ = (-(trainingMetric δ next i * (v i) ^ 2) -
          η * trainingBias β t * (trainingError β f s i * v i)) / η :=
        (eq_div_iff hη.ne').mpr (by nlinarith only [hv])
      _ = _ := by field_simp
  have hsum := Finset.sum_congr rfl (fun i (_ : i ∈ Finset.univ) => hpoint i)
  simpa only [PiLp.inner_apply, RCLike.inner_apply, conj_trivial,
    Finset.sum_sub_distrib, Finset.sum_div, neg_div, ← Finset.sum_neg_distrib,
    ← Finset.mul_sum, mul_comm] using hsum

/-- All balance hypotheses allow positive damping and nonzero momentum,
arXiv:2405.16397v3, Algorithm 1, deterministic specialization. -/
example : (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 := by norm_num

end Transformer.AdaFisher
