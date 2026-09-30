/-
# AdaFisher — gradient energy from actual displacement and moment error

arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension.
An upper bound on the computed min-max Fisher prevents a vanishing
parameter step from hiding a nonzero corrected moment.
-/

import Transformer.AdaFisher.Section3_TrainingBias

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- The actual corrected moment is bounded by its actual displacement.
The metric upper bound comes from normalization, not a trajectory premise.
Source: arXiv:2405.16397v3, Proposition 3.2 and Algorithm 1. -/
theorem trainingMoment_energy_le (η β δ : ℝ) (s : OptimizerState a b)
    (hη : 0 < η) (hδ : 0 < δ) :
    ‖trainingMoment β s‖ ^ 2 ≤
      (1 + δ) ^ 2 / η ^ 2 * ‖trainingVelocity η β δ s‖ ^ 2 := by
  rw [EuclideanSpace.norm_sq_eq, EuclideanSpace.norm_sq_eq, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  simp only [Real.norm_eq_abs, sq_abs]
  have hb := trainingMetric_bounds δ s i
  have hD : 0 ≤ trainingMetric δ s i := hδ.le.trans hb.1
  have hsq : (trainingMetric δ s i) ^ 2 ≤ (1 + δ) ^ 2 := by nlinarith
  have hbal := congrArg (fun z : ℝ => z ^ 2) (trainingVelocity_balance η β δ s hδ i)
  have hηsq : η ^ 2 ≠ 0 := pow_ne_zero 2 hη.ne'
  calc
    _ = ((trainingMetric δ s i) ^ 2 * (trainingVelocity η β δ s i) ^ 2) / η ^ 2 :=
      (eq_div_iff hηsq).mpr (by nlinarith only [hbal])
    _ ≤ ((1 + δ) ^ 2 * (trainingVelocity η β δ s i) ^ 2) / η ^ 2 :=
      div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hsq (sq_nonneg _)) (sq_nonneg η)
    _ = _ := by ring

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) < 1 := by norm_num

/-- The true current gradient is controlled by the next actual step and
the current corrected-momentum error. Source: arXiv:2405.16397v3,
§3.3–3.4, full Algorithm 1 deterministic specialization. -/
theorem trainingRun_gradient_energy_bound (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (t : ℕ) :
    ‖gradient f (trainingPosition (trainingRun η β γ δ f factors initial t))‖ ^ 2 ≤
      2 * ((1 + δ) ^ 2 / η ^ 2 *
          ‖trainingVelocity η β δ (trainingRun η β γ δ f factors initial (t + 1))‖ ^ 2 +
        β ^ 2 * ‖trainingError β f (trainingRun η β γ δ f factors initial t)‖ ^ 2) := by
  let s := trainingRun η β γ δ f factors initial t
  let next := trainingRun η β γ δ f factors initial (t + 1)
  let e := trainingError β f s
  let p := trainingMoment β next
  have hm := trainingRun_moment_balance η β γ δ f factors initial hβ hβ' t
  have heq : gradient f (trainingPosition s) = p - trainingBias β t • e := by
    change p = gradient f (trainingPosition s) + trainingBias β t • e at hm
    rw [hm]
    module
  obtain ⟨hb0, hb⟩ := trainingBias_bounds β hβ hβ' t
  have hn : ‖gradient f (trainingPosition s)‖ ≤ ‖p‖ + β * ‖e‖ := by
    rw [heq]
    have h := norm_sub_le p (trainingBias β t • e)
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hb0] at h
    have h' := mul_le_mul_of_nonneg_right hb (norm_nonneg e)
    linarith
  have hs := pow_le_pow_left₀ (norm_nonneg _) hn 2
  have hquad : (‖p‖ + β * ‖e‖) ^ 2 ≤ 2 * (‖p‖ ^ 2 + β ^ 2 * ‖e‖ ^ 2) := by
    nlinarith only [sq_nonneg (‖p‖ - β * ‖e‖)]
  have hP := trainingMoment_energy_le η β δ next hη hδ
  nlinarith only [hs, hquad, hP]

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 := by norm_num

end Transformer.AdaFisher
