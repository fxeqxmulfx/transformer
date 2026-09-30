/-
# AdaFisher — the effective coefficient of corrected momentum

arXiv:2405.16397v3, §3.3, Algorithm 1 and Table 1.
Bias correction turns constant raw momentum into a time-dependent
coefficient between zero and beta. The first coefficient is zero.
-/

import Transformer.AdaFisher.Section3_TrainingModels

noncomputable section

namespace Transformer.AdaFisher

/-- Effective coefficient after `t` previous batches, §3.3, Algorithm 1.
This is derived from the source's bias-corrected weighted moment. -/
def trainingBias (β : ℝ) (t : ℕ) : ℝ :=
  β * (1 - β ^ t) / (1 - β ^ (t + 1))

/-- Positive-time correction preserves a coefficient in `[0,beta]`.
Source: arXiv:2405.16397v3, §3.3, Algorithm 1 and Table 1. -/
theorem trainingBias_bounds (β : ℝ) (hβ : 0 ≤ β) (hβ' : β < 1) (t : ℕ) :
    0 ≤ trainingBias β t ∧ trainingBias β t ≤ β := by
  have hden := correctedMomentum_denominator_pos β hβ hβ' (t + 1) (by omega)
  have hp : β ^ t ≤ 1 := pow_le_one₀ hβ hβ'.le
  have hpow : β ^ (t + 1) ≤ β ^ t := by
    rw [pow_succ]
    exact mul_le_of_le_one_right (pow_nonneg hβ t) hβ'.le
  constructor
  · exact div_nonneg (mul_nonneg hβ (sub_nonneg.mpr hp)) hden.le
  · apply (div_le_iff₀ hden).mpr
    exact mul_le_mul_of_nonneg_left (by linarith : 1 - β ^ t ≤ 1 - β ^ (t + 1)) hβ

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 := by norm_num

/-- Exact scalar corrected-momentum recurrence, including the first step.
At zero time the actual raw buffer is zero. Source: arXiv:2405.16397v3,
Algorithm 1, corrected positive-time bias denominator. -/
theorem trainingBias_recurrence (β m g : ℝ) (t : ℕ)
    (hβ : 0 ≤ β) (hβ' : β < 1) (hzero : t = 0 → m = 0) :
    (β * m + (1 - β) * g) / (1 - β ^ (t + 1)) =
      g + trainingBias β t * (m / (1 - β ^ t) - g) := by
  have hnew := (correctedMomentum_denominator_pos β hβ hβ' (t + 1) (by omega)).ne'
  have hb : 1 - β ≠ 0 := (sub_pos.mpr hβ').ne'
  cases t with
  | zero => simp [hzero rfl, trainingBias, hb]
  | succ t =>
    have hold := (correctedMomentum_denominator_pos β hβ hβ' (t + 1) (by omega)).ne'
    dsimp only [trainingBias]
    field_simp [hold, hnew]
    rw [pow_succ β (t + 1)]
    ring

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    ((0 : ℕ) = 0 → (0 : ℝ) = 0) := by norm_num

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- The actual new corrected moment is the true current gradient plus
the effective coefficient times the current momentum error. Source:
arXiv:2405.16397v3, §3.3, Algorithm 1, deterministic specialization. -/
theorem trainingRun_moment_balance (η β γ δ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hβ : 0 ≤ β) (hβ' : β < 1) (t : ℕ) :
    trainingMoment β (trainingRun η β γ δ f factors initial (t + 1)) =
      gradient f (trainingPosition (trainingRun η β γ δ f factors initial t)) +
        trainingBias β t • trainingError β f (trainingRun η β γ δ f factors initial t) := by
  ext i
  simp only [trainingRun, trainingStep, optimizerStep, trainingMoment, trainingError,
    trainingRun_time, PiLp.toLp_apply, PiLp.add_apply, PiLp.smul_apply, PiLp.sub_apply,
    smul_eq_mul]
  apply trainingBias_recurrence β _ _ t hβ hβ'
  intro ht
  subst t
  rfl

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 := by norm_num

/-- The actual Fisher solve encodes corrected momentum exactly.
Source: arXiv:2405.16397v3, equation (4), Algorithm 1. -/
theorem trainingVelocity_balance (η β δ : ℝ) (s : OptimizerState a b)
    (hδ : 0 < δ) (i : Fin (a * b)) :
    trainingMetric δ s i * trainingVelocity η β δ s i = -η * trainingMoment β s i := by
  have hD : trainingMetric δ s i ≠ 0 :=
    (hδ.trans_le (trainingMetric_bounds δ s i).1).ne'
  change _ * (-η * _ / _) = _
  field_simp

/-- A concrete witness satisfies the preceding theorem's assumptions.
Source: arXiv:2405.16397v3, Algorithm 1 and §3.4, deterministic extension. -/
example : (0 : ℝ) < 1 / 1000 := by norm_num

end Transformer.AdaFisher
