/-
# AdaFisher — contracting error of the actual corrected momentum

arXiv:2405.16397v3, §3.4, assumption (i), deterministic specialization.
Gradient Lipschitz continuity is an objective property. No small-error,
alignment or bounded-iterate premise is imposed on the training run.
-/

import Transformer.AdaFisher.Section3_TrainingCoefficients

noncomputable section

namespace Transformer.AdaFisher

variable {a b : ℕ}

/-- Lipschitz continuity of the fixed loss's actual gradient, as in
arXiv:2405.16397v3, §3.4, Proposition 3.4 assumption (i). -/
def LipschitzGradient (f : TrainingSpace a b → ℝ) (L : ℝ) : Prop :=
  ∀ x y, ‖gradient f x - gradient f y‖ ≤ L * ‖x - y‖

/-- The quadratic has a genuine Lipschitz gradient and is nonconstant.
Source: arXiv:2405.16397v3, §3.4, deterministic convergence witness. -/
theorem energy_lipschitz : LipschitzGradient
    (Optimization.energy : TrainingSpace a b → ℝ) 1 := by
  intro x y
  rw [Optimization.energy_gradient]
  simp only [id_eq, one_mul]
  exact le_rfl

/-- The new corrected-momentum error is controlled by the previous error
and actual displacement. Source: arXiv:2405.16397v3, §3.3–3.4,
Algorithm 1, deterministic full-gradient specialization. -/
theorem trainingRun_error_bound [NeZero a] [NeZero b] (η β γ δ L : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hgrad : LipschitzGradient f L) (t : ℕ) :
    ‖trainingError β f (trainingRun η β γ δ f factors initial (t + 1))‖ ^ 2 ≤
      β * ‖trainingError β f (trainingRun η β γ δ f factors initial t)‖ ^ 2 +
        L ^ 2 / (1 - β) *
          ‖trainingVelocity η β δ (trainingRun η β γ δ f factors initial (t + 1))‖ ^ 2 := by
  let s := trainingRun η β γ δ f factors initial t
  let next := trainingRun η β γ δ f factors initial (t + 1)
  let e := trainingError β f s
  let v := trainingVelocity η β δ next
  have heq : trainingError β f next = trainingBias β t • e +
      (gradient f (trainingPosition s) - gradient f (trainingPosition next)) := by
    dsimp only [trainingError]
    rw [trainingRun_moment_balance η β γ δ f factors initial hβ hβ' t]
    dsimp only [e, trainingError]
    module
  have hpos : trainingPosition next - trainingPosition s = v := by
    have h := trainingStep_position η β γ δ f factors s
    change trainingPosition next = trainingPosition s + v at h
    rw [h]
    exact add_sub_cancel_left _ _
  have hLip := hgrad (trainingPosition s) (trainingPosition next)
  rw [norm_sub_rev (trainingPosition s) (trainingPosition next), hpos] at hLip
  obtain ⟨hb0, hb⟩ := trainingBias_bounds β hβ hβ' t
  have hn : ‖trainingError β f next‖ ≤ β * ‖e‖ + L * ‖v‖ := by
    rw [heq]
    have h := norm_add_le (trainingBias β t • e)
      (gradient f (trainingPosition s) - gradient f (trainingPosition next))
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hb0] at h
    have hm := mul_le_mul_of_nonneg_right hb (norm_nonneg e)
    linarith
  exact (pow_le_pow_left₀ (norm_nonneg _) hn 2).trans
    (training_error_young β L ‖e‖ ‖v‖ hβ hβ')

/-- The error-bound assumptions hold with a nonconstant objective and
nonzero momentum, arXiv:2405.16397v3, §3.4. -/
example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
    LipschitzGradient (Optimization.energy : TrainingSpace 2 2 → ℝ) 1 :=
  ⟨by norm_num, by norm_num, energy_lipschitz⟩

end Transformer.AdaFisher
