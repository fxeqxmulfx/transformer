import Transformer.DoubleDescent.Section2_EffectiveComplexity
import Mathlib.Tactic.Ring
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul

/-!
# Training time does not by itself order EMC

arXiv:1912.02292v1, Sections 1 and 6 assert that increasing training time
increases EMC. This needs a training-improvement assumption, supplied by
`EMC_mono_of_training_improvement`. Here ordinary gradient descent on a
one-parameter squared-loss model gives a counterexample without that premise.
All data points have the same target one; the model predicts its parameter.
-/

namespace Transformer.DoubleDescent

open MeasureTheory
open scoped ENNReal

/-- Sections 2 and 6, counterexample: squared loss on a single constant-target
regression input. The unit input has target one. -/
noncomputable def constantTargetLoss (weight : ℝ) (input : Unit) : ℝ≥0∞ :=
  match input with
  | () => ENNReal.ofReal ((weight - 1) ^ 2)

/-- Sections 2 and 6, counterexample: the constant predictor at a specified
iterate, on a dataset containing only the one regression input. -/
def constantTargetTrain (weight : ℝ) : TrainingProcedure Unit ℝ := fun _ _ => weight

/-- Sections 1 and 6, counterexample: an ordinary gradient-descent step for
the real squared loss `(weight-1)^2`, whose derivative is `2*(weight-1)`. -/
noncomputable def constantTargetStep (rate weight : ℝ) : ℝ :=
  weight - rate * (2 * (weight - 1))

/-- Sections 1 and 6, counterexample: the update above uses the actual
derivative of its squared loss, rather than a stipulated direction. -/
theorem constantTarget_loss_derivative (weight : ℝ) :
    HasDerivAt (fun w : ℝ => (w - 1) ^ 2) (2 * (weight - 1)) weight := by
  have h := (hasDerivAt_id weight).sub_const 1
  convert h.fun_mul h using 1 <;> simp [pow_two, two_mul]

/-- Sections 2 and 6: the empirical risk of this predictor is its actual
squared error whenever the sample is nonempty. -/
theorem constantTarget_empiricalRisk (weight : ℝ) {n : ℕ} (hn : n ≠ 0)
    (sample : Fin n → Unit) :
    empiricalRisk constantTargetLoss weight sample = ENNReal.ofReal ((weight - 1) ^ 2) := by
  have hi (i : Fin n) : constantTargetLoss weight (sample i) =
      ENNReal.ofReal ((weight - 1) ^ 2) := by
    cases sample i
    rfl
  unfold empiricalRisk
  simp_rw [hi]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_comm]
  exact ENNReal.mul_div_cancel_right (by exact_mod_cast hn) (by simp)

/-- Sections 2 and 6: the nonempty-sample premise has a one-point witness. -/
example : (1 : ℕ) ≠ 0 := by decide

/-- Sections 2 and 6: averaging over the genuine iid point-mass distribution
does not change the squared error; the empty-sample risk is zero. -/
theorem constantTarget_expectedRisk (weight : ℝ) (n : ℕ) :
    expectedTrainingRisk (Measure.dirac ()) constantTargetLoss (constantTargetTrain weight) n =
      if n = 0 then 0 else ENNReal.ofReal ((weight - 1) ^ 2) := by
  by_cases hn : n = 0
  · subst n
    simp [expectedTrainingRisk, empiricalRisk]
  · unfold expectedTrainingRisk constantTargetTrain
    simp_rw [constantTarget_empiricalRisk weight hn]
    simp [hn]

/-- Sections 1 and 6: an excessive but positive learning rate increases the
loss from `1/25` to `9/25` in one legitimate gradient-descent step. -/
theorem constantTarget_overshoot : constantTargetStep 2 (4 / 5) = 8 / 5 := by
  norm_num [constantTargetStep]

/-- Sections 1, 2, and 6: before that step, every sample size meets the
paper's heuristic tolerance `ε=0.1`, so EMC is infinite. -/
theorem constantTarget_initial_EMC :
    EMC (Measure.dirac ()) constantTargetLoss (constantTargetTrain (4 / 5)) (1 / 10) = ⊤ := by
  apply top_unique
  rw [← ENat.iSup_natCast]
  refine iSup_le fun n => le_effectiveComplexity _ _ ?_
  rw [constantTarget_expectedRisk]
  split_ifs <;>
    norm_num [ENNReal.ofReal_div_of_pos (by norm_num : (0 : ℝ) < 25)]

/-- Sections 1, 2, and 6: after that step, no positive sample size meets the
same tolerance, so EMC is zero. -/
theorem constantTarget_after_step_EMC :
    EMC (Measure.dirac ()) constantTargetLoss
      (constantTargetTrain (constantTargetStep 2 (4 / 5))) (1 / 10) = 0 := by
  rw [constantTarget_overshoot]
  apply effectiveComplexity_eq_max _ _ 0
  · rw [constantTarget_expectedRisk]
    simp
  · intro n hn
    rw [constantTarget_expectedRisk] at hn
    by_cases hz : n = 0
    · simp [hz]
    · simp only [hz, ite_false] at hn
      rw [ENNReal.ofReal_le_iff_le_toReal (by norm_num)] at hn
      norm_num [ENNReal.toReal_div] at hn

/-- Sections 1 and 6: the assertion that longer training increases EMC is
false without a risk-improvement assumption. This example uses the same
architecture, data distribution, and tolerance before and after one step. -/
theorem training_time_EMC_counterexample :
    ¬ EMC (Measure.dirac ()) constantTargetLoss (constantTargetTrain (4 / 5)) (1 / 10) ≤
      EMC (Measure.dirac ()) constantTargetLoss
        (constantTargetTrain (constantTargetStep 2 (4 / 5))) (1 / 10) := by
  rw [constantTarget_initial_EMC, constantTarget_after_step_EMC]
  simp

end Transformer.DoubleDescent
