import Transformer.Grokking.CircuitEfficiency.SectionC_TableLoss

/-!
# Actual finite-class training confidence and CE order

Source: Varma et al., arXiv:2309.02390v1, section 3's distinction
between correct decisions and confidence, appendix C's common Gen/Mem
train logits and full train-loss denominator. Separate readout facts
from claims about native optimization or learned transformer dynamics.

The target probability is computed from the actual exponential sum
over remaining+2 classes. Its formula contains one active target and
remaining+1 zero-logit competitors. At every finite score its value
lies strictly between zero and one, and at zero score it is uniform.
It increases strictly with the actual total training score.

The original full cross-entropy equals minus the logarithm of this
probability and decreases strictly with score. Training confidence
above uniform, and train CE below uniform-class loss, both correspond
exactly to a positive total training score. Neither distinguishes the
underlying Gen/Mem allocation: the two tables agree on training labels.

These pointwise measurement laws contain no convergent trajectory,
optimizer update or future reference. They support a separate bridge
from original weak-decay recurrent scores to confidence/CE observations.
Fixed circuit tables differ from learned stochastic/floating-point
GPTMini heads; no grokking event is inferred from a training measurement.
The source's coupled penalty is kept outside the actual CE throughout.
-/

namespace Transformer.Grokking.CircuitEfficiency

open scoped BigOperators

/-- Target softmax probability from the actual complete train logits.
Source: appendix C train exponential denominator; every class is
included and the score is numerical, without a correctness predicate. -/
noncomputable def tableTrainTargetProbability (remaining : ℕ) (score : ℝ) : ℝ :=
  Real.exp (trainTableLogits remaining score 0 0) /
    ∑ k, Real.exp (trainTableLogits remaining score 0 k)

/-- Actual target probability has the full finite-class train formula.
Source: appendix C's remaining+1 competing exponentials, computed
from the complete forward rather than assigned as a surrogate. -/
theorem table_train_target_probability_formula (remaining : ℕ) (score : ℝ) :
    tableTrainTargetProbability remaining score =
      Real.exp score / (Real.exp score + (remaining : ℝ) + 1) := by
  unfold tableTrainTargetProbability
  rw [train_table_exp_sum]
  simp only [trainTableLogits, ite_true, add_zero]

/-- The complete two-circuit train softmax uses only their total
score. Source: appendix C's common Gen/Mem training labels and full
exponential sum; this identity preserves both circuit arguments in
the readout and exposes training-only allocation indistinguishability. -/
theorem table_train_target_probability_allocation (remaining : ℕ) (genScore memScore : ℝ) :
    Real.exp (trainTableLogits remaining genScore memScore 0) /
      (∑ k, Real.exp (trainTableLogits remaining genScore memScore k)) =
        tableTrainTargetProbability remaining (genScore + memScore) := by
  rw [table_train_target_probability_formula, train_table_exp_sum]
  simp only [trainTableLogits, ite_true]

/-- Every finite actual train target probability is strictly positive.
Source: appendix C full finite exponential denominator; no score sign
or perfect decision is assumed. -/
theorem table_train_target_probability_positive (remaining : ℕ) (score : ℝ) :
    0 < tableTrainTargetProbability remaining score := by
  rw [table_train_target_probability_formula]
  exact div_pos (Real.exp_pos score) (by positivity)

/-- A finite train score never gives exact unit target probability.
Source: appendix C's nonempty competing-class denominator; increasing
confidence towards one remains distinct from reaching a finite maximum. -/
theorem table_train_target_probability_lt_one (remaining : ℕ) (score : ℝ) :
    tableTrainTargetProbability remaining score < 1 := by
  rw [table_train_target_probability_formula]
  apply (div_lt_one (by positivity)).mpr
  have hn : (0 : ℝ) ≤ (remaining : ℝ) := by positivity
  linarith only [hn]

/-- Zero current training score gives the uniform finite-class target.
Source: appendix C zero initial product logits and full class count;
remaining+2 is kept rather than silently replacing the task by binary CE. -/
theorem table_train_target_probability_zero (remaining : ℕ) :
    tableTrainTargetProbability remaining 0 = 1 / ((remaining : ℝ) + 2) := by
  rw [table_train_target_probability_formula, Real.exp_zero]
  congr 1
  ring

/-- Actual train target confidence increases strictly with score.
Source: section 3's confidence scaling and appendix C train forward;
the remaining+1 competing classes make the finite-point increase strict. -/
theorem table_train_target_probability_strictMono (remaining : ℕ) :
    StrictMono (tableTrainTargetProbability remaining) := by
  intro x y hxy
  rw [table_train_target_probability_formula, table_train_target_probability_formula]
  apply (div_lt_div_iff₀ (by positivity) (by positivity)).mpr
  have hn : 0 < (remaining : ℝ) + 1 := by positivity
  have he := Real.exp_lt_exp.mpr hxy
  have hh := mul_pos hn (show 0 < Real.exp y - Real.exp x by linarith only [he])
  nlinarith only [hh]

/-- Full ordinary CE is minus log of the actual train target probability.
Source: appendix C train CE, with the penalty excluded as in TableLoss;
neither correctness nor an optimizer progress condition enters the loss. -/
theorem table_train_ce_probability (remaining : ℕ) (score : ℝ) :
    tableTrainCE remaining score = -Real.log (tableTrainTargetProbability remaining score) := by
  rw [table_train_ce_formula, table_train_target_probability_formula,
    Real.log_div (ne_of_gt (Real.exp_pos score)) (by positivity), Real.log_exp]
  ring

/-- Actual train CE strictly decreases with its finite total score.
Source: section 3's confidence incentive and appendix C common train
logits; this is a readout law, not monotonicity of native training clocks. -/
theorem table_train_ce_strictAnti (remaining : ℕ) : StrictAnti (tableTrainCE remaining) := by
  intro x y hxy
  have hh := Real.log_lt_log (table_train_target_probability_positive remaining x)
    (table_train_target_probability_strictMono remaining hxy)
  rw [table_train_ce_probability, table_train_ce_probability]
  linarith only [hh]

/-- Above-uniform actual train target confidence is exactly positive
total train score. Source: appendix C common labels and complete finite
denominator; this condition does not determine which circuit is active. -/
theorem table_train_target_probability_above_uniform_iff (remaining : ℕ) (score : ℝ) :
    1 / ((remaining : ℝ) + 2) < tableTrainTargetProbability remaining score ↔ 0 < score := by
  rw [← table_train_target_probability_zero remaining]
  constructor
  · intro hh
    by_contra hscore
    have hle := (table_train_target_probability_strictMono remaining).monotone (le_of_not_gt hscore)
    exact (not_le_of_gt hh) hle
  · intro hscore
    exact table_train_target_probability_strictMono remaining hscore

/-- Below-uniform-class train CE is exactly positive total train score.
Source: section 3's confidence incentive and appendix C train loss;
test correctness and learned-rule formation remain separate measurements. -/
theorem table_train_ce_below_uniform_iff (remaining : ℕ) (score : ℝ) :
    tableTrainCE remaining score < Real.log ((remaining : ℝ) + 2) ↔ 0 < score := by
  rw [← table_train_ce_initial remaining]
  constructor
  · intro hh
    by_contra hscore
    have hle := (table_train_target_probability_strictMono remaining).monotone (le_of_not_gt hscore)
    have hlog := Real.log_le_log (table_train_target_probability_positive remaining score) hle
    rw [table_train_ce_probability, table_train_ce_probability] at hh
    linarith only [hh, hlog]
  · intro hscore
    exact table_train_ce_strictAnti remaining hscore

end Transformer.Grokking.CircuitEfficiency
