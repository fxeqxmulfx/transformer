import Transformer.Grokking.CircuitEfficiency.SectionC_GainSmallDecayScores
import Transformer.Grokking.CircuitEfficiency.SectionC_TrainProbability

/-!
# Joint original weak-decay training confidence and CE excursions

Sources: Varma et al., arXiv:2309.02390v1, section 3 confidence and
appendix C full train logits/CE; initialized original retained score
recurrence at lab commit 722638e and complete confidence laws at 6d7b5ec.

Measure original train logits from both actual gained physical products,
target probability from their complete finite-class exponential sum,
and ordinary CE from the same actual logits. Prove their identities
with the existing total-score measurements before making a path claim.

Every positive Gen seed on the small-decay native path has a fixed
above-uniform confidence level and a fixed below-uniform-class CE level
jointly attained beyond every prescribed clock. The actual generated
training-score excursions supply the same clocks for both observations.
Neither training probability nor actual train CE can have its uniform
limit. Parameter/input convergence, a future level and successful
reference are not supplied; both buffers and clocks stay original.

At every successor clock training confidence is above uniform and CE
below uniform-class loss, even when Gen is initially fully absent.
The two fixed tables agree on train labels. These observations therefore
do not certify rule discovery or eventual true held-out selection.

Recurrent levels do not imply stable positive confidence, a positive
uniform tail gap, convergent weights or an observed transition time.
Fixed gains/tables/plain CE and uniform native decay differ from the
source's coupled norm-cost GD and learned stochastic/floating-point
GPTMini. Actual native constants are unchanged throughout each path.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Filter Transformer.Grokking.NaiveLoss
open scoped BigOperators

/-- Full original gained training logits at a native clock. Sources:
appendix C common labels and actual small-decay path at 722638e;
each circuit uses both current physical factors in its true product. -/
noncomputable def smallDecayGainTrainLogits (remaining : ℕ) (seed : ℝ) (n : ℕ) : Fin (remaining + 2) → ℝ :=
  trainTableLogits remaining
    (physicalCircuitScore 3 (smallDecayGainPath remaining seed n 0).parameter (smallDecayGainPath remaining seed n 1).parameter)
    (physicalCircuitScore 2 (smallDecayGainPath remaining seed n 2).parameter (smallDecayGainPath remaining seed n 3).parameter)

/-- Original full finite-class training target softmax probability.
Sources: appendix C complete exponential denominator and original
train forward at 722638e; no correctness predicate enters the observer. -/
noncomputable def smallDecayGainTrainProbability (remaining : ℕ) (seed : ℝ) (n : ℕ) : ℝ :=
  Real.exp (smallDecayGainTrainLogits remaining seed n 0) /
    ∑ k, Real.exp (smallDecayGainTrainLogits remaining seed n k)

/-- Ordinary full CE on the original actual training logits. Sources:
appendix C train CE and original forward at 722638e; the source's
coupled penalty is not inserted into this confidence measurement. -/
noncomputable def smallDecayGainTrainCE (remaining : ℕ) (seed : ℝ) (n : ℕ) : ℝ :=
  crossEntropy (smallDecayGainTrainLogits remaining seed n) 0

/-- Original full training probability agrees with the proved actual
total-score confidence formula. Sources: appendix C train logits and
complete allocation identity at 6d7b5ec, using both evolved products. -/
theorem small_decay_gain_train_probability_eq (remaining : ℕ) (seed : ℝ) (n : ℕ) :
    smallDecayGainTrainProbability remaining seed n =
      tableTrainTargetProbability remaining (smallDecayGainTrainScore remaining seed n) := by
  exact table_train_target_probability_allocation remaining _ _

/-- Original full training CE agrees with the complete total-score
CE. Sources: appendix C train forward and TableLoss; equality of actual
logit vectors precedes the measurement identity, without task-success data. -/
theorem small_decay_gain_train_ce_eq (remaining : ℕ) (seed : ℝ) (n : ℕ) :
    smallDecayGainTrainCE remaining seed n = tableTrainCE remaining (smallDecayGainTrainScore remaining seed n) := by
  have hz : smallDecayGainTrainLogits remaining seed n =
      trainTableLogits remaining (smallDecayGainTrainScore remaining seed n) 0 := by
    funext k
    unfold smallDecayGainTrainLogits smallDecayGainTrainScore gainNativeTotalScore trainTableLogits
    simp only [add_zero]
  change crossEntropy (smallDecayGainTrainLogits remaining seed n) 0 =
    crossEntropy (trainTableLogits remaining (smallDecayGainTrainScore remaining seed n) 0) 0
  rw [hz]

/-- Original double-zero initialization has uniform confidence and
uniform-class CE for every numerical partner seed. Sources: section 3
zero first factors and appendix C train logits; no optimizer update or
training decision is inserted into either initial observer. -/
theorem small_decay_gain_initial_training_observers (remaining : ℕ) (seed : ℝ) :
    smallDecayGainTrainProbability remaining seed 0 = 1 / ((remaining : ℝ) + 2) ∧
      smallDecayGainTrainCE remaining seed 0 = Real.log ((remaining : ℝ) + 2) := by
  have hscore : smallDecayGainTrainScore remaining seed 0 = 0 := by
    change gainNativeTotalScore 3 2 (seededNativeSubweights ((0, seed), (0, 1))) = 0
    norm_num [gainNativeTotalScore, physicalCircuitScore, seededNativeSubweights,
      Transformer.Grokking.AdamW.seededScalarState]
  rw [small_decay_gain_train_probability_eq, small_decay_gain_train_ce_eq, hscore]
  exact ⟨table_train_target_probability_zero remaining, table_train_ce_initial remaining⟩

/-- Joint actual above-uniform confidence and below-uniform CE levels
occur arbitrarily late on the initialized small-decay native path.
Sources: section 3 confidence, appendix C logits/CE and generated
score excursions at 722638e; the two observations use the same clocks. -/
theorem small_decay_gain_joint_training_returns (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ∃ confidence loss : ℝ, 1 / ((remaining : ℝ) + 2) < confidence ∧
      loss < Real.log ((remaining : ℝ) + 2) ∧ ∀ start : ℕ, ∃ n : ℕ, start ≤ n ∧
        confidence ≤ smallDecayGainTrainProbability remaining seed n ∧ smallDecayGainTrainCE remaining seed n ≤ loss := by
  obtain ⟨level, hlevel, hreturn⟩ := small_decay_gain_train_score_returns remaining seed hseed
  refine ⟨tableTrainTargetProbability remaining level, tableTrainCE remaining level,
    (table_train_target_probability_above_uniform_iff remaining level).mpr hlevel,
    (table_train_ce_below_uniform_iff remaining level).mpr hlevel, ?_⟩
  intro start
  obtain ⟨n, hn, hscore⟩ := hreturn start
  refine ⟨n, hn, ?_, ?_⟩
  · rw [small_decay_gain_train_probability_eq]
    exact (table_train_target_probability_strictMono remaining).monotone hscore
  · rw [small_decay_gain_train_ce_eq]
    exact (table_train_ce_strictAnti remaining).antitone hscore

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- Original training target probability cannot tend to uniform on
this small-decay initialized path. Sources: section 3 confidence and
actual recurrent score at 722638e; no finite weight limit is assumed. -/
theorem small_decay_gain_train_probability_not_uniform (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ¬Tendsto (smallDecayGainTrainProbability remaining seed) atTop (nhds (1 / ((remaining : ℝ) + 2))) := by
  intro huniform
  obtain ⟨confidence, loss, hconfidence, _, hreturn⟩ := small_decay_gain_joint_training_returns remaining seed hseed
  obtain ⟨start, htail⟩ := eventually_atTop.mp (huniform.eventually_lt_const hconfidence)
  obtain ⟨n, hn, hprobability, _⟩ := hreturn start
  exact (not_le_of_gt (htail n hn)) hprobability

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- Original full training CE cannot tend to uniform-class loss
under these same initial data. Sources: appendix C train CE and
generated recurrent score at 722638e; stable confidence remains open. -/
theorem small_decay_gain_train_ce_not_uniform (remaining : ℕ) (seed : ℝ) (hseed : 0 < seed) :
    ¬Tendsto (smallDecayGainTrainCE remaining seed) atTop (nhds (Real.log ((remaining : ℝ) + 2))) := by
  intro huniform
  obtain ⟨confidence, loss, _, hloss, hreturn⟩ := small_decay_gain_joint_training_returns remaining seed hseed
  obtain ⟨start, htail⟩ := eventually_atTop.mp (huniform.eventually_const_lt hloss)
  obtain ⟨n, hn, _, hce⟩ := hreturn start
  exact (not_le_of_gt (htail n hn)) hce

example : (0 : ℝ) < 1 / 200 := by norm_num

/-- Actual train confidence is above uniform and CE below uniform
at every successor, even with zero Gen seed. Sources: appendix C's
common training labels and original successor decisions at 722638e;
the untrained zero-clock ties are excluded. This is not a test guarantee. -/
theorem small_decay_gain_train_confident_successors (remaining : ℕ) (seed : ℝ) (n : ℕ) (hseed : 0 ≤ seed) :
    1 / ((remaining : ℝ) + 2) < smallDecayGainTrainProbability remaining seed (n + 1) ∧
      smallDecayGainTrainCE remaining seed (n + 1) < Real.log ((remaining : ℝ) + 2) := by
  have hpositive := (train_table_strict_correct_iff remaining _ _).mp
    (small_decay_gain_train_correct_successors remaining seed n hseed)
  change 0 < smallDecayGainTrainScore remaining seed (n + 1) at hpositive
  rw [small_decay_gain_train_probability_eq, small_decay_gain_train_ce_eq]
  exact ⟨(table_train_target_probability_above_uniform_iff remaining _).mpr hpositive,
    (table_train_ce_below_uniform_iff remaining _).mpr hpositive⟩

example : (0 : ℝ) ≤ 0 := by norm_num

end Transformer.Grokking.CircuitEfficiency
