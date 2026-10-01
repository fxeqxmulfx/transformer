import Transformer.DoubleDescent.Section2_TrainingTime

/-!
# What the generalized double-descent hypothesis assumes

arXiv:1912.02292v1, Section 2, Hypothesis 1. "Natural", "perturbation",
and "sufficiently" are explicitly left undefined by the authors. The
unrestricted claim that EMC alone orders population risk is false, even
arbitrarily far into the over-parameterized regime. The following actual
learning procedures, on the same iid probability distribution, refute that
interpretation. They do not refute a future precisely restricted conjecture
about natural procedures. No unproved universal risk guarantee is assumed.
-/

namespace Transformer.DoubleDescent

open MeasureTheory
open scoped ENNReal

/-- Section 2, Hypothesis 1, counterexample: a scalar learner uses its
specified good weight through a sample-size cutoff, and the weight zero
after the cutoff. The input distribution and model architecture stay fixed. -/
def cutoffTrain (cutoff : ℕ) (goodWeight : ℝ) : TrainingProcedure Unit ℝ :=
  fun n _ => if n ≤ cutoff then goodWeight else 0

/-- Section 2, Definition 1: this learner has the ordinary iid training risk
of the selected constant predictor. -/
theorem cutoff_expectedTrainingRisk (cutoff : ℕ) (weight : ℝ) (n : ℕ) :
    expectedTrainingRisk (Measure.dirac ()) constantTargetLoss (cutoffTrain cutoff weight) n =
      if n = 0 then 0 else
        if n ≤ cutoff then ENNReal.ofReal ((weight - 1) ^ 2) else 1 := by
  by_cases hn : n ≤ cutoff
  · have he : ∀ sample, cutoffTrain cutoff weight n sample =
        constantTargetTrain weight n sample := by simp [cutoffTrain, constantTargetTrain, hn]
    unfold expectedTrainingRisk
    simp_rw [he]
    rw [← expectedTrainingRisk, constantTarget_expectedRisk]
    simp [hn]
  · have he : ∀ sample, cutoffTrain cutoff weight n sample =
        constantTargetTrain 0 n sample := by simp [cutoffTrain, constantTargetTrain, hn]
    unfold expectedTrainingRisk
    simp_rw [he]
    rw [← expectedTrainingRisk, constantTarget_expectedRisk]
    simp [hn]

/-- Section 2, Definition 1: the cutoff is exactly EMC when the good weight
fits within tolerance and the fallback weight fails it. -/
theorem cutoff_EMC (cutoff : ℕ) (weight : ℝ) (ε : ℝ≥0∞)
    (hε : ε < 1) (hw : ENNReal.ofReal ((weight - 1) ^ 2) ≤ ε) :
    EMC (Measure.dirac ()) constantTargetLoss (cutoffTrain cutoff weight) ε =
      (cutoff : ℕ∞) := by
  apply effectiveComplexity_eq_max
  · rw [cutoff_expectedTrainingRisk]
    split_ifs <;> simp_all
  · intro n hn
    rw [cutoff_expectedTrainingRisk] at hn
    by_cases hz : n = 0
    · simp [hz]
    · by_cases hc : n ≤ cutoff
      · exact hc
      · simp only [hz, ite_false, hc] at hn
        exact False.elim ((not_le_of_gt hε) hn)

/-- Section 2: the cutoff theorem's hypotheses are jointly satisfiable
at the paper's heuristic tolerance `ε=0.1`. -/
example : (1 / 10 : ℝ≥0∞) < 1 ∧ ENNReal.ofReal (((1 : ℝ) - 1) ^ 2) ≤ (1 / 10) := by
  norm_num

/-- Section 2, Hypothesis 1: the exact expected population test risk of the
cutoff learner, averaging the independent fresh example and training draw. -/
theorem cutoff_expectedTestRisk (cutoff : ℕ) (weight : ℝ) (n : ℕ) :
    expectedTestRisk (Measure.dirac ()) constantTargetLoss (cutoffTrain cutoff weight) n =
      if n ≤ cutoff then ENNReal.ofReal ((weight - 1) ^ 2) else 1 := by
  unfold expectedTestRisk
  have he : ∀ sample z, constantTargetLoss (cutoffTrain cutoff weight n sample) z =
      if n ≤ cutoff then ENNReal.ofReal ((weight - 1) ^ 2) else 1 := by
    intro sample z
    cases z
    by_cases hn : n ≤ cutoff <;> simp [cutoffTrain, constantTargetLoss, hn]
  simp_rw [he]
  simp

/-- Section 2, Hypothesis 1: no finite critical margin makes "higher EMC
implies lower test risk" true for arbitrary procedures. For every margin,
two scalar learners have EMCs strictly above `1+margin`, yet the higher-EMC
learner has strictly worse population risk on one training sample. The
source's undefined naturalness and perturbation restrictions are essential. -/
theorem EMC_alone_does_not_order_testRisk (margin : ℕ) :
    ∃ train₁ train₂ : TrainingProcedure Unit ℝ,
      (1 + margin : ℕ∞) < EMC (Measure.dirac ()) constantTargetLoss train₁ (1 / 10) ∧
      EMC (Measure.dirac ()) constantTargetLoss train₁ (1 / 10) <
        EMC (Measure.dirac ()) constantTargetLoss train₂ (1 / 10) ∧
      expectedTestRisk (Measure.dirac ()) constantTargetLoss train₁ 1 <
        expectedTestRisk (Measure.dirac ()) constantTargetLoss train₂ 1 := by
  refine ⟨cutoffTrain (margin + 2) 1, cutoffTrain (margin + 3) (4 / 5), ?_⟩
  have hε : (1 / 10 : ℝ≥0∞) < 1 := by norm_num
  have hzero : ENNReal.ofReal (((1 : ℝ) - 1) ^ 2) ≤ (1 / 10) := by norm_num
  have hsmall : ENNReal.ofReal (((4 / 5 : ℝ) - 1) ^ 2) ≤ (1 / 10) := by
    norm_num [ENNReal.ofReal_div_of_pos (by norm_num : (0 : ℝ) < 25)]
  rw [cutoff_EMC _ _ _ hε hzero, cutoff_EMC _ _ _ hε hsmall,
    cutoff_expectedTestRisk, cutoff_expectedTestRisk]
  have hc₁ : 1 ≤ margin + 2 := by omega
  have hc₂ : 1 ≤ margin + 3 := by omega
  simp only [hc₁, hc₂, ite_true]
  refine ⟨?_, ?_, ?_⟩
  · exact_mod_cast (show 1 + margin < margin + 2 by omega)
  · exact_mod_cast (show margin + 2 < margin + 3 by omega)
  · norm_num

end Transformer.DoubleDescent
