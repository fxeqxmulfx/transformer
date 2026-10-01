/-
# Real analytic preparation: DivisionEvaluation

Adapted from LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/DivisionUniqueness.lean.
The scalar field is real; only the required division infrastructure is copied.
Apache-2.0 licenses: third_party/classical-complex-wpt/LICENSE and
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.EvaluationUniqueness

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/-- Evaluation of a normalized finite prepared tail in physical coordinates.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_preparedTailSeq_physical {n d : ℕ} (r : ℝ≥0)
    (b : Fin d → Base n → ℝ) (z : Base n) {w : ℝ} (hw : ‖w‖ < 1) :
    evalL1PowerSeries (preparedTailSeq r d b z) w =
      (((r : ℝ) ^ d)⁻¹) *
        ∑ i : Fin d, b i z * ((r : ℝ) * w) ^ (i : ℕ) := by
  rw [evalL1PowerSeries_eq_sum_fin_of_highShift_eq_zero d _
    (seqHighShift_preparedTailSeq r b z) hw]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  rw [preparedTailSeq_apply_fin, mul_pow]
  ring

/-- Evaluation of a quotient sequence times the normalized prepared divisor.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_sequenceProduct_preparedPolynomial {n d : ℕ} (r : ℝ≥0)
    (hr0 : 0 < r) (a : Fin d → Base n → ℝ) (z : Base n)
    (q : L1Sequence) {w : ℝ} (hw : ‖w‖ < 1) :
    evalL1PowerSeries
        (seqLowShift d q + convolution q (preparedTailSeq r d a z)) w =
      evalL1PowerSeries q w * (((r : ℝ) ^ d)⁻¹) *
        preparedPolynomial d a (z, (r : ℝ) * w) := by
  have hseq : seqLowShift d q + convolution q (preparedTailSeq r d a z) =
      convolution q (monomialSeq d + preparedTailSeq r d a z) := by
    rw [← convolution_monomialSeq q d, convolution_add_right]
  rw [hseq, evalL1PowerSeries_convolution _ _ hw,
    eval_preparedPolynomialSeq r hr0 a z hw]
  ring


end Transformer.AnalyticPreparation
