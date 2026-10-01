/-
# Real analytic preparation: AnalyticDivisionSequences

Adapted from LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/DivisionCore.lean.
The scalar field is real; only the required division infrastructure is copied.
Apache-2.0 licenses: third_party/classical-complex-wpt/LICENSE and
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.PreparedPolynomialSequences

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/-- Analytic divisor-tail/dividend input for WPT's total sequence division maps.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def divisionInput {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (z : Base n) : L1Sequence × L1Sequence :=
  (preparedTailSeq r d a z, (weightedCoefficientSeries p r).sum z)

/-- Sequence quotient supplied by the pinned WPT division operator.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def quotientSeq {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (z : Base n) : L1Sequence :=
  seqDivisionQuotientGlobal d (divisionInput p r a z)

/-- Sequence remainder supplied by the pinned WPT division operator.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def remainderSeq {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (z : Base n) : L1Sequence :=
  seqDivisionRemainderGlobal d (divisionInput p r a z)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedTailSeq_zero {n d : ℕ} (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (ha0 : ∀ i, a i 0 = 0) :
    preparedTailSeq r d a 0 = 0 := by
  classical
  unfold preparedTailSeq
  simp [ha0]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_divisionInput_fst_zero_lt_one {n d : ℕ} (r : ℝ≥0)
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ)
    (a : Fin d → Base n → ℝ) (ha0 : ∀ i, a i 0 = 0) :
    ‖(divisionInput p r a 0).1‖ < 1 := by
  simp [divisionInput, preparedTailSeq_zero r a ha0]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_divisionInput {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hrp : (r : ℝ≥0∞) < p.radius)
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0) :
    AnalyticAt ℝ (divisionInput p r a) 0 := by
  exact (analyticAt_preparedTailSeq r a ha).prod
    (analyticAt_weightedCoefficientSeries_sum p r hrp)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_quotientSeq {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hrp : (r : ℝ≥0∞) < p.radius)
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    AnalyticAt ℝ (quotientSeq p r a) 0 := by
  have hout := analyticAt_seqDivisionQuotientGlobal d (divisionInput p r a 0)
    (norm_divisionInput_fst_zero_lt_one r p a ha0)
  exact hout.comp (analyticAt_divisionInput p r hrp a ha)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_remainderSeq {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hrp : (r : ℝ≥0∞) < p.radius)
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    AnalyticAt ℝ (remainderSeq p r a) 0 := by
  have hout := analyticAt_seqDivisionRemainderGlobal d (divisionInput p r a 0)
    (norm_divisionInput_fst_zero_lt_one r p a ha0)
  exact hout.comp (analyticAt_divisionInput p r hrp a ha)


end Transformer.AnalyticPreparation
