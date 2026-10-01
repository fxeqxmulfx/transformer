/-
# Real analytic preparation: AnalyticDivisionFunctions

Adapted from LocalComplexGeometry, revision
029697242294aea7989bf5d391199696a43490df, WPTBridge/Division.lean.
The scalar field is real; only the required division infrastructure is copied.
Apache-2.0 licenses: third_party/classical-complex-wpt/LICENSE and
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.AnalyticDivisionSequences

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/-- The analytic quotient reconstructed in the original distinguished variable.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def analyticDivisionQuotient {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (x : Ambient n) : ℝ :=
  (((r : ℝ) ^ d)⁻¹) *
    evalL1PowerSeries (quotientSeq p r a x.1) ((r : ℝ)⁻¹ * x.2)

/-- The `i`-th remainder coefficient, rescaled back to the original variable.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def analyticDivisionRemainderCoefficient {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (i : Fin d) (z : Base n) : ℝ :=
  remainderSeq p r a z (i : ℕ) * (r : ℝ)⁻¹ ^ (i : ℕ)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_analyticDivisionQuotient {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hrp : (r : ℝ≥0∞) < p.radius)
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) :
    AnalyticAt ℝ (analyticDivisionQuotient p r a) 0 := by
  have heval := Transformer.AnalyticPreparation.AnalyticAt.evalL1PowerSeries
    (analyticAt_quotientSeq p r hrp a ha ha0) ((r : ℝ)⁻¹)
  exact analyticAt_const.mul heval

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_analyticDivisionRemainderCoefficient {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hrp : (r : ℝ≥0∞) < p.radius)
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0)
    (ha0 : ∀ i, a i 0 = 0) (i : Fin d) :
    AnalyticAt ℝ (analyticDivisionRemainderCoefficient p r a i) 0 := by
  have hrem := analyticAt_remainderSeq p r hrp a ha ha0
  have hcoord : AnalyticAt ℝ (fun z : Base n ↦ remainderSeq p r a z (i : ℕ)) 0 := by
    change AnalyticAt ℝ
      (fun z : Base n ↦ coefficientEval (i : ℕ) (remainderSeq p r a z)) 0
    exact ((coefficientEval (i : ℕ)).analyticAt _).comp hrem
  exact hcoord.mul analyticAt_const

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticDivision_pointwise {n d : ℕ}
    {h : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr0 : 0 < r)
    (a : Fin d → Base n → ℝ)
    (z : Base n) (w : ℝ)
    (hsmall : ‖preparedTailSeq r d a z‖ < 1)
    (hw : ‖(r : ℝ)⁻¹ * w‖ < 1)
    (hreconstruct :
      evalL1PowerSeries ((weightedCoefficientSeries p r).sum z)
        ((r : ℝ)⁻¹ * w) = h (z, w)) :
    h (z, w) =
      analyticDivisionQuotient p r a (z, w) * preparedPolynomial d a (z, w) +
        ∑ i : Fin d, analyticDivisionRemainderCoefficient p r a i z * w ^ (i : ℕ) := by
  let tail : L1Sequence := preparedTailSeq r d a z
  let dividend : L1Sequence := (weightedCoefficientSeries p r).sum z
  let quotient : L1Sequence := quotientSeq p r a z
  let remainder : L1Sequence := remainderSeq p r a z
  let t : ℝ := (r : ℝ)⁻¹ * w
  have hrC : (r : ℝ) ≠ 0 := by exact_mod_cast hr0.ne'
  have hrt : (r : ℝ) * t = w := by
    simp [t, hrC]
  have hseq : dividend =
      seqLowShift d quotient + convolution quotient tail + remainder := by
    simpa only [tail, dividend, quotient, remainder, quotientSeq, remainderSeq,
      divisionInput] using
      seqDivisionGlobal_factorization d (preparedTailSeq r d a z)
        ((weightedCoefficientSeries p r).sum z) hsmall
  have hdivisorSeq :
      seqLowShift d quotient + convolution quotient tail =
        convolution quotient (monomialSeq d + tail) := by
    rw [← convolution_monomialSeq quotient d, convolution_add_right]
  have hquotientEval :
      evalL1PowerSeries (seqLowShift d quotient + convolution quotient tail) t =
        evalL1PowerSeries quotient t * (((r : ℝ) ^ d)⁻¹) *
          preparedPolynomial d a (z, w) := by
    rw [hdivisorSeq, evalL1PowerSeries_convolution _ _ hw,
      eval_preparedPolynomialSeq r hr0 a z hw, hrt]
    ring
  have hremainderSupport : seqHighShift d remainder = 0 := by
    simpa only [tail, dividend, remainder, remainderSeq, divisionInput] using
      seqHighShift_divisionRemainderGlobal d (preparedTailSeq r d a z)
        ((weightedCoefficientSeries p r).sum z) hsmall
  have hremainderEval :
      evalL1PowerSeries remainder t =
        ∑ i : Fin d,
          analyticDivisionRemainderCoefficient p r a i z * w ^ (i : ℕ) := by
    rw [evalL1PowerSeries_eq_sum_fin_of_highShift_eq_zero d remainder
      hremainderSupport hw]
    apply Finset.sum_congr rfl
    intro i hi
    simp only [analyticDivisionRemainderCoefficient, remainder]
    rw [mul_pow]
    ring
  calc
    h (z, w) = evalL1PowerSeries dividend t := by
      simpa only [dividend, t] using hreconstruct.symm
    _ = evalL1PowerSeries
        (seqLowShift d quotient + convolution quotient tail + remainder) t := by
      rw [hseq]
    _ = evalL1PowerSeries (seqLowShift d quotient + convolution quotient tail) t +
          evalL1PowerSeries remainder t := by
      rw [evalL1PowerSeries_add]
    _ = analyticDivisionQuotient p r a (z, w) *
          preparedPolynomial d a (z, w) +
          ∑ i : Fin d,
            analyticDivisionRemainderCoefficient p r a i z * w ^ (i : ℕ) := by
      rw [hquotientEval, hremainderEval]
      simp only [analyticDivisionQuotient, quotient, t]
      ring


end Transformer.AnalyticPreparation
