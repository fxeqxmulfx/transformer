/-
# Real analytic preparation: EvaluationUniqueness

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d, PreparationUniqueness.lean.
The scalar field is real; only the required division infrastructure is copied.
Apache-2.0 licenses: third_party/classical-complex-wpt/LICENSE and
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.SequenceUniqueness

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/-- Equality of normalized prepared tails recovers equality of every
coefficient when the weight is positive.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedTailSeq_eq_iff {n d : ℕ} {r : ℝ≥0} (hr : 0 < r)
    (a a' : Fin d → Base n → ℝ) (z : Base n) :
    preparedTailSeq r d a z = preparedTailSeq r d a' z ↔
      ∀ i, a i z = a' i z := by
  constructor
  · intro h i
    have hi := congrArg (fun q : L1Sequence ↦ q (i : ℕ)) h
    rw [preparedTailSeq_apply_fin, preparedTailSeq_apply_fin] at hi
    have hrC : (r : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have hc : (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ)) ≠ 0 :=
      mul_ne_zero (inv_ne_zero (pow_ne_zero _ hrC)) (pow_ne_zero _ hrC)
    exact mul_left_cancel₀ hc hi
  · intro h
    unfold preparedTailSeq
    apply Finset.sum_congr rfl
    intro i hi
    rw [h i]

/-- The analytic function on the unit disc represented by an `ℓ¹` sequence
determines every coefficient.  The hypothesis is stated as germ equality,
which is exactly what is available after shrinking the common analytic
neighborhood in preparation uniqueness.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eq_of_evalL1PowerSeries_eventuallyEq (a b : L1Sequence)
    (h : (fun w ↦ evalL1PowerSeries a w) =ᶠ[nhds 0]
      (fun w ↦ evalL1PowerSeries b w)) : a = b := by
  let appA : (L1Sequence →L[ℝ] ℝ) →L[ℝ] ℝ :=
    ContinuousLinearMap.apply ℝ ℝ a
  let appB : (L1Sequence →L[ℝ] ℝ) →L[ℝ] ℝ :=
    ContinuousLinearMap.apply ℝ ℝ b
  let pA : FormalMultilinearSeries ℝ ℝ ℝ :=
    appA.compFormalMultilinearSeries l1OperatorSeries
  let pB : FormalMultilinearSeries ℝ ℝ ℝ :=
    appB.compFormalMultilinearSeries l1OperatorSeries
  have hrad : (0 : ℝ≥0∞) < l1OperatorSeries.radius :=
    (by norm_num : (0 : ℝ≥0∞) < 1).trans_le one_le_radius_l1OperatorSeries
  have hA : HasFPowerSeriesAt (fun w ↦ evalL1PowerSeries a w) pA 0 := by
    have hball := appA.comp_hasFPowerSeriesOnBall
      (l1OperatorSeries.hasFPowerSeriesOnBall hrad)
    exact hball.hasFPowerSeriesAt.congr (by
      filter_upwards
      intro w
      rfl)
  have hB : HasFPowerSeriesAt (fun w ↦ evalL1PowerSeries b w) pB 0 := by
    have hball := appB.comp_hasFPowerSeriesOnBall
      (l1OperatorSeries.hasFPowerSeriesOnBall hrad)
    exact hball.hasFPowerSeriesAt.congr (by
      filter_upwards
      intro w
      rfl)
  have hp : pA = pB := hA.eq_formalMultilinearSeries_of_eventually hB h
  apply lp.ext
  funext k
  have hk := congrArg
    (fun p : FormalMultilinearSeries ℝ ℝ ℝ ↦ p k (fun _ ↦ (1 : ℝ))) hp
  simpa [pA, pB, appA, appB, l1OperatorSeries, coefficientEval, lp.evalCLM,
    ContinuousLinearMap.compFormalMultilinearSeries_apply] using hk


end Transformer.AnalyticPreparation
