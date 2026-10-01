/-
# Real analytic preparation: PreparationOrigin

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.PreparationSequences

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem normalizedPreparationRemainder_zero {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) (r : ℝ≥0) (hr0 : 0 < r)
    (hr : (r : ℝ≥0∞) < p.radius)
    (hsmall : ‖normalizedPreparationTail p r hr d 0‖ < 1) :
    normalizedPreparationRemainder p r hr d 0 = 0 := by
  rw [normalizedPreparationRemainder, seqDivisionRemainderGlobal_eq d _ _ hsmall]
  apply lp.ext
  funext k
  by_cases hk : k < d
  · rw [seqDivisionRemainder, seqLowCut_apply_of_lt d _ hk]
    change monomialSeq d k - convolution
      (seqDivisionQuotient d (normalizedPreparationTail p r hr d 0) hsmall
        (monomialSeq d)) (normalizedPreparationTail p r hr d 0) k = 0
    rw [monomialSeq_apply_ne (ne_of_lt hk)]
    rw [convolution_apply_eq_zero_of_right (fun j hj ↦
      normalizedPreparationTail_zero_of_le p hp horder r hr0 hr (hj.trans hk.le))]
    exact sub_zero 0
  · rw [seqDivisionRemainder, seqLowCut_apply_of_le d _ (Nat.le_of_not_gt hk)]
    rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem normalizedPreparationQuotient_zero_apply_zero {n d : ℕ}
    {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) (r : ℝ≥0) (hr0 : 0 < r)
    (hr : (r : ℝ≥0∞) < p.radius)
    (hsmall : ‖normalizedPreparationTail p r hr d 0‖ < 1) :
    normalizedPreparationQuotient p r hr d 0 0 = 1 := by
  rw [normalizedPreparationQuotient, seqDivisionQuotientGlobal_eq d _ _ hsmall]
  have heq := seqDivisionQuotient_equation d
    (normalizedPreparationTail p r hr d 0) hsmall (monomialSeq d)
  have hconv : convolution
      (seqDivisionQuotient d (normalizedPreparationTail p r hr d 0) hsmall
        (monomialSeq d)) (normalizedPreparationTail p r hr d 0) d = 0 := by
    exact convolution_apply_eq_zero_of_right (fun j hj ↦
      normalizedPreparationTail_zero_of_le p hp horder r hr0 hr hj)
  have hcoord := congrArg (fun s : OriginSeq ↦ s 0) heq
  simp only [lp.coeFn_add, Pi.add_apply, seqHighShift_apply, zero_add] at hcoord
  rw [hconv, monomialSeq_apply_same, add_zero] at hcoord
  exact hcoord

/-- Exact order yields a radius and analytic quotient/remainder coefficient
maps with the preparation factorization, vanishing origin remainder, and
unit-normalized origin quotient.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_normalizedPreparationSequences {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) :
    ∃ (r : ℝ≥0) (hr : (r : ℝ≥0∞) < p.radius),
      0 < r ∧ originWeightedCoeffs p r hr d ≠ 0 ∧
      AnalyticAt ℝ (normalizedPreparationQuotient p r hr d) 0 ∧
      AnalyticAt ℝ (normalizedPreparationRemainder p r hr d) 0 ∧
      (∀ᶠ z in 𝓝 (0 : Base n),
        monomialSeq d = convolution (normalizedPreparationQuotient p r hr d z)
            (analyticNormalizedCoefficientMap p r hr d z) +
          normalizedPreparationRemainder p r hr d z ∧
        seqHighShift d (normalizedPreparationRemainder p r hr d z) = 0) ∧
      normalizedPreparationRemainder p r hr d 0 = 0 ∧
      normalizedPreparationQuotient p r hr d 0 0 = 1 := by
  obtain ⟨r, hr, hr0, htop, han, hclose, hevent⟩ :=
    exists_normalizedAnalyticCoefficientMap p hp horder
  have hsmall : ‖normalizedPreparationTail p r hr d 0‖ < 1 := by
    exact hclose.trans (by norm_num)
  refine ⟨r, hr, hr0, htop,
    analyticAt_normalizedPreparationQuotient p r hr d hsmall,
    analyticAt_normalizedPreparationRemainder p r hr d hsmall,
    eventually_normalizedPreparation_factorization p r hr hclose,
    normalizedPreparationRemainder_zero p hp horder r hr0 hr hsmall,
    normalizedPreparationQuotient_zero_apply_zero p hp horder r hr0 hr hsmall⟩

end Transformer.AnalyticPreparation
