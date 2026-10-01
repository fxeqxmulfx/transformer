/-
# Real analytic preparation: PreparationSequences

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.NormalizedMaps

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- The small perturbation of the normalized distinguished monomial. -/
noncomputable def normalizedPreparationTail {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) : Base n → OriginSeq :=
  fun z ↦ analyticNormalizedCoefficientMap p r hr d z - monomialSeq d

/-- The quotient produced by dividing the degree-`d` monomial by the
normalized moving coefficient sequence. -/
noncomputable def normalizedPreparationQuotient {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) : Base n → OriginSeq :=
  fun z ↦ seqDivisionQuotientGlobal d
    (normalizedPreparationTail p r hr d z, monomialSeq d)

/-- The low-degree remainder produced by the same division. -/
noncomputable def normalizedPreparationRemainder {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) : Base n → OriginSeq :=
  fun z ↦ seqDivisionRemainderGlobal d
    (normalizedPreparationTail p r hr d z, monomialSeq d)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_normalizedPreparationTail {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) :
    AnalyticAt ℝ (normalizedPreparationTail p r hr d) 0 := by
  exact (analyticAt_analyticNormalizedCoefficientMap p r hr d).sub analyticAt_const

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_normalizedPreparationQuotient {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ)
    (hsmall : ‖normalizedPreparationTail p r hr d 0‖ < 1) :
    AnalyticAt ℝ (normalizedPreparationQuotient p r hr d) 0 := by
  have hin : AnalyticAt ℝ (fun z : Base n ↦
      (normalizedPreparationTail p r hr d z, monomialSeq d)) 0 :=
    (analyticAt_normalizedPreparationTail p r hr d).prod analyticAt_const
  exact (analyticAt_seqDivisionQuotientGlobal d _ hsmall).comp hin

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_normalizedPreparationRemainder {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ)
    (hsmall : ‖normalizedPreparationTail p r hr d 0‖ < 1) :
    AnalyticAt ℝ (normalizedPreparationRemainder p r hr d) 0 := by
  have hin : AnalyticAt ℝ (fun z : Base n ↦
      (normalizedPreparationTail p r hr d z, monomialSeq d)) 0 :=
    (analyticAt_normalizedPreparationTail p r hr d).prod analyticAt_const
  exact (analyticAt_seqDivisionRemainderGlobal d _ hsmall).comp hin

/-- Convolution with the degree-`d` monomial inserts `d` leading zero
coefficients.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem convolution_monomialSeq (q : OriginSeq) (d : ℕ) :
    convolution q (monomialSeq d) = seqLowShift d q := by
  apply lp.ext
  funext n
  rw [convolution_apply]
  by_cases hn : n < d
  · rw [seqLowShift_apply_of_lt d q hn]
    apply Finset.sum_eq_zero
    intro ij hij
    have hadd : ij.1 + ij.2 = n := Finset.mem_antidiagonal.mp hij
    have hne : ij.2 ≠ d := by omega
    rw [monomialSeq_apply_ne hne, mul_zero]
  · have hdn : d ≤ n := Nat.le_of_not_gt hn
    rw [seqLowShift_apply_of_le d q hdn]
    rw [Finset.sum_eq_single (n - d, d)]
    · rw [monomialSeq_apply_same, mul_one]
    · intro ij hij hne
      have hdne : ij.2 ≠ d := by
        intro heq
        apply hne
        have hadd : ij.1 + ij.2 = n := Finset.mem_antidiagonal.mp hij
        apply Prod.ext
        · simp only
          omega
        · simpa only [Prod.snd] using heq
      rw [monomialSeq_apply_ne hdne, mul_zero]
    · intro hnot
      exfalso
      apply hnot
      rw [Finset.mem_antidiagonal]
      omega

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma convolution_apply_eq_zero_of_right {q p : OriginSeq} {n : ℕ}
    (hp : ∀ k ≤ n, p k = 0) : convolution q p n = 0 := by
  rw [convolution_apply]
  apply Finset.sum_eq_zero
  intro ij hij
  have hadd : ij.1 + ij.2 = n := Finset.mem_antidiagonal.mp hij
  rw [hp ij.2 (by omega), mul_zero]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma normalizedPreparationTail_zero_of_le {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) (r : ℝ≥0) (hr0 : 0 < r)
    (hr : (r : ℝ≥0∞) < p.radius) {k : ℕ} (hk : k ≤ d) :
    normalizedPreparationTail p r hr d 0 k = 0 := by
  rw [normalizedPreparationTail, analyticNormalizedCoefficientMap_zero]
  by_cases hkd : k < d
  · change normalizedOriginCoeffs p r hr d k - monomialSeq d k = 0
    rw [normalizedOriginCoeffs_apply]
    have hcoeff : lastTaylorCoefficient p k 0 = 0 :=
      ((exactOrderInLastVariable_iff_lastTaylorCoefficients p hp).mp horder).1 k hkd
    rw [hcoeff, mul_zero, mul_zero]
    rw [monomialSeq_apply_ne (ne_of_lt hkd), sub_zero]
  · have hkeq : k = d := le_antisymm hk (Nat.le_of_not_gt hkd)
    subst k
    change normalizedOriginCoeffs p r hr d d - monomialSeq d d = 0
    rw [normalizedOriginCoeffs_apply, monomialSeq_apply_same]
    have htop : (r : ℝ) ^ d * lastTaylorCoefficient p d 0 ≠ 0 := by
      have hrC : (r : ℝ) ≠ 0 := by exact_mod_cast hr0.ne'
      exact mul_ne_zero (pow_ne_zero _ hrC)
        ((exactOrderInLastVariable_iff_lastTaylorCoefficients p hp).mp horder).2
    rw [inv_mul_cancel₀ htop, sub_self]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eventually_normalizedPreparation_factorization {n d : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius)
    (hclose : ‖analyticNormalizedCoefficientMap p r hr d 0 - monomialSeq d‖ <
      (1 : ℝ) / 2) :
    ∀ᶠ z in 𝓝 (0 : Base n),
      monomialSeq d = convolution (normalizedPreparationQuotient p r hr d z)
          (analyticNormalizedCoefficientMap p r hr d z) +
        normalizedPreparationRemainder p r hr d z ∧
      seqHighShift d (normalizedPreparationRemainder p r hr d z) = 0 := by
  have hsmall := eventually_norm_normalizedCoefficientMap_sub_monomial_lt_one
    (weightedCoefficientSeries p r).sum (originWeightedCoeffs p r hr d)
    (analyticAt_weightedCoefficientSeries_sum p r hr) hclose
  filter_upwards [hsmall] with z hz
  let tail := normalizedPreparationTail p r hr d z
  let q := normalizedPreparationQuotient p r hr d z
  let rem := normalizedPreparationRemainder p r hr d z
  have hfac := seqDivisionGlobal_factorization d tail (monomialSeq d) hz
  have hsupp := seqHighShift_divisionRemainderGlobal d tail (monomialSeq d) hz
  have hG : analyticNormalizedCoefficientMap p r hr d z = monomialSeq d + tail := by
    dsimp only [tail, normalizedPreparationTail]
    abel
  have hconv : convolution q (analyticNormalizedCoefficientMap p r hr d z) =
      seqLowShift d q + convolution q tail := by
    rw [hG, convolution_add_right, convolution_monomialSeq]
  refine ⟨?_, ?_⟩
  · rw [hconv]
    simpa only [q, rem, tail, normalizedPreparationQuotient,
      normalizedPreparationRemainder, add_assoc] using hfac
  · simpa only [rem, tail, normalizedPreparationRemainder] using hsupp

end Transformer.AnalyticPreparation
