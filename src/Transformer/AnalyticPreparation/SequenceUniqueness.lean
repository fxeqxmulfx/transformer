/-
# Real analytic preparation: SequenceUniqueness

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d, L1Division.lean and
PreparationUniqueness.lean.
The scalar field is real; only the required division infrastructure is copied.
Apache-2.0 licenses: third_party/classical-complex-wpt/LICENSE and
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.AnalyticDivision
import Mathlib.Analysis.Analytic.Uniqueness

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/-- High and low coefficient shifts cancel. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem seqHighShift_seqLowShift (d : ℕ) (q : L1Sequence) :
    seqHighShift d (seqLowShift d q) = q := by
  apply lp.ext
  funext k
  exact seqLowShift_apply_add d q k

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivisionInverse_left (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (q : L1Coeff ℕ) :
    seqDivisionInverse d p hp (q + seqHighShift d (convolution q p)) = q := by
  let T := -(seqDivisionPerturbation d p)
  let hT : ‖T‖ < 1 := norm_neg_seqDivisionPerturbation_lt_one d p hp
  have h := congrArg (fun L : L1Coeff ℕ →L[ℝ] L1Coeff ℕ ↦ L q)
    (Units.inv_mul (Units.oneSub T hT))
  simpa [seqDivisionInverse, T, hT, seqDivisionPerturbation_apply,
    sub_eq_add_neg] using h

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivisionQuotient_unique (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (f q r : L1Coeff ℕ)
    (hfac : f = seqLowShift d q + convolution q p + r)
    (hr : seqHighShift d r = 0) : q = seqDivisionQuotient d p hp f := by
  have hmap := congrArg (seqHighShiftCLM d) hfac
  have heq : q + seqHighShift d (convolution q p) = seqHighShift d f := by
    change seqHighShift d f = seqHighShift d (seqLowShift d q + convolution q p + r) at hmap
    rw [seqHighShift_add, seqHighShift_add, seqHighShift_seqLowShift, hr, add_zero] at hmap
    exact hmap.symm
  change q = seqDivisionInverse d p hp (seqHighShift d f)
  rw [← heq]
  exact (seqDivisionInverse_left d p hp q).symm

/-- Any two quotient/remainder decompositions for the same small normalized
divisor agree.  This is the form of `seqDivision_existsUnique` used by germ
uniqueness.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivision_factorizations_unique (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (f q₁ r₁ q₂ r₂ : L1Coeff ℕ)
    (hfac₁ : f = seqLowShift d q₁ + convolution q₁ p + r₁)
    (hr₁ : seqHighShift d r₁ = 0)
    (hfac₂ : f = seqLowShift d q₂ + convolution q₂ p + r₂)
    (hr₂ : seqHighShift d r₂ = 0) :
    q₁ = q₂ ∧ r₁ = r₂ := by
  have hq₁ : q₁ = seqDivisionQuotient d p hp f :=
    seqDivisionQuotient_unique d p hp f q₁ r₁ hfac₁ hr₁
  have hq₂ : q₂ = seqDivisionQuotient d p hp f :=
    seqDivisionQuotient_unique d p hp f q₂ r₂ hfac₂ hr₂
  have hq : q₁ = q₂ := hq₁.trans hq₂.symm
  refine ⟨hq, ?_⟩
  have hfac₁' : f = seqLowShift d q₂ + convolution q₂ p + r₁ := by
    simpa only [hq] using hfac₁
  have hsum : (seqLowShift d q₂ + convolution q₂ p) + r₁ =
      (seqLowShift d q₂ + convolution q₂ p) + r₂ := by
    simpa only [add_assoc] using hfac₁'.symm.trans hfac₂
  exact add_left_cancel hsum


end Transformer.AnalyticPreparation
