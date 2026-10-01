/-
# Real analytic preparation: SequenceDivision

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.SequenceOperators
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Analytic.Constructions

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivision_factorization (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (f : L1Coeff ℕ) :
    f = seqLowShift d (seqDivisionQuotient d p hp f) +
      convolution (seqDivisionQuotient d p hp f) p + seqDivisionRemainder d p hp f := by
  let q := seqDivisionQuotient d p hp f
  have hq := seqDivisionQuotient_equation d p hp f
  have hhigh : seqHighShift d (f - convolution q p) = q := by
    change seqHighShiftCLM d (f - convolution q p) = q
    rw [map_sub]
    change seqHighShift d f - seqHighShift d (convolution q p) = q
    rw [← hq]
    abel
  have hdec := seqLowShift_highShift_add_lowCut d (f - convolution q p)
  rw [hhigh] at hdec
  change f = seqLowShift d q + convolution q p + seqLowCut d (f - convolution q p)
  calc
    f = (f - convolution q p) + convolution q p := by abel
    _ = (seqLowShift d q + seqLowCut d (f - convolution q p)) + convolution q p :=
      congrArg (fun x ↦ x + convolution q p) hdec.symm
    _ = seqLowShift d q + convolution q p + seqLowCut d (f - convolution q p) := by abel

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivisionGlobal_factorization (d : ℕ) (p f : L1Coeff ℕ) (hp : ‖p‖ < 1) :
    f = seqLowShift d (seqDivisionQuotientGlobal d (p,f)) +
      convolution (seqDivisionQuotientGlobal d (p,f)) p +
        seqDivisionRemainderGlobal d (p,f) := by
  rw [seqDivisionQuotientGlobal_eq d p f hp, seqDivisionRemainderGlobal_eq d p f hp]
  exact seqDivision_factorization d p hp f

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem seqHighShift_divisionRemainderGlobal (d : ℕ) (p f : L1Coeff ℕ)
    (hp : ‖p‖ < 1) :
    seqHighShift d (seqDivisionRemainderGlobal d (p,f)) = 0 := by
  rw [seqDivisionRemainderGlobal_eq d p f hp]
  exact seqHighShift_divisionRemainder d p hp f

end Transformer.AnalyticPreparation
