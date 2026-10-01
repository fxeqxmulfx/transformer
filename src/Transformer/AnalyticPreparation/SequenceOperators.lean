/-
# Real analytic preparation: SequenceOperators

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.SequenceShifts
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Analytic.Constructions

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

def seqDivisionPerturbationMap (d : ℕ) :
    L1Coeff ℕ →L[ℝ] (L1Coeff ℕ →L[ℝ] L1Coeff ℕ) :=
  ({
    toFun := seqDivisionPerturbation d
    map_add' := by
      intro p₁ p₂
      apply ContinuousLinearMap.ext
      intro q
      change seqHighShift d (convolution q (p₁ + p₂)) =
        seqHighShift d (convolution q p₁) + seqHighShift d (convolution q p₂)
      rw [convolution_add_right, seqHighShift_add]
    map_smul' := by
      intro c p
      apply ContinuousLinearMap.ext
      intro q
      change seqHighShift d (convolution q (c • p)) = c • seqHighShift d (convolution q p)
      rw [convolution_smul_right, seqHighShift_smul]
    } : L1Coeff ℕ →ₗ[ℝ] (L1Coeff ℕ →L[ℝ] L1Coeff ℕ)).mkContinuous 1 (by
      intro p
      simpa using norm_seqDivisionPerturbation_le d p)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqDivisionPerturbationMap_apply (d : ℕ) (p : L1Coeff ℕ) :
    seqDivisionPerturbationMap d p = seqDivisionPerturbation d p := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma norm_neg_seqDivisionPerturbation_lt_one (d : ℕ) (p : L1Coeff ℕ)
    (hp : ‖p‖ < 1) : ‖-(seqDivisionPerturbation d p)‖ < 1 := by
  rw [norm_neg]
  exact lt_of_le_of_lt (norm_seqDivisionPerturbation_le d p) hp

noncomputable def seqDivisionInverse (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1) :
    L1Coeff ℕ →L[ℝ] L1Coeff ℕ :=
  ↑((Units.oneSub (-(seqDivisionPerturbation d p))
    (norm_neg_seqDivisionPerturbation_lt_one d p hp))⁻¹)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivisionInverse_right (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (b : L1Coeff ℕ) :
    seqDivisionInverse d p hp b +
      seqHighShift d (convolution (seqDivisionInverse d p hp b) p) = b := by
  let T := -(seqDivisionPerturbation d p)
  let hT : ‖T‖ < 1 := norm_neg_seqDivisionPerturbation_lt_one d p hp
  have h := congrArg (fun L : L1Coeff ℕ →L[ℝ] L1Coeff ℕ ↦ L b)
    (Units.mul_inv (Units.oneSub T hT))
  simpa [seqDivisionInverse, T, hT, seqDivisionPerturbation_apply,
    sub_eq_add_neg] using h

noncomputable def seqDivisionQuotient (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (f : L1Coeff ℕ) : L1Coeff ℕ :=
  seqDivisionInverse d p hp (seqHighShift d f)

noncomputable def seqDivisionRemainder (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (f : L1Coeff ℕ) : L1Coeff ℕ :=
  seqLowCut d (f - convolution (seqDivisionQuotient d p hp f) p)

def seqDivisionOperatorInput (d : ℕ) :
    (L1Coeff ℕ × L1Coeff ℕ) →L[ℝ] (L1Coeff ℕ →L[ℝ] L1Coeff ℕ) :=
  seqDivisionPerturbationMap d ∘L
    ContinuousLinearMap.fst ℝ (L1Coeff ℕ) (L1Coeff ℕ)

def seqDivisionRhsInput (d : ℕ) :
    (L1Coeff ℕ × L1Coeff ℕ) →L[ℝ] L1Coeff ℕ :=
  seqHighShiftCLM d ∘L ContinuousLinearMap.snd ℝ (L1Coeff ℕ) (L1Coeff ℕ)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqDivisionOperatorInput_apply (d : ℕ) (pf : L1Coeff ℕ × L1Coeff ℕ) :
    seqDivisionOperatorInput d pf = seqDivisionPerturbation d pf.1 := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqDivisionRhsInput_apply (d : ℕ) (pf : L1Coeff ℕ × L1Coeff ℕ) :
    seqDivisionRhsInput d pf = seqHighShift d pf.2 := rfl

/-- Proof-independent quotient, jointly analytic on the open set `‖p‖ < 1`. -/
noncomputable def seqDivisionQuotientGlobal (d : ℕ) (pf : L1Coeff ℕ × L1Coeff ℕ) :
    L1Coeff ℕ :=
  Ring.inverse (1 + seqDivisionOperatorInput d pf) (seqDivisionRhsInput d pf)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivisionQuotientGlobal_eq (d : ℕ) (p f : L1Coeff ℕ) (hp : ‖p‖ < 1) :
    seqDivisionQuotientGlobal d (p,f) = seqDivisionQuotient d p hp f := by
  let K := seqDivisionPerturbation d p
  let hK : ‖-K‖ < 1 := norm_neg_seqDivisionPerturbation_lt_one d p hp
  have hinv : Ring.inverse (1 + K) = ↑((Units.oneSub (-K) hK)⁻¹) := by
    rw [show 1 + K = 1 - (-K) by abel]
    exact NormedRing.inverse_one_sub (-K) hK
  change Ring.inverse (1 + K) (seqHighShift d f) =
    (↑((Units.oneSub (-K) hK)⁻¹) : L1Coeff ℕ →L[ℝ] L1Coeff ℕ) (seqHighShift d f)
  rw [hinv]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_seqDivisionQuotientGlobal (d : ℕ)
    (pf : L1Coeff ℕ × L1Coeff ℕ) (hp : ‖pf.1‖ < 1) :
    AnalyticAt ℝ (seqDivisionQuotientGlobal d) pf := by
  have hK : ‖seqDivisionOperatorInput d pf‖ < 1 :=
    lt_of_le_of_lt (norm_seqDivisionPerturbation_le d pf.1) hp
  exact analyticAt_inverseOneAdd_apply
    (seqDivisionOperatorInput d) (seqDivisionRhsInput d) pf hK

/-- Proof-independent remainder, jointly analytic wherever the divisor tail has norm below one. -/
noncomputable def seqDivisionRemainderGlobal (d : ℕ) (pf : L1Coeff ℕ × L1Coeff ℕ) :
    L1Coeff ℕ :=
  seqLowCut d (pf.2 - convolution (seqDivisionQuotientGlobal d pf) pf.1)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivisionRemainderGlobal_eq (d : ℕ) (p f : L1Coeff ℕ) (hp : ‖p‖ < 1) :
    seqDivisionRemainderGlobal d (p,f) = seqDivisionRemainder d p hp f := by
  simp only [seqDivisionRemainderGlobal, seqDivisionRemainder]
  rw [seqDivisionQuotientGlobal_eq d p f hp]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_seqDivisionRemainderGlobal (d : ℕ)
    (pf : L1Coeff ℕ × L1Coeff ℕ) (hp : ‖pf.1‖ < 1) :
    AnalyticAt ℝ (seqDivisionRemainderGlobal d) pf := by
  have hq := analyticAt_seqDivisionQuotientGlobal d pf hp
  have happ := (convolutionRightMap (A := ℕ)).analyticAt_bilinear
    (pf.1, seqDivisionQuotientGlobal d pf)
  have hconv : AnalyticAt ℝ
      (fun x : L1Coeff ℕ × L1Coeff ℕ ↦
        convolution (seqDivisionQuotientGlobal d x) x.1) pf := by
    have hpair := (analyticAt_fst (𝕜 := ℝ)).prod hq
    simpa [Function.comp_def] using AnalyticAt.comp (x := pf) happ hpair
  have hdiff : AnalyticAt ℝ
      (fun x : L1Coeff ℕ × L1Coeff ℕ ↦
        x.2 - convolution (seqDivisionQuotientGlobal d x) x.1) pf :=
    (analyticAt_snd (𝕜 := ℝ)).sub hconv
  have hout := (seqLowCutCLM d).analyticAt
    (pf.2 - convolution (seqDivisionQuotientGlobal d pf) pf.1)
  change AnalyticAt ℝ (fun x : L1Coeff ℕ × L1Coeff ℕ ↦
    seqLowCut d (x.2 - convolution (seqDivisionQuotientGlobal d x) x.1)) pf
  have hcomp := AnalyticAt.comp (x := pf) hout hdiff
  simpa [seqLowCutCLM, Function.comp_def] using hcomp

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqDivisionQuotient_equation (d : ℕ) (p : L1Coeff ℕ) (hp : ‖p‖ < 1)
    (f : L1Coeff ℕ) :
    seqDivisionQuotient d p hp f +
      seqHighShift d (convolution (seqDivisionQuotient d p hp f) p) =
        seqHighShift d f := by
  exact seqDivisionInverse_right d p hp (seqHighShift d f)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem seqHighShift_divisionRemainder (d : ℕ) (p : L1Coeff ℕ)
    (hp : ‖p‖ < 1) (f : L1Coeff ℕ) :
    seqHighShift d (seqDivisionRemainder d p hp f) = 0 :=
  seqHighShift_seqLowCut d _

end Transformer.AnalyticPreparation
