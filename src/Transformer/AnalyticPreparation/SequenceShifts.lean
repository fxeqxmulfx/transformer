/-
# Real analytic preparation: SequenceShifts

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.SequenceEmbedding
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Analytic.Constructions

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma seqHighShift_smul (d : ℕ) (c : ℝ) (f : L1Coeff ℕ) :
    seqHighShift d (c • f) = c • seqHighShift d f := by
  apply lp.ext
  funext n
  rfl

def seqHighShiftCLM (d : ℕ) : L1Coeff ℕ →L[ℝ] L1Coeff ℕ :=
  ({
    toFun := seqHighShift d
    map_add' := seqHighShift_add d
    map_smul' := seqHighShift_smul d
    } : L1Coeff ℕ →ₗ[ℝ] L1Coeff ℕ).mkContinuous 1 (by
      intro f
      simpa using norm_seqHighShift_le d f)

def seqLowIndex (d : ℕ) : ℕ ↪ ℕ where
  toFun n := n + d
  inj' := fun _ _ h ↦ Nat.add_right_cancel h

/-- Insert `d` zero coefficients at the beginning of a one-variable sequence. -/
def seqLowShift (d : ℕ) (f : L1Coeff ℕ) : L1Coeff ℕ :=
  L1Coeff.embed (seqLowIndex d) f

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqLowShift_apply_add (d : ℕ) (f : L1Coeff ℕ) (n : ℕ) :
    seqLowShift d f (n + d) = f n := L1Coeff.embed_apply_self (seqLowIndex d) f n

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma seqLowShift_apply_of_lt (d : ℕ) (f : L1Coeff ℕ) {n : ℕ} (hn : n < d) :
    seqLowShift d f n = 0 := by
  apply L1Coeff.embed_apply_of_not_mem
  rintro ⟨m, hm⟩
  change m + d = n at hm
  omega

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma seqLowShift_apply_of_le (d : ℕ) (f : L1Coeff ℕ) {n : ℕ} (hn : d ≤ n) :
    seqLowShift d f n = f (n - d) := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le hn
  have hsub : d + m - d = m := Nat.add_sub_cancel_left d m
  rw [hsub]
  simpa only [Nat.add_comm] using seqLowShift_apply_add d f m

/-- Keep precisely the coefficients below degree `d`. -/
def seqLowCut (d : ℕ) (f : L1Coeff ℕ) : L1Coeff ℕ :=
  ⟨fun n ↦ if n < d then f n else 0, by
    apply memℓp_gen
    have hs : Summable (fun n : ℕ ↦ ‖if n < d then f n else 0‖) :=
      (L1Coeff.summable_norm f).of_nonneg_of_le
        (fun _ ↦ norm_nonneg _) (fun n ↦ by split_ifs <;> simp)
    simpa using hs⟩

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqLowCut_apply_of_lt (d : ℕ) (f : L1Coeff ℕ) {n : ℕ} (hn : n < d) :
    seqLowCut d f n = f n := by simp [seqLowCut, hn]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqLowCut_apply_of_le (d : ℕ) (f : L1Coeff ℕ) {n : ℕ} (hn : d ≤ n) :
    seqLowCut d f n = 0 := by simp [seqLowCut, Nat.not_lt.mpr hn]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_seqLowCut_le (d : ℕ) (f : L1Coeff ℕ) : ‖seqLowCut d f‖ ≤ ‖f‖ := by
  rw [L1Coeff.norm_eq_tsum_norm, L1Coeff.norm_eq_tsum_norm]
  apply (L1Coeff.summable_norm (seqLowCut d f)).tsum_le_tsum
  · intro n
    change ‖if n < d then f n else 0‖ ≤ ‖f n‖
    split_ifs <;> simp
  · exact L1Coeff.summable_norm f

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma seqLowCut_add (d : ℕ) (f g : L1Coeff ℕ) :
    seqLowCut d (f + g) = seqLowCut d f + seqLowCut d g := by
  apply lp.ext
  funext n
  simp only [seqLowCut, lp.coeFn_add, Pi.add_apply]
  split_ifs <;> simp

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma seqLowCut_smul (d : ℕ) (c : ℝ) (f : L1Coeff ℕ) :
    seqLowCut d (c • f) = c • seqLowCut d f := by
  apply lp.ext
  funext n
  simp only [seqLowCut, lp.coeFn_smul, Pi.smul_apply]
  split_ifs <;> simp

def seqLowCutCLM (d : ℕ) : L1Coeff ℕ →L[ℝ] L1Coeff ℕ :=
  ({
    toFun := seqLowCut d
    map_add' := seqLowCut_add d
    map_smul' := seqLowCut_smul d
    } : L1Coeff ℕ →ₗ[ℝ] L1Coeff ℕ).mkContinuous 1 (by
      intro f
      simpa using norm_seqLowCut_le d f)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem seqHighShift_seqLowCut (d : ℕ) (f : L1Coeff ℕ) :
    seqHighShift d (seqLowCut d f) = 0 := by
  apply lp.ext
  funext n
  rw [seqHighShift_apply, seqLowCut_apply_of_le d f (by omega)]
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem seqLowShift_highShift_add_lowCut (d : ℕ) (f : L1Coeff ℕ) :
    seqLowShift d (seqHighShift d f) + seqLowCut d f = f := by
  apply lp.ext
  funext n
  by_cases hn : n < d
  · simp [seqLowShift_apply_of_lt d _ hn, seqLowCut_apply_of_lt d _ hn]
  · change seqLowShift d (seqHighShift d f) n + seqLowCut d f n = f n
    rw [seqLowCut_apply_of_le d f (Nat.le_of_not_gt hn), add_zero]
    rw [seqLowShift_apply_of_le d _ (Nat.le_of_not_gt hn)]
    simp [Nat.sub_add_cancel (Nat.le_of_not_gt hn)]

/-- The high-shifted convolution perturbation `S_d C_p` on `ℓ¹(ℕ)`. -/
def seqDivisionPerturbation (d : ℕ) (p : L1Coeff ℕ) : L1Coeff ℕ →L[ℝ] L1Coeff ℕ :=
  seqHighShiftCLM d ∘L convolutionRight p

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqDivisionPerturbation_apply (d : ℕ) (p q : L1Coeff ℕ) :
    seqDivisionPerturbation d p q = seqHighShift d (convolution q p) := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_seqDivisionPerturbation_le (d : ℕ) (p : L1Coeff ℕ) :
    ‖seqDivisionPerturbation d p‖ ≤ ‖p‖ := by
  calc
    ‖seqDivisionPerturbation d p‖ ≤ ‖seqHighShiftCLM d‖ * ‖convolutionRight p‖ :=
      ContinuousLinearMap.opNorm_comp_le _ _
    _ ≤ 1 * ‖p‖ := by
      apply mul_le_mul
      · apply ContinuousLinearMap.opNorm_le_bound _ zero_le_one
        intro f
        change ‖seqHighShift d f‖ ≤ 1 * ‖f‖
        simpa using norm_seqHighShift_le d f
      · exact norm_convolutionRight_le p
      · exact norm_nonneg _
      · exact zero_le_one
    _ = ‖p‖ := one_mul _

end Transformer.AnalyticPreparation
