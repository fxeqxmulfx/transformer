/-
# Real analytic preparation: SequenceEmbedding

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.SequenceConvolution
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Analysis.Analytic.Constructions

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

namespace L1Coeff

noncomputable def embedFun (e : I ↪ J) (f : L1Coeff I) (j : J) : ℝ := by
  classical
  exact if h : ∃ i, e i = j then f h.choose else 0


/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma embedFun_apply_self (e : I ↪ J) (f : L1Coeff I) (i : I) :
    embedFun e f (e i) = f i := by
  classical
  let h : ∃ k, e k = e i := ⟨i, rfl⟩
  rw [embedFun, dite_eq_left h]
  have hk : e h.choose = e i := h.choose_spec
  rw [e.injective hk]


/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma embedFun_apply_of_not_mem (e : I ↪ J) (f : L1Coeff I) (j : J)
    (hj : j ∉ Set.range e) : embedFun e f j = 0 := by
  classical
  rw [embedFun, dite_eq_right]
  simpa only [Set.mem_range] using hj


def embed (e : I ↪ J) (f : L1Coeff I) : L1Coeff J :=
  ⟨embedFun e f, by
    classical
    apply memℓp_gen
    apply (Function.Injective.summable_iff e.injective (fun j hj ↦ ?_)).mp
    · simpa [Function.comp_def] using L1Coeff.summable_norm f
    · simp [embedFun_apply_of_not_mem e f j hj]⟩


/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma embed_apply_self (e : I ↪ J) (f : L1Coeff I) (i : I) :
    embed e f (e i) = f i := embedFun_apply_self e f i


/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma embed_apply_of_not_mem (e : I ↪ J) (f : L1Coeff I) (j : J)
    (hj : j ∉ Set.range e) : embed e f j = 0 := embedFun_apply_of_not_mem e f j hj

end L1Coeff

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma convolution_comm (f g : L1Coeff A) : convolution f g = convolution g f := by
  apply lp.ext
  funext n
  rw [convolution_apply, convolution_apply]
  refine Finset.sum_bij (fun kl _ ↦ kl.swap) ?_ ?_ ?_ ?_
  · intro kl hkl
    simpa [add_comm] using hkl
  · intro kl₁ h₁ kl₂ h₂ h
    exact Prod.swap_injective h
  · intro kl hkl
    exact ⟨kl.swap, by simpa [add_comm] using hkl, by simp⟩
  · intro kl hkl
    simp [mul_comm]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma convolution_add_right (f g₁ g₂ : L1Coeff A) :
    convolution f (g₁ + g₂) = convolution f g₁ + convolution f g₂ := by
  rw [convolution_comm f, convolution_add_left, convolution_comm g₁, convolution_comm g₂]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma convolution_smul_right (c : ℝ) (f g : L1Coeff A) :
    convolution f (c • g) = c • convolution f g := by
  rw [convolution_comm f, convolution_smul_left, convolution_comm g]

/-- Right convolution depends continuously and linearly on the coefficient family. -/
def convolutionRightMap :
    L1Coeff A →L[ℝ] (L1Coeff A →L[ℝ] L1Coeff A) :=
  ({
    toFun := convolutionRight
    map_add' := by
      intro p₁ p₂
      apply ContinuousLinearMap.ext
      intro q
      exact convolution_add_right q p₁ p₂
    map_smul' := by
      intro c p
      apply ContinuousLinearMap.ext
      intro q
      exact convolution_smul_right c q p
    } : L1Coeff A →ₗ[ℝ]
      (L1Coeff A →L[ℝ] L1Coeff A)).mkContinuous 1 (by
        intro p
        simpa using norm_convolutionRight_le p)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma convolutionRightMap_apply (p : L1Coeff A) :
    convolutionRightMap p = convolutionRight p := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_inverseOneAdd_apply
    {X E : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (K : X →L[ℝ] (E →L[ℝ] E)) (b : X →L[ℝ] E) (x : X) (hK : ‖K x‖ < 1) :
    AnalyticAt ℝ (fun y ↦ Ring.inverse (1 + K y) (b y)) x := by
  let op : X → (E →L[ℝ] E) := fun y ↦ 1 + K y
  have hop : AnalyticAt ℝ op x := analyticAt_const.add (K.analyticAt x)
  let hneg : ‖-(K x)‖ < 1 := by simpa using hK
  let z := Units.oneSub (-(K x)) hneg
  have hz : (z : E →L[ℝ] E) = op x := by simp [z, op, sub_eq_add_neg]
  have hinvAt : AnalyticAt ℝ Ring.inverse (op x) := by
    rw [← hz]
    exact analyticAt_inverse z
  have hinv : AnalyticAt ℝ (fun y ↦ Ring.inverse (op y)) x := by
    simpa [Function.comp_def] using hinvAt.comp hop
  have happ := (ContinuousLinearMap.apply ℝ E).analyticAt_bilinear
    (b x, Ring.inverse (op x))
  have hpair := (b.analyticAt x).prod hinv
  have hcomp := AnalyticAt.comp (x := x) happ hpair
  simpa [op, Function.comp_def] using hcomp

/-- Delete the first `d` coefficients of a one-variable `ℓ¹` sequence. -/
def seqHighShift (d : ℕ) (f : L1Coeff ℕ) : L1Coeff ℕ :=
  ⟨fun n ↦ f (n + d), by
    apply memℓp_gen
    simpa [Function.comp_def] using
      (L1Coeff.summable_norm f).comp_injective (fun _ _ h ↦ Nat.add_right_cancel h)⟩

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma seqHighShift_apply (d : ℕ) (f : L1Coeff ℕ) (n : ℕ) :
    seqHighShift d f n = f (n + d) := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_seqHighShift_le (d : ℕ) (f : L1Coeff ℕ) : ‖seqHighShift d f‖ ≤ ‖f‖ := by
  rw [L1Coeff.norm_eq_tsum_norm, L1Coeff.norm_eq_tsum_norm]
  apply (L1Coeff.summable_norm (seqHighShift d f)).tsum_le_tsum_of_inj (fun n ↦ n + d)
    (fun _ _ h ↦ Nat.add_right_cancel h)
  · intro n hn
    exact norm_nonneg _
  · intro n
    exact le_rfl
  · exact L1Coeff.summable_norm f

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma seqHighShift_add (d : ℕ) (f g : L1Coeff ℕ) :
    seqHighShift d (f + g) = seqHighShift d f + seqHighShift d g := by
  apply lp.ext
  funext n
  rfl

end Transformer.AnalyticPreparation
