/-
# Coyote realization of primary-form kaleidoscope products

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope` and
`lmm: kaleido-coyote`. Butterfly operands here use the manuscript's
primary factor-product representation, on the original row-major layout.
Transpose and all scalar matrix-vector operations are proved. The common
workspace is 9n rows and d features, with exact depth and scalar storage.
-/

import Transformer.Zoology.Appendix_ButterflyProgramMatrix

open scoped BigOperators

namespace Transformer.Zoology

/-- A BB* factor represented by two primary-form butterfly products.
Source: Appendix `def: kaleidoscope`; over the reals, star is transpose. -/
structure RowMajorBBStar (p k : ℕ) where
  left : RowMajorButterfly p k
  right : RowMajorButterfly p k

/-- B_left times the ordinary transpose matrix of B_right.
Source: Appendix `def: kaleidoscope`, exact BB* matrix semantics. -/
def RowMajorBBStar.apply {p k : ℕ} (f : RowMajorBBStar p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  f.left.apply (fun i q => ∑ input : ButterflyGrid p k,
    binaryButterflyStagesMatrix f.right.stages input (i, q) * u input.1 input.2)

/-- Factor order for one BB* product.
Source: Appendix `lmm: kaleido-coyote`, 2 log(nd) butterfly factors. -/
def RowMajorBBStar.stages {p k : ℕ} (f : RowMajorBBStar p k) :
    List (BinaryButterflyStage p k) :=
  transposeBinaryButterflyStages f.right.stages ++ f.left.stages

/-- The factor list computes the true BB* matrix-vector product.
Source: Appendix `def: kaleidoscope` and `lmm: kaleido-coyote`. -/
theorem RowMajorBBStar.stages_correct {p k : ℕ} (f : RowMajorBBStar p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    applyBinaryButterflyStages f.stages u = f.apply u := by
  unfold RowMajorBBStar.stages
  rw [applyBinaryButterflyStages_append]
  have hright : applyBinaryButterflyStages (transposeBinaryButterflyStages f.right.stages) u =
      fun i q => ∑ input : ButterflyGrid p k,
        binaryButterflyStagesMatrix f.right.stages input (i, q) * u input.1 input.2 := by
    funext i q
    rw [applyBinaryButterflyStages_eq_matrix _ u (i, q)]
    simp only [binaryButterflyStagesMatrix_transpose]
  rw [hright]
  rfl

/-- There are exactly two log(nd) factors in each BB* element.
Source: Appendix `lmm: kaleido-coyote`, factor-count argument. -/
theorem RowMajorBBStar.stages_length {p k : ℕ} (f : RowMajorBBStar p k) :
    f.stages.length = 2 * (k + p) := by
  simp [RowMajorBBStar.stages, transposeBinaryButterflyStages,
    RowMajorButterfly.stages_length]
  omega

/-- Width-w product in the primary-form kaleidoscope hierarchy.
Source: Appendix `def: kaleidoscope`, class (BB*)^w. -/
structure RowMajorKaleidoscope (p k : ℕ) where
  factors : List (RowMajorBBStar p k)

/-- The actual number of BB* factors.
Source: Appendix `def: kaleidoscope`, hierarchy width. -/
def RowMajorKaleidoscope.width {p k : ℕ} (K : RowMajorKaleidoscope p k) : ℕ :=
  K.factors.length

/-- Execute the complete kaleidoscope product in its stored order.
Source: Appendix `def: kaleidoscope`, M_w ... M_1. -/
def RowMajorKaleidoscope.apply {p k : ℕ} (K : RowMajorKaleidoscope p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  K.factors.foldl (fun state f => f.apply state) u

/-- All binary factors of the stored matrix product.
Source: Appendix `lmm: kaleido-coyote`, stacking all BB* factors. -/
def RowMajorKaleidoscope.stages {p k : ℕ} (K : RowMajorKaleidoscope p k) :
    List (BinaryButterflyStage p k) :=
  K.factors.flatMap RowMajorBBStar.stages

/-- Flattening the factor lists preserves the exact matrix action.
Source: Appendix `lmm: kaleido-coyote` and `lem: stacking-layers`. -/
theorem RowMajorKaleidoscope.stages_correct {p k : ℕ} (K : RowMajorKaleidoscope p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    applyBinaryButterflyStages K.stages u = K.apply u := by
  rcases K with ⟨fs⟩
  induction fs generalizing u with
  | nil => rfl
  | cons f fs ih =>
      simp only [RowMajorKaleidoscope.stages] at ih
      change applyBinaryButterflyStages (f.stages ++
        (fs.flatMap RowMajorBBStar.stages)) u = _
      rw [applyBinaryButterflyStages_append, f.stages_correct, ih]
      rfl

/-- A width-w product contains exactly 2w log(nd) binary factors.
Source: Appendix `lmm: kaleido-coyote`, depth derivation. -/
theorem RowMajorKaleidoscope.stages_length {p k : ℕ} (K : RowMajorKaleidoscope p k) :
    K.stages.length = 2 * K.width * (k + p) := by
  rcases K with ⟨fs⟩
  induction fs with
  | nil => simp [RowMajorKaleidoscope.stages, RowMajorKaleidoscope.width]
  | cons f fs ih =>
      simp only [RowMajorKaleidoscope.stages] at ih
      change (f.stages ++ fs.flatMap RowMajorBBStar.stages).length = _
      rw [List.length_append, f.stages_length, ih]
      simp only [RowMajorKaleidoscope.width, List.length_cons]
      ring

/-- Exact compiled depth, with actual K weights in every Coyote layer.
Source: Appendix `lmm: kaleido-coyote`, O(w log(nd)) depth. -/
theorem RowMajorKaleidoscope.compile_layerCount {p k : ℕ} (K : RowMajorKaleidoscope p k) :
    (compileBinaryButterflyStages K.stages).layerCount = 6 * K.width * (k + p) := by
  rw [compileBinaryButterflyStages_layerCount, K.stages_length]
  ring

/-- The externally shaped compiled model realizes the full matrix product.
Source: Appendix `lmm: kaleido-coyote`. The proved sequence workspace is
9n rather than the source's exact n; feature width remains exactly d. -/
theorem RowMajorKaleidoscope.model_correct {p k : ℕ} (K : RowMajorKaleidoscope p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (binaryButterflyProgramModel K.stages).run u = K.apply u := by
  rw [binaryButterflyProgramModel_correct, K.stages_correct]

end Transformer.Zoology
