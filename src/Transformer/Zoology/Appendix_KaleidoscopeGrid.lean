/-
# Recursive kaleidoscope matrices on the original n × d layout

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope` and
`lmm: kaleido-coyote`. Reindexing the ordinary scalar matrix proves that
the existing BB* and hierarchy objects have exactly the primary grid
semantics, including genuine transpose. The verified compiler is then
applicable to those existing objects without an assumed bridge.
-/

import Transformer.Zoology.Appendix_ButterflyGridStages

open scoped BigOperators

namespace Transformer.Zoology

/-- Grid matrix entries are precisely the original recursive matrix's
entries in row-major coordinates. Source: Appendix `def: butterfly`. -/
theorem ButterflyTree.toGrid_matrix {p k : ℕ} (tree : ButterflyTree (k + p))
    (out input : ButterflyGrid p k) :
    binaryButterflyStagesMatrix tree.toGrid.stages out input =
      tree.matrix (butterflyGridFlatten p k out.1 out.2)
        (butterflyGridFlatten p k input.1 input.2) := by
  change tree.toGrid.apply (butterflyGridBasis input) out.1 out.2 = _
  rw [tree.toGrid_correct, butterflyGridEncode_basis]
  rfl

/-- Transport both recursive operands of an existing BB* factor.
Source: Appendix `def: kaleidoscope`, B_left B_rightᵀ. -/
def BBStar.toGrid {p k : ℕ} (f : BBStar (k + p)) : RowMajorBBStar p k := {
  left := f.left.toGrid
  right := f.right.toGrid
}

/-- Ordinary scalar transpose commutes with row-major transport.
Source: Appendix `def: kaleidoscope`, the actual B_rightᵀ operator. -/
theorem ButterflyTree.toGrid_transpose {p k : ℕ} (tree : ButterflyTree (k + p))
    (v : Fin (butterflyWidth (k + p)) → ℝ) :
    (fun i q => ∑ input : ButterflyGrid p k,
      binaryButterflyStagesMatrix tree.toGrid.stages input (i, q) *
        butterflyGridDecode v input.1 input.2) =
      butterflyGridDecode (p := p) (k := k)
        (fun j => ∑ input, tree.matrix input j * v input) := by
  funext i q
  simp only [tree.toGrid_matrix, butterflyGridDecode]
  exact (butterflyGridEquiv p k).sum_comp
    (fun input => tree.matrix input (butterflyGridFlatten p k i q) * v input)

/-- The transported BB* factor is exactly the original scalar factor.
Source: Appendix `def: kaleidoscope`, not a merely formal factor reversal. -/
theorem BBStar.toGrid_decode {p k : ℕ} (f : BBStar (k + p))
    (v : Fin (butterflyWidth (k + p)) → ℝ) :
    f.toGrid.apply (butterflyGridDecode v) =
      butterflyGridDecode (p := p) (k := k) (f.apply v) := by
  change f.left.toGrid.apply (fun i q => ∑ input : ButterflyGrid p k,
    binaryButterflyStagesMatrix f.right.toGrid.stages input (i, q) *
      butterflyGridDecode v input.1 input.2) = _
  rw [f.right.toGrid_transpose, f.left.toGrid_decode]
  rfl

/-- Transport every existing recursive BB* factor in its stored order.
Source: Appendix `def: kaleidoscope`, the complete width-w class. -/
def Kaleidoscope.toGrid {p k : ℕ} (K : Kaleidoscope (k + p)) :
    RowMajorKaleidoscope p k := {
  factors := K.factors.map BBStar.toGrid
}

/-- Reindexing preserves hierarchy width exactly.
Source: Appendix `def: kaleidoscope`, w is the number of BB* factors. -/
theorem Kaleidoscope.toGrid_width {p k : ℕ} (K : Kaleidoscope (k + p)) :
    K.toGrid.width = K.width := by
  simp [Kaleidoscope.toGrid, RowMajorKaleidoscope.width, Kaleidoscope.width]

/-- All factors commute with the same exact row-major decoding.
Source: Appendix `lmm: kaleido-coyote`, complete hierarchy product. -/
theorem Kaleidoscope.toGrid_decode {p k : ℕ} (K : Kaleidoscope (k + p))
    (v : Fin (butterflyWidth (k + p)) → ℝ) :
    K.toGrid.apply (butterflyGridDecode v) =
      butterflyGridDecode (p := p) (k := k) (K.apply v) := by
  rcases K with ⟨fs⟩
  induction fs generalizing v with
  | nil => rfl
  | cons f fs ih =>
      change ({factors := fs.map BBStar.toGrid} : RowMajorKaleidoscope p k).apply
        (f.toGrid.apply (butterflyGridDecode v)) =
          butterflyGridDecode (p := p) (k := k)
            (({factors := fs} : Kaleidoscope (k + p)).apply (f.apply v))
      rw [f.toGrid_decode]
      exact ih (f.apply v)

/-- The existing recursive hierarchy acts on arbitrary original inputs.
Source: Appendix `lmm: kaleido-coyote`, row-major input correspondence. -/
theorem Kaleidoscope.toGrid_correct {p k : ℕ} (K : Kaleidoscope (k + p))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    K.toGrid.apply u = butterflyGridDecode
      (p := p) (k := k) (K.apply (butterflyGridEncode u)) := by
  rw [← K.toGrid_decode, butterflyGridDecode_encode]

/-- An actual K-constrained network compiles every existing hierarchy
matrix on n × d. Source: Appendix `lmm: kaleido-coyote`; the proved bound
uses 9n workspace rows, exactly d features and 6w log₂(nd) layers. -/
theorem Kaleidoscope.gridModel_correct {p k : ℕ} (K : Kaleidoscope (k + p))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (binaryButterflyProgramModel K.toGrid.stages).run u =
      butterflyGridDecode (p := p) (k := k) (K.apply (butterflyGridEncode u)) := by
  rw [K.toGrid.model_correct, K.toGrid_correct]

/-- Exact depth is expressed in the original recursive object's width.
Source: Appendix `lmm: kaleido-coyote`, O(w log(nd)) factor-count bound. -/
theorem Kaleidoscope.gridModel_layerCount {p k : ℕ} (K : Kaleidoscope (k + p)) :
    (compileBinaryButterflyStages K.toGrid.stages).layerCount =
      6 * K.width * (k + p) := by
  rw [K.toGrid.compile_layerCount, K.toGrid_width]

end Transformer.Zoology
