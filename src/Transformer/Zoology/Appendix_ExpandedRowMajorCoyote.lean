/-
# Expanded primary-form K matrices as externally shaped Coyote models

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope` and
`lmm: kaleido-coyote`. Expansion 2^e enlarges the sequence dimension, while
the feature dimension is unchanged. The construction proves workspace
9·2^e·n, rather than the source lemma's exact 2^e·n. Its exact depth is
6w log₂(2^e nd), and all layer weights are actual K representations.
-/

import Transformer.Zoology.Appendix_RowMajorKaleidoscope

namespace Transformer.Zoology

/-- Upper-left submatrix of an expanded primary-form K product.
Source: Appendix `def: kaleidoscope`, S E Sᵀ, expansion 2^e. -/
structure ExpandedRowMajorKaleidoscope (p k e : ℕ) where
  inner : RowMajorKaleidoscope (p + e) k

/-- Pad, apply the larger matrix, and retain the original coordinates.
Source: Appendix `def: kaleidoscope`, exact expanded matrix semantics. -/
def ExpandedRowMajorKaleidoscope.apply {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  cropSequence (butterflyWidth_le_expanded p e) (le_refl _)
    (K.inner.apply (padSequence u))

/-- Compile the expanded K matrix with one common ninefold workspace.
Source: Appendix `lmm: kaleido-coyote`, explicit larger workspace.
The paper states innerLength = 2^e n; this construction proves 9·2^e n.
Feature width d and the O(w log(2^e nd)) depth are retained; the source's
exact inner-length claim for d ≥ 2 is established by the separate
in-place compiler in Appendix_ExactKaleidoscopeCoyote. -/
def ExpandedRowMajorKaleidoscope.model {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e) :
    PaddedCyclicCoyoteModel (butterflyWidth p) (butterflyWidth k) := {
  innerLength := 9 * butterflyWidth (p + e)
  innerWidth := butterflyWidth k
  lengthBound := le_trans (butterflyWidth_le_expanded p e) (by omega)
  widthBound := le_refl _
  network := (compileBinaryButterflyStages K.inner.stages).toNetwork
}

/-- Direct padding into the workspace equals padding first into the
expanded matrix and then into its workspace.
Source: Appendix `def: kaleidoscope`, nested zero embeddings. -/
theorem expandedRowMajor_workspace_pad {p k e : ℕ}
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (padSequence u : RealSequence (9 * butterflyWidth (p + e)) (butterflyWidth k)) =
      shiftSumPad (padSequence u :
        RealSequence (butterflyWidth (p + e)) (butterflyWidth k)) := by
  funext j q
  have hwidth := butterflyWidth_le_expanded p e
  by_cases hj : j.val < butterflyWidth (p + e)
  · simp [padSequence, shiftSumPad, hj, q.isLt]
  · have hout : ¬ j.val < butterflyWidth p := by omega
    simp [padSequence, shiftSumPad, hj, hout]

/-- The external model computes the expanded matrix's upper-left action
on every input. Source: Appendix `lmm: kaleido-coyote`, construction
using 9·2^e·n workspace rows rather than exactly 2^e·n. -/
theorem ExpandedRowMajorKaleidoscope.model_correct {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    K.model.run u = K.apply u := by
  unfold PaddedCyclicCoyoteModel.run ExpandedRowMajorKaleidoscope.model
  rw [expandedRowMajor_workspace_pad,
    ((compileBinaryButterflyStages K.inner.stages).toNetwork_correct _).1,
    compileBinaryButterflyStages_correct, K.inner.stages_correct]
  funext i q
  have hi : i.val < butterflyWidth (p + e) :=
    lt_of_lt_of_le i.isLt (butterflyWidth_le_expanded p e)
  simp [ExpandedRowMajorKaleidoscope.apply, cropSequence, shiftSumPad, hi]

/-- Exact depth for every hierarchy width and power-of-two expansion.
Source: Appendix `lmm: kaleido-coyote`, logarithmic depth component. -/
theorem ExpandedRowMajorKaleidoscope.model_layerCount {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e) :
    K.model.network.layerCount = 6 * K.inner.width * (k + (p + e)) := by
  rw [ExpandedRowMajorKaleidoscope.model,
    ((compileBinaryButterflyStages K.inner.stages).toNetwork_correct
      (fun _ _ => 0)).2, K.inner.compile_layerCount]

/-- Exact stored scalars of the K-constrained compiled stack.
Source: Appendix `prop: single-baseconv`, actual K/filter/bias storage. -/
theorem ExpandedRowMajorKaleidoscope.compile_parameterCount {p k e : ℕ}
    (K : ExpandedRowMajorKaleidoscope p k e) :
    (compileBinaryButterflyStages K.inner.stages).parameterCount =
      (2 * K.inner.width * (k + (p + e))) *
        ((81 * butterflyWidth (p + e) + 24 * (k + 1)) * butterflyWidth k) := by
  rw [compileBinaryButterflyStages_parameterCount, K.inner.stages_length]

end Transformer.Zoology
