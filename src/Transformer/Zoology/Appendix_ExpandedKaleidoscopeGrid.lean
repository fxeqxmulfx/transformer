/-
# Expanded recursive K matrices compiled on n × d

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope` and
`lmm: kaleido-coyote`. The existing ExpandedKaleidoscope type now connects
to the verified mixed-layout compiler, including scalar zero padding and
cropping. For expansion 2^e, workspace is 9·2^e·n rows and exactly d features.
The source's exact 2^e·n-row bound for d ≥ 2 is proved by the separate
in-place compiler in Appendix_ExactKaleidoscopeCoyote.
-/

import Transformer.Zoology.Appendix_ButterflyGridPadding

namespace Transformer.Zoology

/-- Transport the existing expanded recursive matrix to its original
row/feature layout. Source: Appendix `def: kaleidoscope`, S E Sᵀ. -/
def ExpandedKaleidoscope.toGrid {p k e : ℕ} (K : ExpandedKaleidoscope (k + p) e) :
    ExpandedRowMajorKaleidoscope p k e := {
  inner := (K.inner.cast (Nat.add_assoc k p e)).toGrid
}

/-- The expansion and transport retain the original hierarchy width.
Source: Appendix `def: kaleidoscope`, unchanged number of BB* factors. -/
theorem ExpandedKaleidoscope.toGrid_inner_width {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) : K.toGrid.inner.width = K.inner.width := by
  change (K.inner.cast (Nat.add_assoc k p e)).toGrid.width = K.inner.width
  rw [Kaleidoscope.toGrid_width, Kaleidoscope.cast_width]

/-- Padding, the whole expanded matrix, and cropping all commute with
row-major decoding. Source: Appendix `def: kaleidoscope`, full expanded class. -/
theorem ExpandedKaleidoscope.toGrid_decode {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e)
    (v : Fin (butterflyWidth (k + p)) → ℝ) :
    K.toGrid.apply (butterflyGridDecode v) =
      butterflyGridDecode (p := p) (k := k) (K.apply v) := by
  change cropSequence (butterflyWidth_le_expanded p e) (le_refl _)
    ((K.inner.cast (Nat.add_assoc k p e)).toGrid.apply
      (padSequence (butterflyGridDecode v))) = _
  rw [butterflyGrid_pad_decode, Kaleidoscope.toGrid_decode,
    Kaleidoscope.cast_apply, butterflyGrid_crop_decode]
  rfl

/-- The original expanded matrix acts exactly on every n × d input.
Source: Appendix `lmm: kaleido-coyote`, functional row-major component. -/
theorem ExpandedKaleidoscope.toGrid_correct {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    K.toGrid.apply u = butterflyGridDecode
      (p := p) (k := k) (K.apply (butterflyGridEncode u)) := by
  rw [← K.toGrid_decode, butterflyGridDecode_encode]

/-- Actual compiled model for the existing recursive expanded class.
Source: Appendix `lmm: kaleido-coyote`; explicit workspace 9·2^e·n,
unchanged feature width d, and K-constrained projections throughout. -/
def ExpandedKaleidoscope.gridModel {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) :
    PaddedCyclicCoyoteModel (butterflyWidth p) (butterflyWidth k) := K.toGrid.model

/-- The compiled model realizes the existing matrix with its true scalar
semantics on the original external layout. Source: Appendix `lmm: kaleido-coyote`;
workspace is ninefold the expanded length, rather than its exact source value. -/
theorem ExpandedKaleidoscope.gridModel_correct {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    K.gridModel.run u = butterflyGridDecode
      (p := p) (k := k) (K.apply (butterflyGridEncode u)) := by
  rw [ExpandedKaleidoscope.gridModel, K.toGrid.model_correct, K.toGrid_correct]

/-- Exact depth is measured in the original recursive matrix's width.
Source: Appendix `lmm: kaleido-coyote`, O(w log(2^e nd)). -/
theorem ExpandedKaleidoscope.gridModel_layerCount {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) :
    K.gridModel.network.layerCount = 6 * K.inner.width * (k + (p + e)) := by
  rw [ExpandedKaleidoscope.gridModel, K.toGrid.model_layerCount, K.toGrid_inner_width]

/-- The depth is exactly six times width times the expanded binary logarithm.
Source: Appendix `lmm: kaleido-coyote`, logarithmic depth component. -/
theorem ExpandedKaleidoscope.gridModel_logDepth {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) :
    K.gridModel.network.layerCount = 6 * K.inner.width *
      Nat.log 2 (butterflyWidth (p + e) * butterflyWidth k) := by
  rw [ExpandedKaleidoscope.gridModel, K.toGrid.model_logDepth, K.toGrid_inner_width]

/-- The original d features are retained throughout the compiled network.
Source: Appendix `lmm: kaleido-coyote`; the exact proved workspace is 9en,
not the paper's unproved exact en, with expansion factor 2^e. -/
theorem ExpandedKaleidoscope.gridModel_dimensions {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) :
    K.gridModel.innerLength = 9 * butterflyWidth (p + e) ∧
      K.gridModel.innerWidth = butterflyWidth k := K.toGrid.model_dimensions

/-- Every stored weight is an actual width-one K matrix with expansion two.
Source: Appendix `def: W-kmat`, projection restriction in the source model. -/
theorem ExpandedKaleidoscope.gridCompile_weightWidths {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) :
    ((compileBinaryButterflyStages K.toGrid.inner.stages).layers.map fun layer =>
      layer.weight.inner.width) =
        List.replicate (6 * K.inner.width * (k + (p + e))) 1 := by
  rw [K.toGrid.compile_weightWidths, K.toGrid_inner_width]

/-- Exact scalar storage is measured in the original recursive object's
width, including all K weights, filters and biases. Source: Appendix
`prop: single-baseconv` and `lmm: kaleido-coyote`. -/
theorem ExpandedKaleidoscope.gridCompile_parameterCount {p k e : ℕ}
    (K : ExpandedKaleidoscope (k + p) e) :
    (compileBinaryButterflyStages K.toGrid.inner.stages).parameterCount =
      (2 * K.inner.width * (k + (p + e))) *
        ((81 * butterflyWidth (p + e) + 24 * (k + 1)) * butterflyWidth k) := by
  rw [K.toGrid.compile_parameterCount, K.toGrid_inner_width]

end Transformer.Zoology
