/-
# Products of row-major butterfly factors

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`,
`prop: butterfly-hyena`, and `lem: stacking-layers`. All factors, including
those mixing feature coordinates, share exactly 9n rows and d features.
The construction uses three layers per factor with explicit K weights.
-/

import Transformer.Zoology.Appendix_BinaryButterflyStage

namespace Transformer.Zoology

/-- Execute factors in their listed order.
Source: Appendix `def: butterfly` and `def: kaleidoscope`, matrix products. -/
def applyBinaryButterflyStages {p k : ℕ} (stages : List (BinaryButterflyStage p k))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  stages.foldl (fun state s => s.apply state) u

/-- Stack all factor networks on one common workspace.
Source: Appendix `prop: butterfly-hyena` and `lem: stacking-layers`. -/
def compileBinaryButterflyStages {p k : ℕ} : List (BinaryButterflyStage p k) →
    CyclicKCoyoteNetwork (9 * butterflyWidth p) k 1
  | [] => {layers := []}
  | s :: ss => s.compile.append (compileBinaryButterflyStages ss)

/-- The stacked K network realizes the complete factor product exactly.
Source: Appendix `prop: butterfly-hyena`, constructive composition step. -/
theorem compileBinaryButterflyStages_correct {p k : ℕ}
    (stages : List (BinaryButterflyStage p k))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (compileBinaryButterflyStages stages).run (shiftSumPad u) =
      shiftSumPad (applyBinaryButterflyStages stages u) := by
  induction stages generalizing u with
  | nil => rfl
  | cons s ss ih =>
      rw [compileBinaryButterflyStages, CyclicKCoyoteNetwork.run_append,
        s.compile_correct, ih]
      rfl

/-- Depth is exactly three times the number of factors.
Source: Appendix `prop: butterfly-hyena` and `lem: stacking-layers`. -/
theorem compileBinaryButterflyStages_layerCount {p k : ℕ}
    (stages : List (BinaryButterflyStage p k)) :
    (compileBinaryButterflyStages stages).layerCount = 3 * stages.length := by
  induction stages with
  | nil => rfl
  | cons s ss ih =>
      rw [compileBinaryButterflyStages, CyclicKCoyoteNetwork.layerCount_append,
        s.compile_layerCount, ih, List.length_cons]
      omega

/-- Exact scalar parameter count of the stored K representation.
Source: Appendix `prop: single-baseconv` and `def: W-kmat`. -/
theorem compileBinaryButterflyStages_parameterCount {p k : ℕ}
    (stages : List (BinaryButterflyStage p k)) :
    (compileBinaryButterflyStages stages).parameterCount =
      stages.length * ((81 * butterflyWidth p + 24 * (k + 1)) * butterflyWidth k) := by
  induction stages with
  | nil => simp [compileBinaryButterflyStages, CyclicKCoyoteNetwork.parameterCount]
  | cons s ss ih =>
      rw [compileBinaryButterflyStages, CyclicKCoyoteNetwork.parameterCount_append,
        s.compile_parameterCount, ih, List.length_cons]
      simp only [butterflyWidth]
      ring

/-- Every compiled projection has hierarchy width one, independent of
the number of input factors. Source: Appendix `def: W-kmat`. -/
theorem compileBinaryButterflyStages_weightWidths {p k : ℕ}
    (stages : List (BinaryButterflyStage p k)) :
    ((compileBinaryButterflyStages stages).layers.map fun layer =>
      layer.weight.inner.width) = List.replicate (3 * stages.length) 1 := by
  induction stages with
  | nil => rfl
  | cons s ss ih =>
      change ((s.compile.layers ++ (compileBinaryButterflyStages ss).layers).map
        fun layer => layer.weight.inner.width) = _
      rw [List.map_append, s.compile_weightWidths, ih, List.length_cons]
      rw [show 3 * (ss.length + 1) = 3 + 3 * ss.length by omega,
        List.replicate_add]
      rfl

/-- Padded model retaining the original external and feature dimensions.
Source: Appendix `prop: butterfly-hyena`, explicit resource realization. -/
def binaryButterflyProgramModel {p k : ℕ} (stages : List (BinaryButterflyStage p k)) :
    PaddedCyclicCoyoteModel (butterflyWidth p) (butterflyWidth k) := {
  innerLength := 9 * butterflyWidth p
  innerWidth := butterflyWidth k
  lengthBound := by omega
  widthBound := le_refl _
  network := (compileBinaryButterflyStages stages).toNetwork
}

/-- Cropping the network yields the exact factor product on the input
layout. Source: Appendix `prop: butterfly-hyena`, functional component. -/
theorem binaryButterflyProgramModel_correct {p k : ℕ}
    (stages : List (BinaryButterflyStage p k))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (binaryButterflyProgramModel stages).run u = applyBinaryButterflyStages stages u := by
  have hpad : (padSequence u :
      RealSequence (9 * butterflyWidth p) (butterflyWidth k)) = shiftSumPad u := by
    funext i q
    simp [padSequence, shiftSumPad]
  unfold PaddedCyclicCoyoteModel.run binaryButterflyProgramModel
  rw [hpad, ((compileBinaryButterflyStages stages).toNetwork_correct _).1,
    compileBinaryButterflyStages_correct]
  funext i q
  simp [cropSequence, shiftSumPad]

/-- One coefficient pair for every block size in the paper's primary
product definition, in row-major order. Feature digits precede row digits.
Source: Appendix `def: butterfly`, B_nd ... B_2. -/
structure RowMajorButterfly (p k : ℕ) where
  main : Fin (k + p) → RealSequence (butterflyWidth p) (butterflyWidth k)
  off : Fin (k + p) → RealSequence (butterflyWidth p) (butterflyWidth k)

/-- List the increasing binary block sizes of a row-major butterfly.
Source: Appendix `def: butterfly`, product acts from B_2 upward. -/
def RowMajorButterfly.stages {p k : ℕ} (B : RowMajorButterfly p k) :
    List (BinaryButterflyStage p k) :=
  List.ofFn fun t : Fin (k + p) => {
    axis := if h : t.val < k then .inr ⟨t.val, h⟩
      else .inl ⟨t.val - k, by have ht := t.isLt; omega⟩
    main := B.main t
    off := B.off t
  }

/-- The butterfly product acts by its log(nd) binary factors.
Source: Appendix `def: butterfly`, primary product definition. -/
def RowMajorButterfly.apply {p k : ℕ} (B : RowMajorButterfly p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  applyBinaryButterflyStages B.stages u

/-- One factor is stored for each of the log(nd) block sizes.
Source: Appendix `def: butterfly`, primary product definition. -/
theorem RowMajorButterfly.stages_length {p k : ℕ} (B : RowMajorButterfly p k) :
    B.stages.length = k + p := by
  simp [RowMajorButterfly.stages]

/-- The exact three-log(nd) depth for the primary factor representation.
Source: Appendix `prop: butterfly-hyena`, second component. -/
theorem RowMajorButterfly.compile_layerCount {p k : ℕ} (B : RowMajorButterfly p k) :
    (compileBinaryButterflyStages B.stages).layerCount = 3 * (k + p) := by
  rw [compileBinaryButterflyStages_layerCount]
  simp [RowMajorButterfly.stages]

/-- One external-shape model realizes the whole butterfly product.
Source: Appendix `prop: butterfly-hyena`; uses 9n rows and exactly d features. -/
theorem RowMajorButterfly.model_correct {p k : ℕ} (B : RowMajorButterfly p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (binaryButterflyProgramModel B.stages).run u = B.apply u :=
  binaryButterflyProgramModel_correct B.stages u

end Transformer.Zoology
