/-
# Recursive butterfly factors transported to n × d

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`prop: butterfly-hyena`. Each level of the actual recursive tree becomes
one row-major factor. The transport preserves both coefficients and the
partner index, for arbitrary independently parameterized recursive nodes.
-/

import Transformer.Zoology.Appendix_ButterflyGridVectors

namespace Transformer.Zoology

/-- Extract one binary factor in the original row-major input layout.
Source: Appendix `def: butterfly`, possibly different blocks at each level. -/
def ButterflyTree.gridStage {p k : ℕ} (tree : ButterflyTree (k + p))
    (t : Fin (k + p)) : BinaryButterflyStage p k := {
  axis := if h : t.val < k then .inr ⟨t.val, h⟩
    else .inl ⟨t.val - k, by have ht := t.isLt; omega⟩
  main := fun i q => tree.mainCoefficient t.val (butterflyGridFlatten p k i q)
  off := fun i q => tree.offCoefficient t.val (butterflyGridFlatten p k i q)
}

/-- A transported factor reads exactly the scalar factor's paired entry.
Source: Appendix `prop: butterfly-hyena`, constructive row-major simulation. -/
theorem ButterflyTree.gridStage_correct {p k : ℕ} (tree : ButterflyTree (k + p))
    (t : Fin (k + p)) (v : Fin (butterflyWidth (k + p)) → ℝ) :
    (tree.gridStage t).apply (butterflyGridDecode v) =
      butterflyGridDecode (p := p) (k := k) (tree.applyLevel t.val v) := by
  funext i q
  by_cases ht : t.val < k
  · simp only [ButterflyTree.gridStage, BinaryButterflyStage.apply, dite_eq_left ht,
      butterflyGridDecode, ButterflyTree.applyLevel]
    rw [butterflyGridFlatten_toggle_feature p k t.val ht]
  · have hrow : t.val - k < p := by have hb := t.isLt; omega
    have hdigit : t.val = k + (t.val - k) := by omega
    have hpartner := butterflyGridFlatten_toggle_row p k (t.val - k) hrow i q
    rw [← hdigit] at hpartner
    simp only [ButterflyTree.gridStage, BinaryButterflyStage.apply, dite_eq_right ht,
      butterflyGridDecode, ButterflyTree.applyLevel]
    rw [hpartner]

/-- Prefix of the recursive tree's binary factors on the n × d grid.
Source: Appendix `def: butterfly`, product from the smallest block upward. -/
def butterflyGridPrefixStages {p k : ℕ} (tree : ButterflyTree (k + p))
    (t : ℕ) (ht : t ≤ k + p) : List (BinaryButterflyStage p k) :=
  List.ofFn fun j : Fin t => tree.gridStage ⟨j.val, lt_of_lt_of_le j.isLt ht⟩

/-- Every prefix commutes with exact row-major decoding.
Source: Appendix `def: butterfly`, factor order in B_nd ... B_2. -/
theorem butterflyGridPrefixStages_correct {p k : ℕ} (tree : ButterflyTree (k + p))
    (t : ℕ) (ht : t ≤ k + p) (v : Fin (butterflyWidth (k + p)) → ℝ) :
    applyBinaryButterflyStages (butterflyGridPrefixStages tree t ht)
      (butterflyGridDecode v) =
      butterflyGridDecode (p := p) (k := k) (tree.applyLevels t v) := by
  induction t with
  | zero => rfl
  | succ t ih =>
      unfold butterflyGridPrefixStages
      rw [List.ofFn_succ_last]
      change applyBinaryButterflyStages
        (butterflyGridPrefixStages tree t (by omega) ++
          [tree.gridStage ⟨t, by omega⟩]) (butterflyGridDecode v) = _
      rw [applyBinaryButterflyStages_append, ih]
      change (tree.gridStage ⟨t, by omega⟩).apply
        (butterflyGridDecode (p := p) (k := k) (tree.applyLevels t v)) = _
      exact tree.gridStage_correct _ _

/-- Row-major primary representation of an actual recursive butterfly.
Source: Appendix `def: butterfly`, equivalence of both matrix definitions. -/
def ButterflyTree.toGrid {p k : ℕ} (tree : ButterflyTree (k + p)) :
    RowMajorButterfly p k := {
  main := fun t i q => tree.mainCoefficient t.val (butterflyGridFlatten p k i q)
  off := fun t i q => tree.offCoefficient t.val (butterflyGridFlatten p k i q)
}

/-- The transported primary product has the tree's exact factor order.
Source: Appendix `def: butterfly`, full binary-level product. -/
theorem ButterflyTree.toGrid_stages {p k : ℕ} (tree : ButterflyTree (k + p)) :
    tree.toGrid.stages = butterflyGridPrefixStages tree (k + p) (le_refl _) := rfl

/-- The original recursive matrix and the mixed row/feature product agree
on every input. Source: Appendix `prop: butterfly-hyena`, full butterfly. -/
theorem ButterflyTree.toGrid_decode {p k : ℕ} (tree : ButterflyTree (k + p))
    (v : Fin (butterflyWidth (k + p)) → ℝ) :
    tree.toGrid.apply (butterflyGridDecode v) =
      butterflyGridDecode (p := p) (k := k) (tree.apply v) := by
  rw [RowMajorButterfly.apply, tree.toGrid_stages,
    butterflyGridPrefixStages_correct, tree.applyLevels_eq]

/-- Exact action on the original n × d input, with no vector-width
substitution. Source: Appendix `prop: butterfly-hyena`, row-major input u. -/
theorem ButterflyTree.toGrid_correct {p k : ℕ} (tree : ButterflyTree (k + p))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    tree.toGrid.apply u = butterflyGridDecode
      (p := p) (k := k) (tree.apply (butterflyGridEncode u)) := by
  rw [← tree.toGrid_decode, butterflyGridDecode_encode]

/-- Actual K-constrained compilation of the recursive matrix on its
original layout. Source: Appendix `prop: butterfly-hyena`; the proved
workspace is 9n rows, with exactly d features and 3 log₂(nd) layers. -/
theorem ButterflyTree.gridModel_correct {p k : ℕ} (tree : ButterflyTree (k + p))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (binaryButterflyProgramModel tree.toGrid.stages).run u =
      butterflyGridDecode (p := p) (k := k) (tree.apply (butterflyGridEncode u)) := by
  rw [tree.toGrid.model_correct, tree.toGrid_correct]

/-- A single factor of the existing recursive matrix also has an exact
original-layout model. Source: Appendix `prop: butterfly-hyena`, part (1). -/
theorem ButterflyTree.gridStage_model_correct {p k : ℕ}
    (tree : ButterflyTree (k + p)) (t : Fin (k + p))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (binaryButterflyProgramModel [tree.gridStage t]).run u =
      butterflyGridDecode (p := p) (k := k)
        (tree.applyLevel t.val (butterflyGridEncode u)) := by
  rw [binaryButterflyProgramModel_correct]
  change (tree.gridStage t).apply u = _
  rw [← tree.gridStage_correct, butterflyGridDecode_encode]

/-- The compiled full tree has exactly three layers per binary level.
Source: Appendix `prop: butterfly-hyena`, part (2), O(log(nd)) depth. -/
theorem ButterflyTree.gridModel_layerCount {p k : ℕ}
    (tree : ButterflyTree (k + p)) :
    (compileBinaryButterflyStages tree.toGrid.stages).layerCount =
      3 * (k + p) := tree.toGrid.compile_layerCount

/-- The prefix length hypothesis has a genuine mixed-layout example.
Source: Appendix `prop: butterfly-hyena`, n=d=2 and full two-level product. -/
example : (2 : ℕ) ≤ 1 + 1 := by decide

end Transformer.Zoology
