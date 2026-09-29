/-
# Identity butterfly matrices and unexpanded butterfly weights

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`,
`def: kaleidoscope`, and `def: W-kmat`. A butterfly matrix itself is a
width-one K matrix by taking the right butterfly to be the identity.
One extra level gives a common expansion type with the zero-weight layers.
-/

import Transformer.Zoology.Appendix_DiagonalLinearK

namespace Transformer.Zoology

/-- Identity factor with ones on its main diagonal and zero off-diagonals.
Source: Appendix `def: butterfly`. -/
def identityButterflyFactor (m : ℕ) : ButterflyFactor m := {
  upperLeft := fun _ => 1
  upperRight := fun _ => 0
  lowerLeft := fun _ => 0
  lowerRight := fun _ => 1
}

/-- The identity factor preserves both halves.
Source: Appendix `def: butterfly`, identity diagonals. -/
theorem identityButterflyFactor_apply {m : ℕ} (u : HalfVector m) :
    (identityButterflyFactor m).apply u = u := by
  funext h t
  fin_cases h <;> simp [identityButterflyFactor, ButterflyFactor.apply]

/-- Identity butterfly tree at every power-of-two dimension.
Source: Appendix `def: butterfly`, recursive definition. -/
def identityButterflyTree : (k : ℕ) → ButterflyTree k
  | 0 => .leaf
  | k + 1 => .node (identityButterflyFactor (butterflyWidth k))
      (identityButterflyTree k) (identityButterflyTree k)

/-- Every identity butterfly tree acts identically on all vectors.
Source: Appendix `def: butterfly`, identity instance. -/
theorem identityButterflyTree_apply (k : ℕ)
    (u : Fin (butterflyWidth k) → ℝ) :
    (identityButterflyTree k).apply u = u := by
  induction k with
  | zero => rfl
  | succ k ih =>
      funext j
      have hj := (finProdFinEquiv (m := 2) (n := butterflyWidth k)).apply_symm_apply j
      generalize hp : finProdFinEquiv.symm j = p at hj
      rcases p with ⟨half, t⟩
      rw [← hj]
      fin_cases half <;>
        simp [identityButterflyTree, ButterflyTree.apply,
          identityButterflyFactor, ButterflyFactor.apply, ih]

/-- The matrix of the identity butterfly is the ordinary Kronecker delta.
Source: Appendix `def: kaleidoscope`, identity right factor. -/
theorem identityButterflyTree_matrix (k : ℕ)
    (i j : Fin (butterflyWidth k)) :
    (identityButterflyTree k).matrix i j = if i = j then 1 else 0 := by
  simp [ButterflyTree.matrix, identityButterflyTree_apply]

/-- Embed a butterfly tree in the upper-left half of a larger butterfly.
Source: Appendix `def: kaleidoscope`, expansion and cropping. -/
def expandedButterflyTree {k : ℕ} (tree : ButterflyTree k) : ButterflyTree (k + 1) :=
  .node (identityButterflyFactor (butterflyWidth k)) tree (identityButterflyTree k)

/-- The expanded butterfly acts as the original tree on the retained half.
Source: Appendix `def: kaleidoscope`, upper-left submatrix. -/
theorem expandedButterflyTree_apply {k : ℕ} (tree : ButterflyTree k)
    (u : Fin (butterflyWidth (k + 1)) → ℝ) (i : Fin (butterflyWidth k)) :
    (expandedButterflyTree tree).apply u (finProdFinEquiv (0, i)) =
      tree.apply (fun j => u (finProdFinEquiv (0, j))) i := by
  simp [expandedButterflyTree, ButterflyTree.apply, identityButterflyFactor,
    ButterflyFactor.apply]

/-- A width-one expanded K representation of any butterfly matrix.
Source: Appendix `def: kaleidoscope`, B times identity transpose. -/
def butterflyTreeK {k : ℕ} (tree : ButterflyTree k) : ExpandedKaleidoscope k 1 := {
  inner := {
    factors := [{
      left := expandedButterflyTree tree
      right := identityButterflyTree (k + 1)
    }]
  }
}

/-- Identity transpose has identity action in the width-one BB* factor.
Source: Appendix `def: kaleidoscope`, BB* with identity right butterfly. -/
theorem butterflyTreeK_BBStar_apply {k : ℕ} (tree : ButterflyTree k)
    (u : Fin (butterflyWidth (k + 1)) → ℝ) :
    ({left := expandedButterflyTree tree, right := identityButterflyTree (k + 1)} :
      BBStar (k + 1)).apply u = (expandedButterflyTree tree).apply u := by
  unfold BBStar.apply
  congr 1
  funext j
  simp [identityButterflyTree_matrix]

/-- The K representation has exactly the original butterfly action.
Source: Appendix `def: W-kmat`, representation of an allowed projection. -/
theorem butterflyTreeK_apply {k : ℕ} (tree : ButterflyTree k)
    (u : Fin (butterflyWidth k) → ℝ) :
    (butterflyTreeK tree).apply u = tree.apply u := by
  change cropButterfly
    (({left := expandedButterflyTree tree, right := identityButterflyTree (k + 1)} :
      BBStar (k + 1)).apply (padButterfly u)) = _
  rw [butterflyTreeK_BBStar_apply]
  have hcrop (i : Fin (butterflyWidth k)) :
      (⟨i.val, lt_of_lt_of_le i.isLt (butterflyWidth_le_expanded k 1)⟩ :
        Fin (butterflyWidth (k + 1))) = finProdFinEquiv (0, i) := by
    apply Fin.ext
    simp [finProdFinEquiv_apply_val]
  funext i
  unfold cropButterfly
  rw [hcrop, expandedButterflyTree_apply]
  congr 1
  funext j
  simp [padButterfly, finProdFinEquiv_apply_val, j.isLt]

/-- The representation stores exactly one BB* factor.
Source: Appendix `def: kaleidoscope`, width one. -/
theorem butterflyTreeK_width {k : ℕ} (tree : ButterflyTree k) :
    (butterflyTreeK tree).inner.width = 1 := rfl

end Transformer.Zoology
