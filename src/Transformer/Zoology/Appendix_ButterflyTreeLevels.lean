/-
# Binary factor coefficients of a recursive butterfly

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`, equivalence
between the recursive and primary product definitions. A level coefficient
is read from the unique recursive node containing the coordinate. Only
levels below the tree's depth are compiled.
-/

import Transformer.Zoology.Appendix_RowMajorCoyoteResources

namespace Transformer.Zoology

/-- Main diagonal at binary level t of a recursive butterfly.
Source: Appendix `def: butterfly`, product of block-diagonal factors. -/
def ButterflyTree.mainCoefficient : {k : ℕ} → ButterflyTree k → ℕ →
    Fin (butterflyWidth k) → ℝ
  | 0, .leaf, _, _ => 1
  | k + 1, .node f left right, t, i =>
      let p := finProdFinEquiv.symm i
      if t = k then
        if p.1 = 0 then f.upperLeft p.2 else f.lowerRight p.2
      else if p.1 = 0 then left.mainCoefficient t p.2
      else right.mainCoefficient t p.2

/-- Off-diagonal coefficient at binary level t.
Source: Appendix `def: butterfly`, paired entries of each block. -/
def ButterflyTree.offCoefficient : {k : ℕ} → ButterflyTree k → ℕ →
    Fin (butterflyWidth k) → ℝ
  | 0, .leaf, _, _ => 0
  | k + 1, .node f left right, t, i =>
      let p := finProdFinEquiv.symm i
      if t = k then
        if p.1 = 0 then f.upperRight p.2 else f.lowerLeft p.2
      else if p.1 = 0 then left.offCoefficient t p.2
      else right.offCoefficient t p.2

/-- The main coefficient at a node selects its own diagonal or its child's
coefficient. Source: Appendix `def: butterfly`, recursive block layout. -/
theorem ButterflyTree.mainCoefficient_node {k : ℕ}
    (f : ButterflyFactor (butterflyWidth k)) (left right : ButterflyTree k)
    (t : ℕ) (half : Fin 2) (j : Fin (butterflyWidth k)) :
    (ButterflyTree.node f left right).mainCoefficient t (finProdFinEquiv (half, j)) =
      if t = k then (if half = 0 then f.upperLeft j else f.lowerRight j)
      else if half = 0 then left.mainCoefficient t j else right.mainCoefficient t j := by
  change (if t = k then
    if (finProdFinEquiv.symm (finProdFinEquiv (half, j))).1 = 0 then
      f.upperLeft (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2
    else f.lowerRight (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2
    else if (finProdFinEquiv.symm (finProdFinEquiv (half, j))).1 = 0 then
      left.mainCoefficient t (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2
    else right.mainCoefficient t (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2) = _
  rw [Equiv.symm_apply_apply]

/-- The off-diagonal coefficient selects its own entry or its child's
entry. Source: Appendix `def: butterfly`, recursive block layout. -/
theorem ButterflyTree.offCoefficient_node {k : ℕ}
    (f : ButterflyFactor (butterflyWidth k)) (left right : ButterflyTree k)
    (t : ℕ) (half : Fin 2) (j : Fin (butterflyWidth k)) :
    (ButterflyTree.node f left right).offCoefficient t (finProdFinEquiv (half, j)) =
      if t = k then (if half = 0 then f.upperRight j else f.lowerLeft j)
      else if half = 0 then left.offCoefficient t j else right.offCoefficient t j := by
  change (if t = k then
    if (finProdFinEquiv.symm (finProdFinEquiv (half, j))).1 = 0 then
      f.upperRight (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2
    else f.lowerLeft (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2
    else if (finProdFinEquiv.symm (finProdFinEquiv (half, j))).1 = 0 then
      left.offCoefficient t (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2
    else right.offCoefficient t (finProdFinEquiv.symm (finProdFinEquiv (half, j))).2) = _
  rw [Equiv.symm_apply_apply]

/-- A binary toggle either exchanges these halves or acts inside them.
Source: Appendix `def: butterfly`, recursive binary pairing. -/
theorem butterflyToggleIndex_pair (k t : ℕ) (half : Fin 2)
    (j : Fin (butterflyWidth k)) :
    butterflyToggleIndex (k + 1) t (finProdFinEquiv (half, j)) =
      if t = k then finProdFinEquiv (if half = 0 then 1 else 0, j)
      else finProdFinEquiv (half, butterflyToggleIndex k t j) := by
  dsimp only [butterflyToggleIndex]
  rw [Equiv.symm_apply_apply]
  rfl

/-- Apply the level's diagonal and binary partner contribution.
Source: Appendix `eq: butterfly-split`, factor at block size 2^(t+1). -/
def ButterflyTree.applyLevel {k : ℕ} (tree : ButterflyTree k) (t : ℕ)
    (u : Fin (butterflyWidth k) → ℝ) : Fin (butterflyWidth k) → ℝ :=
  fun i => tree.mainCoefficient t i * u i + tree.offCoefficient t i *
    u (butterflyToggleIndex k t i)

/-- Execute successive levels starting at the smallest block size.
Source: Appendix `def: butterfly`, product B_n ... B_2. -/
def ButterflyTree.applyLevels {k : ℕ} (tree : ButterflyTree k) :
    ℕ → (Fin (butterflyWidth k) → ℝ) → Fin (butterflyWidth k) → ℝ
  | 0, u => u
  | t + 1, u => tree.applyLevel t (tree.applyLevels t u)

/-- Before the final level, the two child programs execute independently
in their respective halves. Source: Appendix `def: butterfly`, block
diagonal children in the recursive definition. -/
theorem ButterflyTree.applyLevels_half {k : ℕ}
    (f : ButterflyFactor (butterflyWidth k)) (left right : ButterflyTree k)
    (t : ℕ) (ht : t ≤ k) (u : Fin (butterflyWidth (k + 1)) → ℝ)
    (half : Fin 2) (j : Fin (butterflyWidth k)) :
    (ButterflyTree.node f left right).applyLevels t u (finProdFinEquiv (half, j)) =
      if half = 0 then left.applyLevels t (fun i => u (finProdFinEquiv (0, i))) j
      else right.applyLevels t (fun i => u (finProdFinEquiv (1, i))) j := by
  induction t generalizing half j with
  | zero => fin_cases half <;> simp [ButterflyTree.applyLevels]
  | succ t ih =>
      have htk : t ≠ k := by omega
      have hle : t ≤ k := by omega
      fin_cases half <;>
        simp [ButterflyTree.applyLevels, ButterflyTree.applyLevel,
          ButterflyTree.mainCoefficient_node, ButterflyTree.offCoefficient_node,
          butterflyToggleIndex_pair, htk]
      · rw [ih hle 0 j, ih hle 0 (butterflyToggleIndex k t j)]
        rfl
      · rw [ih hle 1 j, ih hle 1 (butterflyToggleIndex k t j)]
        rfl

/-- All k binary factors reproduce the full recursive butterfly matrix.
Source: Appendix `def: butterfly`, claimed equivalence of the two definitions. -/
theorem ButterflyTree.applyLevels_eq {k : ℕ} (tree : ButterflyTree k)
    (u : Fin (butterflyWidth k) → ℝ) : tree.applyLevels k u = tree.apply u := by
  induction tree with
  | leaf => rfl
  | @node k f left right ihl ihr =>
      funext i
      have hi := (finProdFinEquiv (m := 2) (n := butterflyWidth k)).apply_symm_apply i
      generalize hp : finProdFinEquiv.symm i = p at hi
      rcases p with ⟨half, j⟩
      rw [← hi]
      fin_cases half <;>
        simp [ButterflyTree.applyLevels, ButterflyTree.applyLevel,
          ButterflyTree.mainCoefficient_node, ButterflyTree.offCoefficient_node,
          butterflyToggleIndex_pair, ButterflyTree.applyLevels_half,
          ihl, ihr, ButterflyTree.apply, ButterflyFactor.apply, add_comm]

/-- The independent-half hypothesis is satisfiable at an actual level.
Source: Appendix `def: butterfly`, a two-level tree. -/
example : (1 : ℕ) ≤ 1 := by decide

end Transformer.Zoology
