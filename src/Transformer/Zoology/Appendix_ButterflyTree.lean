/-
# Recursive butterfly matrices

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly`.
At every level, an arbitrary butterfly factor combines two independently
chosen half-size butterfly matrices.  This is the paper's recursive
definition; the number of scalar coefficients is exactly `2k·2^k` at
size `2^k` before any parameter sharing or compression.
-/

import Transformer.Zoology.Appendix_Butterfly

namespace Transformer.Zoology

/-- The dimension of a depth-`k` butterfly matrix, written recursively so
that the two-half decomposition is definitional.  Source: Appendix
`def: butterfly`. -/
def butterflyWidth : ℕ → ℕ
  | 0 => 1
  | k + 1 => 2 * butterflyWidth k

/-- Recursive butterfly matrices, with two independent half-size factors.
Source: Appendix `def: butterfly`. -/
inductive ButterflyTree : ℕ → Type
  | leaf : ButterflyTree 0
  | node {k : ℕ} (factor : ButterflyFactor (butterflyWidth k))
      (left right : ButterflyTree k) : ButterflyTree (k + 1)

/-- Evaluate a butterfly matrix on a vector.
Source: Appendix `def: butterfly`, recursive display. -/
def ButterflyTree.apply : {k : ℕ} → ButterflyTree k →
    (Fin (butterflyWidth k) → ℝ) → Fin (butterflyWidth k) → ℝ
  | 0, .leaf, u => u
  | k + 1, .node factor left right, u =>
      let halves : HalfVector (butterflyWidth k) :=
        fun half t =>
          if half = 0 then
            left.apply (fun j => u (finProdFinEquiv (0, j))) t
          else right.apply (fun j => u (finProdFinEquiv (1, j))) t
      fun j =>
        let ht := finProdFinEquiv.symm j
        factor.apply halves ht.1 ht.2

/-- Number of real coefficients stored in the recursive butterfly factors.
Source: Appendix `def: butterfly`: each factor at size `2m` has four
length-`m` diagonals. -/
def ButterflyTree.parameterCount : {k : ℕ} → ButterflyTree k → ℕ
  | 0, .leaf => 0
  | k + 1, .node _ left right =>
      4 * butterflyWidth k + left.parameterCount + right.parameterCount

/-- The recursive width is the power of two stated in the paper.
Source: Appendix `def: butterfly`. -/
theorem butterflyWidth_eq_pow (k : ℕ) : butterflyWidth k = 2 ^ k := by
  induction k with
  | zero => rfl
  | succ k ih =>
      simp [butterflyWidth, ih, pow_succ, mul_comm]

/-- Butterfly dimensions are positive, including the identity leaf.
Source: Appendix `def: butterfly`, power-of-two dimensions. -/
theorem butterflyWidth_pos (k : ℕ) : 0 < butterflyWidth k := by
  rw [butterflyWidth_eq_pow]
  positivity

/-- Modular indices are available at every butterfly dimension.
Source: Appendix `def: butterfly`, positive power-of-two dimensions. -/
instance butterflyWidth_neZero (k : ℕ) : NeZero (butterflyWidth k) :=
  ⟨Nat.ne_of_gt (butterflyWidth_pos k)⟩

/-- A depth-`k` butterfly matrix has exactly `2k·2^k` scalar coefficients.
This substantiates the parameter count implicit in the butterfly definition.
Source: Appendix `def: butterfly`. -/
theorem ButterflyTree.parameterCount_eq {k : ℕ} (tree : ButterflyTree k) :
    tree.parameterCount = 2 * k * butterflyWidth k := by
  induction tree with
  | leaf => rfl
  | @node k factor left right ihLeft ihRight =>
      simp only [ButterflyTree.parameterCount, butterflyWidth]
      rw [ihLeft, ihRight]
      ring

/-- Depth-one butterfly matrices exist and act on two real coordinates.
Source: Appendix `def: butterfly`. -/
example : ∃ tree : ButterflyTree 1,
    tree.parameterCount = 4 := by
  let factor : ButterflyFactor 1 := {
    upperLeft := fun _ => 1
    upperRight := fun _ => 0
    lowerLeft := fun _ => 0
    lowerRight := fun _ => 1
  }
  refine ⟨.node factor .leaf .leaf, ?_⟩
  decide

end Transformer.Zoology
