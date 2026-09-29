/-
# Finite indices for block butterfly factors

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`eq: butterfly-split`. Each block has two equal halves. The off-diagonal
reads below stay within their own block, even with global cyclic shifts.
-/

import Transformer.Zoology.Appendix_ShiftSumK

namespace Transformer.Zoology

/-- Row-major index of a coordinate in a two-half butterfly block.
Source: Appendix `def: butterfly`, block diagonal factor matrix. -/
def blockButterflyIndex {b m : ℕ} (block : Fin b) (half : Fin 2)
    (t : Fin m) : Fin (b * (2 * m)) :=
  finProdFinEquiv (block, finProdFinEquiv (half, t))

/-- Decode a row into its block, half, and within-half coordinate.
Source: Appendix `def: butterfly`, block diagonal layout. -/
def blockButterflyParts {b m : ℕ} (j : Fin (b * (2 * m))) :
    Fin b × (Fin 2 × Fin m) :=
  let p := finProdFinEquiv.symm j
  (p.1, finProdFinEquiv.symm p.2)

/-- The concrete row index is the within-half offset plus its block base.
Source: Appendix `def: butterfly`, contiguous blocks. -/
theorem blockButterflyIndex_val {b m : ℕ} (block : Fin b) (half : Fin 2)
    (t : Fin m) :
    (blockButterflyIndex block half t).val =
      t.val + m * half.val + (2 * m) * block.val := by
  simp [blockButterflyIndex, finProdFinEquiv_apply_val]

/-- Decoding an encoded coordinate returns its three original indices.
Source: Appendix `def: butterfly`, block layout. -/
theorem blockButterflyParts_index {b m : ℕ} (block : Fin b)
    (half : Fin 2) (t : Fin m) :
    blockButterflyParts (blockButterflyIndex block half t) =
      (block, half, t) := by
  simp only [blockButterflyParts, blockButterflyIndex,
    Equiv.symm_apply_apply]

/-- Reassembling the decoded indices returns the original row.
Source: Appendix `def: butterfly`, complete block partition. -/
theorem blockButterflyIndex_parts {b m : ℕ} (j : Fin (b * (2 * m))) :
    blockButterflyIndex (blockButterflyParts j).1
      (blockButterflyParts j).2.1 (blockButterflyParts j).2.2 = j := by
  simp only [blockButterflyParts, blockButterflyIndex, Prod.eta,
    Equiv.apply_symm_apply]

/-- A half-block cyclic offset, used only where it stays in the block.
Source: Appendix `eq: butterfly-split`, half-length shifts. -/
def blockButterflyOffset (b m : ℕ) [NeZero b] [NeZero m] :
    Fin (b * (2 * m)) :=
  ⟨m, by
    have h := (blockButterflyIndex (0 : Fin b) (1 : Fin 2)
      (0 : Fin m)).isLt
    simpa [blockButterflyIndex_val] using h⟩

/-- Adding the half-block offset moves an upper-half row to the lower
half of the same block. Source: Appendix `eq: butterfly-split`. -/
theorem blockButterflyIndex_add {b m : ℕ} [NeZero b] [NeZero m]
    (block : Fin b) (t : Fin m) :
    blockButterflyIndex block 0 t + blockButterflyOffset b m =
      blockButterflyIndex block 1 t := by
  apply Fin.ext
  have h := (blockButterflyIndex block (1 : Fin 2) t).isLt
  have hsum : (blockButterflyIndex block (0 : Fin 2) t).val +
      (blockButterflyOffset b m).val < b * (2 * m) := by
    simpa [blockButterflyIndex_val, blockButterflyOffset, add_assoc,
      add_comm, add_left_comm] using h
  rw [Fin.val_add_eq_of_add_lt hsum]
  simp [blockButterflyIndex_val, blockButterflyOffset, add_assoc,
    add_comm]

/-- Subtracting the half-block offset moves a lower-half row to the upper
half of the same block. Source: Appendix `eq: butterfly-split`. -/
theorem blockButterflyIndex_sub {b m : ℕ} [NeZero b] [NeZero m]
    (block : Fin b) (t : Fin m) :
    blockButterflyIndex block 1 t - blockButterflyOffset b m =
      blockButterflyIndex block 0 t := by
  apply Fin.ext
  have hle : blockButterflyOffset b m ≤ blockButterflyIndex block 1 t := by
    change m ≤ (blockButterflyIndex block 1 t).val
    simp only [blockButterflyIndex_val, Fin.val_one, mul_one]
    omega
  rw [Fin.sub_val_of_le hle]
  simp only [blockButterflyIndex_val, blockButterflyOffset,
    Fin.val_one, Fin.val_zero, mul_one, mul_zero, add_zero]
  omega

/-- Two nonempty blocks of half-length two admit the required offset.
Source: Appendix `def: butterfly`, nondegenerate factor example. -/
example : (blockButterflyOffset 2 2).val = 2 := rfl

end Transformer.Zoology
