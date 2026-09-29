/-
# Constant-depth Coyote realization of block butterfly factors

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`prop: butterfly-hyena`. This realizes arbitrary block butterfly factors
along the sequence axis, separately in every feature column. It includes
the off-diagonal shifts omitted in the source proof's later display.
The full row-major factor also requires the feature-axis factors.
-/

import Transformer.Zoology.Appendix_BlockButterflyIndex

open scoped BigOperators

namespace Transformer.Zoology

/-- Independently apply each block's butterfly factor in each column.
Source: Appendix `def: butterfly`, block diagonal factor matrix. -/
def blockButterflyApply {b m d : ℕ}
    (f : Fin b → Fin d → ButterflyFactor m)
    (u : RealSequence (b * (2 * m)) d) : RealSequence (b * (2 * m)) d :=
  fun j q =>
    let p := blockButterflyParts j
    (f p.1 q).apply (fun h t => u (blockButterflyIndex p.1 h t) q)
      p.2.1 p.2.2

/-- The main diagonal and the two masked off-diagonals of the blocks.
Source: Appendix `eq: butterfly-split`, with masks before cyclic shifts. -/
def blockButterflyCoefficients {b m d : ℕ}
    (f : Fin b → Fin d → ButterflyFactor m) :
    Fin 3 → RealSequence (b * (2 * m)) d :=
  fun a j q =>
    let p := blockButterflyParts j
    if a = 0 then
      if p.2.1 = 0 then (f p.1 q).upperLeft p.2.2
      else (f p.1 q).lowerRight p.2.2
    else if a = 1 then
      if p.2.1 = 0 then 0 else (f p.1 q).lowerLeft p.2.2
    else
      if p.2.1 = 0 then (f p.1 q).upperRight p.2.2 else 0

/-- The identity read and the two opposite half-block cyclic reads.
Source: Appendix `eq: butterfly-split`, retained off-diagonal shifts. -/
def blockButterflyShifts (b m : ℕ) [NeZero b] [NeZero m] :
    Fin 3 → Fin (b * (2 * m)) :=
  fun a => if a = 0 then 0 else if a = 1 then blockButterflyOffset b m
    else -blockButterflyOffset b m

/-- Every block factor is exactly a sum of three shifted diagonal maps;
the masks prevent reading any neighboring block.
Source: Appendix `eq: butterfly-split` and `def: butterfly`. -/
theorem blockButterfly_shiftedDiagonalSum {b m d : ℕ} [NeZero b] [NeZero m]
    (f : Fin b → Fin d → ButterflyFactor m)
    (u : RealSequence (b * (2 * m)) d) :
    shiftedDiagonalSum (blockButterflyCoefficients f)
      (blockButterflyShifts b m) u = blockButterflyApply f u := by
  funext j q
  have hj := blockButterflyIndex_parts j
  generalize hp : blockButterflyParts j = p at hj
  rcases p with ⟨block, half, t⟩
  rw [← hj]
  fin_cases half <;>
    (simp [shiftedDiagonalSum, Fin.sum_univ_three,
      blockButterflyCoefficients, blockButterflyShifts,
      blockButterflyApply, blockButterflyParts_index,
      ButterflyFactor.apply, sub_neg_eq_add,
      blockButterflyIndex_add, blockButterflyIndex_sub, add_comm])

/-- Three genuine K-constrained layers implementing the sequence-axis
block factors, with the original feature width.
Source: Appendix `prop: butterfly-hyena` and `def: W-kmat`. -/
def blockButterflyKNetwork {b m k : ℕ} [NeZero b] [NeZero m]
    (f : Fin b → Fin (butterflyWidth k) → ButterflyFactor m) :
    CyclicKCoyoteNetwork (9 * (b * (2 * m))) k 1 :=
  shiftSumKNetwork (blockButterflyCoefficients f) (blockButterflyShifts b m)

/-- The block factor stack computes the exact operator and leaves zero
workspace, so another factor can be applied immediately.
Source: Appendix `prop: butterfly-hyena` and `lem: stacking-layers`. -/
theorem blockButterflyKNetwork_correct {b m k : ℕ} [NeZero b] [NeZero m]
    (f : Fin b → Fin (butterflyWidth k) → ButterflyFactor m)
    (u : RealSequence (b * (2 * m)) (butterflyWidth k)) :
    (blockButterflyKNetwork f).run (shiftSumPad u) =
      shiftSumPad (blockButterflyApply f u) := by
  rw [blockButterflyKNetwork, shiftSumKNetwork_correct,
    blockButterfly_shiftedDiagonalSum]

/-- The block factors require three layers, independent of their size.
Source: Appendix `prop: butterfly-hyena`, constant-depth component. -/
theorem blockButterflyKNetwork_layerCount {b m k : ℕ} [NeZero b] [NeZero m]
    (f : Fin b → Fin (butterflyWidth k) → ButterflyFactor m) :
    (blockButterflyKNetwork f).layerCount = 3 :=
  shiftSumKNetwork_layerCount _ _

/-- Exact count of stored K coefficients, convolution filters, and biases.
Source: Appendix `def: W-kmat` and `prop: single-baseconv`. -/
theorem blockButterflyKNetwork_parameterCount {b m k : ℕ}
    [NeZero b] [NeZero m]
    (f : Fin b → Fin (butterflyWidth k) → ButterflyFactor m) :
    (blockButterflyKNetwork f).parameterCount =
      3 * (27 * (b * (2 * m)) * butterflyWidth k +
        4 * (k + 1) * butterflyWidth (k + 1)) :=
  shiftSumKNetwork_parameterCount _ _

end Transformer.Zoology
