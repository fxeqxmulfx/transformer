/-
# Running-sum primitive on packed input blocks

Arora et al., arXiv:2312.04927v1, Appendix `sec: primitives`, `Add`, and
Proposition `prop: prim-add`. The input sequence consists of an `x` block,
a following `S` block, and zeros. The output has ones in the first block,
`S+x` in the second, and zeros afterward.
-/

import Transformer.Zoology.Appendix_RunningSum

namespace Transformer.Zoology

/-- The position of element `j` within the second length-`n` block.
Source: Appendix `sec: primitives`, `Add`. -/
def secondBlockIndex {N n : ℕ} (hfit : 2 * n ≤ N)
    (j : Fin n) : Fin N :=
  ⟨n + j.val, by omega⟩

/-- Form the input layout `(x,S,0)` used in the paper's addition primitive.
Source: Appendix `sec: primitives`, `Add`. -/
def packedAddInput {N n d : ℕ} (_hfit : 2 * n ≤ N)
    (x S : RealSequence n d) : RealSequence N d :=
  fun i q =>
    if hlow : i.val < n then x ⟨i.val, hlow⟩ q else
    if hhigh : i.val < 2 * n then
      S ⟨i.val - n, by omega⟩ q else 0

/-- The second packed block contains exactly `S`. Source: Appendix
`sec: primitives`, `Add`, input layout. -/
theorem packedAddInput_second {N n d : ℕ} (hfit : 2 * n ≤ N)
    (x S : RealSequence n d) (j : Fin n) (q : Fin d) :
    packedAddInput hfit x S (secondBlockIndex hfit j) q = S j q := by
  simp [packedAddInput, secondBlockIndex,
    show ¬ n + j.val < n by omega,
    show n + j.val < 2 * n by omega]

/-- Looking back by one block from the second block reaches the
corresponding entry of `x`. Source: Appendix Proposition `prop: prim-add`,
polynomial identity `(1+X^n)(x+X^n S)`. -/
theorem packedAddInput_shift_second {N n d : ℕ} [NeZero N]
    (hfit : 2 * n ≤ N) (hn : 0 < n)
    (x S : RealSequence n d) (j : Fin n) (q : Fin d) :
    shiftDown (packedAddInput hfit x S)
      ⟨n, by omega⟩ (secondBlockIndex hfit j) q = x j q := by
  let s : Fin N := ⟨n, by omega⟩
  let i : Fin N := secondBlockIndex hfit j
  have hle : s ≤ i := by
    change n ≤ n + j.val
    omega
  have hsub : (i - s).val = j.val := by
    rw [Fin.sub_val_of_le hle]
    change (n + j.val) - n = j.val
    omega
  have hidx : i - s = ⟨j.val, by omega⟩ := Fin.ext hsub
  change shiftDown (packedAddInput hfit x S) s i q = x j q
  rw [show shiftDown (packedAddInput hfit x S) s i q =
    packedAddInput hfit x S (i - s) q by simp [shiftDown, hle]]
  rw [hidx]
  simp [packedAddInput]

/-- On a packed input, the two-layer output's second block is precisely
`S+x` at every feature coordinate. Source: Appendix Proposition
`prop: prim-add`, concluding display. -/
theorem coyote_addBlock_packed_correct {N n d : ℕ} [NeZero N]
    (hfit : 2 * n ≤ N) (hn : 0 < n)
    (x S : RealSequence n d) (j : Fin n) (q : Fin d) :
    coyoteLayer (addBlockSecondParameters ⟨n, by omega⟩)
      (coyoteLayer (addBlockFirstParameters ⟨n, by omega⟩)
        (packedAddInput hfit x S))
      (secondBlockIndex hfit j) q = S j q + x j q := by
  let s : Fin N := ⟨n, by omega⟩
  change coyoteLayer (addBlockSecondParameters s)
    (coyoteLayer (addBlockFirstParameters s) (packedAddInput hfit x S))
      (secondBlockIndex hfit j) q = _
  rw [show coyoteLayer (addBlockSecondParameters s)
      (coyoteLayer (addBlockFirstParameters s) (packedAddInput hfit x S)) =
      addBlockOutput (packedAddInput hfit x S) s from
    coyote_two_layers_addBlock _ s]
  have hlow : ¬(secondBlockIndex hfit j).val < s.val := by
    simp [secondBlockIndex, s]
  have hhigh : (secondBlockIndex hfit j).val < 2 * s.val := by
    simp [secondBlockIndex, s]
    omega
  simp only [addBlockOutput, hlow, hhigh, ite_false, ite_true]
  rw [packedAddInput_second, packedAddInput_shift_second hfit hn]

/-- The positive size and fit hypotheses are jointly satisfiable. -/
example : 0 < 1 ∧ 2 * 1 ≤ 2 := by norm_num

end Transformer.Zoology
