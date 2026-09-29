/-
# Workspace for a diagonal term plus a shared linear projection

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena` and
`lmm:primitives`. The identity
  (Wu+a)(u+b) - (Wu)u - ab = au + b(Wu)
allows position-dependent diagonal coefficients without increasing the
feature dimension. Three adjacent buffers fit in the common 9n workspace.
-/

import Transformer.Zoology.Appendix_ShiftSumProgram

namespace Transformer.Zoology

/-- Row of the first, second, or third cancellation buffer.
Source: Appendix `prop: butterfly-hyena`, corrected addition construction. -/
def diagonalLinearSlot {n : ℕ} (a : Fin 3) (i : Fin n) : Fin (9 * n) :=
  ⟨a.val * n + i.val, by
    have ha := a.isLt
    have hi := i.isLt
    fin_cases a <;> dsimp at * <;> omega⟩

/-- Bias in the projection branch: a on the first buffer and ab on the
third; the second has zero bias. Source: Appendix `lmm:primitives`,
algebraic cancellation of the gated layer's quadratic term. -/
def diagonalLinearBias₁ {n d : ℕ} (a b : RealSequence n d) :
    RealSequence (9 * n) d :=
  fun j q => if h : j.val < n then a ⟨j.val, h⟩ q
    else if h : 2 * n ≤ j.val ∧ j.val < 3 * n then
      a ⟨j.val - 2 * n, by omega⟩ q * b ⟨j.val - 2 * n, by omega⟩ q
    else 0

/-- Bias in the convolution branch: b on the first buffer and one on the
third. Source: Appendix `lmm:primitives`, cancellation construction. -/
def diagonalLinearBias₂ {n d : ℕ} (b : RealSequence n d) :
    RealSequence (9 * n) d :=
  fun j q => if h : j.val < n then b ⟨j.val, h⟩ q
    else if 2 * n ≤ j.val ∧ j.val < 3 * n then 1 else 0

/-- At the three buffers the first bias is a, zero, and ab respectively.
Source: Appendix `prop: butterfly-hyena`, cancellation buffers. -/
theorem diagonalLinearBias₁_slot {n d : ℕ} (a b : RealSequence n d)
    (r : Fin 3) (i : Fin n) (q : Fin d) :
    diagonalLinearBias₁ a b (diagonalLinearSlot r i) q =
      if r = 0 then a i q else if r = 1 then 0 else a i q * b i q := by
  have hi := i.isLt
  fin_cases r <;>
    simp [diagonalLinearBias₁, diagonalLinearSlot,
      show ¬ n + i.val < n by omega,
      show ¬ 2 * n ≤ n + i.val by omega,
      show ¬ 2 * n + i.val < n by omega,
      show 2 * n + i.val < 3 * n by omega,
      show 2 * n ≤ 2 * n + i.val by omega, hi]

/-- At the three buffers the second bias is b, zero, and one.
Source: Appendix `prop: butterfly-hyena`, cancellation buffers. -/
theorem diagonalLinearBias₂_slot {n d : ℕ} (b : RealSequence n d)
    (r : Fin 3) (i : Fin n) (q : Fin d) :
    diagonalLinearBias₂ b (diagonalLinearSlot r i) q =
      if r = 0 then b i q else if r = 1 then 0 else 1 := by
  have hi := i.isLt
  fin_cases r <;>
    simp [diagonalLinearBias₂, diagonalLinearSlot,
      show ¬ n + i.val < n by omega,
      show ¬ 2 * n ≤ n + i.val by omega,
      show ¬ 2 * n + i.val < n by omega,
      show 2 * n + i.val < 3 * n by omega,
      show 2 * n ≤ 2 * n + i.val by omega, hi]

/-- The input copy layer fills the first two buffers and leaves the third
zero. Source: Appendix `prop: prim-add`, duplication stage. -/
theorem diagonalLinearCopy_slot {n d : ℕ} [NeZero n]
    (u : RealSequence n d) (r : Fin 3) (i : Fin n) (q : Fin d) :
    shiftSumCopyState u (diagonalLinearSlot r i) q =
      if r = 2 then 0 else u i q := by
  fin_cases r
  · simpa [diagonalLinearSlot, shiftSumInputSlot] using
      shiftSumCopyState_first u i q
  · simpa using shiftSumCopyState_read u i (0 : Fin n)
      (diagonalLinearSlot 1 i) (by simp [diagonalLinearSlot]) q
  · apply shiftSumCopyState_outside
    simp [diagonalLinearSlot]

/-- Signed convolution taps for collecting the cancellation buffers.
Source: Appendix `prop: prim-add`, signed buffer sum. -/
def diagonalLinearTap (n : ℕ) [NeZero n] (r : Fin 3) : Fin (9 * n) :=
  -(⟨r.val * n, by
    have hr := r.isLt
    have hn := NeZero.pos n
    fin_cases r <;> dsimp at * <;> omega⟩ : Fin (9 * n))

/-- The collecting tap reads the matching row in its buffer.
Source: Appendix `prop: prim-add`, cancellation alignment. -/
theorem diagonalLinearTap_read {n : ℕ} [NeZero n] (r : Fin 3) (i : Fin n) :
    shiftSumInputSlot i - diagonalLinearTap n r = diagonalLinearSlot r i := by
  apply Fin.ext
  simp only [diagonalLinearTap, sub_neg_eq_add, Fin.val_add,
    shiftSumInputSlot, diagonalLinearSlot]
  have h := (diagonalLinearSlot r i).isLt
  dsimp [diagonalLinearSlot] at h
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

end Transformer.Zoology
