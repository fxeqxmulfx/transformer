/-
# Lattice codebooks and sign bytes

The i-quants (`IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`) store groups of 4 or 8
weights as an index into a table of grid points, `ggml/src/ggml-common.h`
(llama.cpp commit `335b21f`).  An entry is a `uint32_t` or `uint64_t` whose
byte `j` is coordinate `j`; the tables are lists of those integers, and
`gridByte` reads a coordinate.

The IQ2 and IQ3 grids have nonnegative coordinates; the signs come separately,
7 bits per group of 8 through `ksigns_iq2xs`, whose eighth bit is the parity
of the other seven (`ksigns_eq`): every group has an even number of minus
signs, and the eighth sign costs nothing.
-/

import Mathlib.Data.List.GetD
import Mathlib.Algebra.BigOperators.Group.List.Basic

namespace Transformer
namespace GGUF
namespace Grid

/-- Coordinate `j` of entry `k` of a table: byte `j` of the little-endian
integer. -/
def gridByte (t : List ℕ) (k j : ℕ) : ℕ := t.getD k 0 / 256 ^ j % 256

/-- A coordinate property checked entry by entry holds at every index. -/
theorem gridByte_mem {t : List ℕ} {w : ℕ} {S : List ℕ}
    (h : t.all (fun v => (List.range w).all fun j => v / 256 ^ j % 256 ∈ S) = true)
    {k j : ℕ} (hk : k < t.length) (hj : j < w) : gridByte t k j ∈ S := by
  rw [List.all_eq_true] at h
  have := h _ (List.getElem_mem hk)
  rw [List.all_eq_true] at this
  have := this j (List.mem_range.2 hj)
  rw [gridByte, List.getD_eq_getElem _ _ hk]
  simpa using this

/-- `ksigns_iq2xs`. -/
def ksignsTable : List ℕ := [0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255]

/-- `ksigns_iq2xs[k]`. -/
def ksigns (k : ℕ) : ℕ := ksignsTable.getD k 0

/-- **The eighth sign bit is the parity of the other seven.** -/
theorem ksigns_eq {k : ℕ} (hk : k < 128) :
    ksigns k = k + 128 * (((List.range 7).map fun j => k / 2 ^ j % 2).sum % 2) := by
  have h : (List.range 128).all (fun k =>
      ksigns k == k + 128 * (((List.range 7).map fun j => k / 2 ^ j % 2).sum % 2)) = true := by
    decide +kernel
  rw [List.all_eq_true] at h
  simpa using h k (List.mem_range.2 hk)

/-- The sign of bit `j` of a sign byte, `signs & kmask_iq2xs[j] ? -1 : 1`. -/
def sgn (b j : ℕ) : ℤ := if b / 2 ^ j % 2 = 1 then -1 else 1

theorem sgn_eq (b j : ℕ) : sgn b j = 1 ∨ sgn b j = -1 := by
  unfold sgn; split_ifs <;> simp

/-- The hypotheses of `gridByte_mem` and `ksigns_eq` are satisfiable. -/
example : ([0x0102] : List ℕ).all (fun v => (List.range 2).all fun j => v / 256 ^ j % 256 ∈ [1, 2])
    = true ∧ 0 < ([0x0102] : List ℕ).length ∧ 1 < 2 ∧ 5 < 128 := by decide

end Grid
end GGUF
end Transformer
