/-
# The padded rows, and that dropping them drops nothing else

`Transformer.ALM.Dense` proves the interleaved layout changes no answer, and
says in as many words what it does not cover: the padding.  `Dense::of` rounds
the row count up to a multiple of `LANES` with zeros, `Dense::apply` computes
those rows' sums along with the real ones, and then throws them away by copying
only `&s[..out.len()]` into the output chunk.  Until now that was a fact about
slice lengths carried by a `debug_assert_eq!`.

It is still a fact about slice lengths -- that is the point, and why no
arithmetic appears below.  The padded lanes are not shown to sum to zero; over
an arbitrary `add` and `mul` they do not, and `Dense` is stated at exactly that
generality.  They are shown to be *dropped*: the lanes `apply` copies out are
the row indices the file has, each exactly once, and the lanes it truncates are
row indices the file never had.

Five claims, and the last two are the ones that reach back into `Dense`:

* `writeAt_lt_rows` -- a lane the kernel copies is a row the matrix has.
* `dropped_ge_rows` -- and a lane it truncates is past the last row.
* `chunkOf_writeAt`, `writeAt_chunkOf` and `chunkOf_mem` -- block and lane are
  the quotient and the remainder, so every row is written by exactly one lane
  of exactly one block, and none is left unwritten.
* `rows_le_blockCount_mul` -- the block count `Dense::of` allocates for
  satisfies the hypothesis `packed_lt_buffer` assumes, so the address bound
  proved there applies to the buffer that is really allocated.
* `blocks_of_buffer` -- and the weight side of the `zip` has exactly as many
  groups as the output side has chunks, so neither is truncated.

Source: `vm-rs/alm-model/src/linear.rs`, `Dense::of` and `Dense::apply`.
-/

import Transformer.ALM.Dense
import Mathlib.Algebra.Order.Floor.Div

namespace Transformer
namespace ALM

/-! ### The shapes the loop runs over -/

/-- **How many blocks the kernel runs**, spelled as `Dense::of` spells it:
`rows.div_ceil(LANES)`. -/
def blockCount (lanes rows : ℕ) : ℕ := (rows + lanes - 1) / lanes

/-- The same number Mathlib writes `⌈/⌉`, which is how `Nat.div_ceil` and the
Rust method agree. -/
theorem blockCount_eq_ceilDiv (lanes rows : ℕ) : blockCount lanes rows = rows ⌈/⌉ lanes :=
  (Nat.ceilDiv_eq_add_pred_div rows lanes).symm

/-- **The length of one output chunk**: `y.chunks_mut(LANES)` hands the last
block a short chunk, and `out.len()` is that length. -/
def outLen (lanes rows b : ℕ) : ℕ := min lanes (rows - b * lanes)

/-- **Where lane `l` of block `b` lands** in the output. -/
def writeAt (lanes b l : ℕ) : ℕ := b * lanes + l

/-- **And which block and lane a row belongs to**: the quotient and the
remainder, which is the whole of `chunks_mut`. -/
def chunkOf (lanes i : ℕ) : ℕ × ℕ := (i / lanes, i % lanes)

/-! ### Every row is written, once -/

/-- **A lane the kernel copies out is a row the matrix has.**  `out.len()` is
what bounds the copy, and it never reaches past `rows`. -/
theorem writeAt_lt_rows {lanes rows b l : ℕ} (hl : l < outLen lanes rows b) :
    writeAt lanes b l < rows := by
  rw [outLen, lt_min_iff] at hl
  rw [writeAt]
  omega

/-- **A lane it truncates is past the last row.**  Beyond `out.len()` the
indices `s[l]` would have gone to are rows `Dense::of` zero-filled, and there
is nothing there to lose. -/
theorem dropped_ge_rows {lanes rows b l : ℕ} (hlt : l < lanes) (hl : outLen lanes rows b ≤ l) :
    rows ≤ writeAt lanes b l := by
  rw [outLen, min_le_iff] at hl
  rw [writeAt]
  omega

/-- **Block and lane recover the row.** -/
theorem writeAt_chunkOf (lanes i : ℕ) : writeAt lanes (chunkOf lanes i).1 (chunkOf lanes i).2 = i := by
  rw [writeAt, chunkOf]
  exact Nat.div_add_mod' i lanes

/-- **And the row recovers block and lane**, so no two lanes of the loop write
the same output word and none is left unwritten. -/
theorem chunkOf_writeAt {lanes b l : ℕ} (hl : l < lanes) :
    chunkOf lanes (writeAt lanes b l) = (b, l) := by
  have hlanes : 0 < lanes := lt_of_le_of_lt (Nat.zero_le l) hl
  rw [chunkOf, writeAt, Nat.mul_comm b lanes, Nat.mul_add_div hlanes, Nat.mul_add_mod,
    Nat.div_eq_of_lt hl, Nat.mod_eq_of_lt hl, Nat.add_zero]

/-! ### And the allocation is the one `Dense` assumed -/

/-- **`Dense::of`'s block count covers every row**, which is exactly the
hypothesis `packed_lt_buffer` takes on faith: with `blocks` the allocated
`rows.div_ceil(LANES)`, every address the scatter writes is inside the
buffer. -/
theorem rows_le_blockCount_mul {lanes rows : ℕ} (hlanes : 0 < lanes) :
    rows ≤ blockCount lanes rows * lanes := by
  have hdm : (rows + lanes - 1) / lanes * lanes + (rows + lanes - 1) % lanes = rows + lanes - 1 :=
    Nat.div_add_mod' _ lanes
  have hmod : (rows + lanes - 1) % lanes < lanes := Nat.mod_lt _ hlanes
  rw [blockCount]
  omega

/-- **Every row of the matrix is one of the lanes the kernel copies.**  The
block is in range and the lane is inside that block's chunk, so the loop
writes `y[i]` exactly once. -/
theorem chunkOf_mem {lanes rows i : ℕ} (hlanes : 0 < lanes) (hi : i < rows) :
    (chunkOf lanes i).1 < blockCount lanes rows ∧
      (chunkOf lanes i).2 < outLen lanes rows (chunkOf lanes i).1 := by
  have hdm : i / lanes * lanes + i % lanes = i := Nat.div_add_mod' i lanes
  have hmod : i % lanes < lanes := Nat.mod_lt _ hlanes
  refine ⟨?_, ?_⟩
  · show i / lanes < blockCount lanes rows
    exact (Nat.div_lt_iff_lt_mul hlanes).mpr (lt_of_lt_of_le hi (rows_le_blockCount_mul hlanes))
  · show i % lanes < outLen lanes rows (i / lanes)
    rw [outLen, lt_min_iff]
    omega

/-- **The two loops the kernel zips have the same length.**  The weight buffer
holds `blocks * cols * lanes` words, which `as_chunks::<LANES>` and
`chunks_exact(cols)` cut into `blocks` groups -- the number of chunks
`y.chunks_mut(LANES)` produces.  Neither side of the `zip` is truncated. -/
theorem blocks_of_buffer {lanes cols blocks : ℕ} (hlanes : 0 < lanes) (hcols : 0 < cols) :
    blocks * cols * lanes / lanes / cols = blocks := by
  rw [Nat.mul_div_cancel _ hlanes, Nat.mul_div_cancel _ hcols]

/-! ### The hypotheses are satisfiable -/

/-- The shipped output head is `915 x 38`, so the kernel runs 115 blocks of
eight and the last one copies three lanes of its eight. -/
example : blockCount 8 915 = 115 ∧ outLen 8 915 114 = 3 := by
  constructor <;> rfl

/-- Row `914`, the last, is lane `2` of block `114` -- a lane the loop copies,
since `2 < 3`. -/
example : chunkOf 8 (writeAt 8 114 2) = (114, 2) ∧ writeAt 8 114 2 = 914 :=
  ⟨chunkOf_writeAt (by norm_num), rfl⟩

/-- And lane `3` of that block, the first the truncation drops, is row `915`:
one past the matrix, and one of the five rows `Dense::of` filled with zeros. -/
example : (915 : ℕ) ≤ writeAt 8 114 3 := dropped_ge_rows (by norm_num) (by norm_num [outLen])

/-- The satisfiable form of `chunkOf_mem`'s hypotheses, at that same row, and
of `rows_le_blockCount_mul`'s: eight lanes cover the 915 rows in 920 words. -/
example : (0 : ℕ) < 8 ∧ (914 : ℕ) < 915 ∧ (915 : ℕ) ≤ blockCount 8 915 * 8 :=
  ⟨by norm_num, by norm_num, rows_le_blockCount_mul (by norm_num)⟩

end ALM
end Transformer
