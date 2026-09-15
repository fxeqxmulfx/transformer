/-
# The interleaved weight layout changes no answer

`vm-rs/alm-model/src/linear.rs` is where the forward pass spends its time: on
the `sudoku` trace `Dense::apply` carries 48 % of the run's samples.  It earns
that back by not storing the matrix the way the file does.  `model.bin` is
row-major, and `Dense::of` scatters it — row `i`, column `j` of the matrix
lands at `((i / LANES) * cols + j) * LANES + i % LANES`, so the `j`-th entry
of `LANES` consecutive rows sits in `LANES` adjacent words and one vector FMA
advances that many rows at once.

The module docstring claims this "changes no answer", and gives the reason:
each lane still sums its own row left to right, so no regrouping happens and
the reference traces stay bit-for-bit what they were.  That claim has two
halves and they need different arguments.

The first half is combinatorial and is where a transcription error would
live: the scatter is a *relabelling*, every `(i, j)` reaching its own word and
no two sharing one.  `unpack_packed` proves it by exhibiting the inverse, and
`packed_lt_buffer` proves the words are inside the buffer `Dense::of`
allocates.

The second half is about arithmetic, and the point is that it needs none.
`rowFold` folds a row with an *arbitrary* `add` and `mul` — not `ℝ`'s, not
assumed associative, commutative, or anything else — because the order within
a row is untouched.  `apply_eq_rowMajor` then says the two layouts fold to
the identical element for any such pair, which is the bit-exactness claim:
had the kernel regrouped a row, no theorem at this generality could hold.

What is *not* covered here is the padding.  `Dense::of` rounds the row count
up to a multiple of `LANES` with zeros and `apply` truncates their sums away
with `&s[..out.len()]`; that is a fact about slice lengths, not about the
layout, and it is left to `debug_assert_eq!(y.len(), self.rows)`.

Source: `vm-rs/alm-model/src/linear.rs`, `Dense::of` and `Dense::apply`.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Range
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Ring

namespace Transformer
namespace ALM

/-! ### The two addresses of one entry -/

/-- Where row `i`, column `j` sits in the file: `model.bin` is row-major, and
`SparseHead::of` reads it as `w[i * cols + j]`. -/
def rowMajor (cols i j : ℕ) : ℕ := i * cols + j

/-- Where `Dense::of` puts that same entry: the row is split into a block
index `i / lanes` and a lane `i % lanes`, and the lane is the fastest-varying
coordinate so that one load covers `lanes` rows. -/
def packed (lanes cols i j : ℕ) : ℕ := ((i / lanes) * cols + j) * lanes + i % lanes

/-- Reading a packed address back as a row and a column. -/
def unpack (lanes cols k : ℕ) : ℕ × ℕ :=
  ((k / lanes / cols) * lanes + k % lanes, k / lanes % cols)

/-- **The scatter loses nothing.**  `unpack` is a left inverse of `packed` on
every address the loop in `Dense::of` writes, so distinct entries land in
distinct words and each one can be read back.  This is the half of "changes
no answer" that a mistyped index would break. -/
theorem unpack_packed {lanes cols i j : ℕ} (hlanes : 0 < lanes) (hcols : 0 < cols)
    (hj : j < cols) : unpack lanes cols (packed lanes cols i j) = (i, j) := by
  have hr : i % lanes < lanes := Nat.mod_lt _ hlanes
  have hdiv : packed lanes cols i j / lanes = (i / lanes) * cols + j := by
    rw [packed, show ((i / lanes) * cols + j) * lanes = lanes * ((i / lanes) * cols + j) by ring,
      Nat.mul_add_div hlanes, Nat.div_eq_of_lt hr, Nat.add_zero]
  have hmod : packed lanes cols i j % lanes = i % lanes := by
    rw [packed, show ((i / lanes) * cols + j) * lanes = lanes * ((i / lanes) * cols + j) by ring,
      Nat.mul_add_mod, Nat.mod_eq_of_lt hr]
  have hcomm : (i / lanes) * cols + j = cols * (i / lanes) + j := by ring
  rw [unpack, hdiv, hmod, hcomm, Nat.mul_add_div hcols, Nat.div_eq_of_lt hj, Nat.add_zero,
    Nat.mul_add_mod, Nat.mod_eq_of_lt hj, Nat.mul_comm, Nat.div_add_mod]

/-- The hypotheses are satisfiable at the shape the kernel actually runs:
eight lanes, and row `9` column `2` of a three-column matrix goes to word
`41`. -/
example : unpack 8 3 (packed 8 3 9 2) = (9, 2) := unpack_packed (by norm_num) (by norm_num) (by norm_num)

/-- **And it stays inside the allocation.**  `Dense::of` allocates
`rows.div_ceil(LANES) * cols * LANES` words; write the block count as
`blocks`, which is exactly the assumption `rows ≤ blocks * lanes`, and every
address the loop writes is below the buffer's length. -/
theorem packed_lt_buffer {lanes cols rows blocks i j : ℕ} (hlanes : 0 < lanes)
    (hblocks : rows ≤ blocks * lanes) (hi : i < rows) (hj : j < cols) :
    packed lanes cols i j < blocks * cols * lanes := by
  have hb : i / lanes < blocks := (Nat.div_lt_iff_lt_mul hlanes).mpr (by omega)
  have hr : i % lanes < lanes := Nat.mod_lt _ hlanes
  have hA : (i / lanes) * cols + j < blocks * cols :=
    lt_of_lt_of_le (by rw [Nat.succ_mul]; omega) (Nat.mul_le_mul_right cols hb)
  calc packed lanes cols i j < ((i / lanes) * cols + j + 1) * lanes := by
        rw [packed, Nat.succ_mul]; omega
    _ ≤ (blocks * cols) * lanes := Nat.mul_le_mul_right lanes hA

/-- The hypotheses are satisfiable at a shape with real padding in it: ten
rows in two blocks of eight, and the last row's last column is word `41` of
a forty-eight word buffer. -/
example : packed 8 3 9 2 < 2 * 3 * 8 :=
  packed_lt_buffer (lanes := 8) (rows := 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-! ### The fold that is not allowed to regroup -/

/-- One row's inner product, summed left to right, over an **arbitrary**
`add` and `mul`.  Nothing is assumed of either — not associativity, not
commutativity, not that they are `ℝ`'s — because the claim being made about
the layout is that it changes no partial sum, and a claim that weak in its
arithmetic is exactly a claim that strong about the order.

`c j` is the row's `j`-th coefficient and `x j` the input's `j`-th entry;
`Dense::apply` runs this fold once per lane, with `c` the lane's slice of the
packed buffer. -/
def rowFold (add mul : ℝ → ℝ → ℝ) (c x : ℕ → ℝ) (cols : ℕ) : ℝ :=
  (List.range cols).foldl (fun s j => add s (mul (c j) (x j))) 0

/-- Coefficients that agree on the columns that exist fold to the same
element — the only thing one can say without knowing anything about `add`. -/
theorem rowFold_congr (add mul : ℝ → ℝ → ℝ) (x : ℕ → ℝ) {c c' : ℕ → ℝ} {cols : ℕ}
    (h : ∀ j < cols, c j = c' j) : rowFold add mul c x cols = rowFold add mul c' x cols := by
  induction cols with
  | zero => rfl
  | succ n ih =>
    have hn : ∀ j < n, c j = c' j := fun j hj => h j (by omega)
    rw [rowFold, rowFold, List.range_succ, List.foldl_append, List.foldl_append]
    rw [show (List.range n).foldl (fun s j => add s (mul (c j) (x j))) 0
        = (List.range n).foldl (fun s j => add s (mul (c' j) (x j))) 0 from ih hn]
    simp only [List.foldl_cons, List.foldl_nil, h n (by omega)]

/-- The hypothesis is satisfiable without the two functions being equal: they
need only agree below `cols`, and these differ at `cols` itself. -/
example (add mul : ℝ → ℝ → ℝ) (x : ℕ → ℝ) :
    rowFold add mul (fun _ => 0) x 2 = rowFold add mul (fun j => if j < 2 then 0 else 1) x 2 :=
  rowFold_congr add mul x (fun j hj => by simp [hj])

/-- **The interleaved kernel computes the row-major answer.**  Given only that
the packed buffer holds at each scattered address what the file holds at the
matching row-major one — which is what `Dense::of`'s loop establishes — the
fold over the packed row and the fold over the file's row are the same
element, for every `add` and `mul` whatsoever.

That generality is the content.  The equality holds step for step, so it
holds in `f64` too, with its non-associative addition: the partial sums are
the ones `transformer.cpp` forms, in the order it forms them, and the
reference traces are reproduced bit for bit rather than to within a rounding.

Source: `vm-rs/alm-model/src/linear.rs`, `Dense::apply`. -/
theorem apply_eq_rowMajor (add mul : ℝ → ℝ → ℝ) (x : ℕ → ℝ) {w p : ℕ → ℝ}
    {lanes cols rows i : ℕ}
    (hpack : ∀ i j, i < rows → j < cols → p (packed lanes cols i j) = w (rowMajor cols i j))
    (hi : i < rows) :
    rowFold add mul (fun j => p (packed lanes cols i j)) x cols
      = rowFold add mul (fun j => w (rowMajor cols i j)) x cols :=
  rowFold_congr add mul x (fun j hj => hpack i j hi hj)

/-- The hypotheses are satisfiable at a shape where the two layouts genuinely
disagree: two lanes, two columns, and the packed buffer is the file's four
weights with the middle two exchanged.  Row `1` still folds to what the file
says it folds to. -/
example (add mul : ℝ → ℝ → ℝ) (x : ℕ → ℝ) :
    rowFold add mul
        (fun j => (fun k : ℕ => if k = 1 then (2 : ℝ) else if k = 2 then 1 else k)
          (packed 2 2 1 j)) x 2
      = rowFold add mul (fun j => (fun k : ℕ => (k : ℝ)) (rowMajor 2 1 j)) x 2 :=
  apply_eq_rowMajor add mul x (lanes := 2) (cols := 2) (rows := 2) (i := 1)
    (p := fun k : ℕ => if k = 1 then (2 : ℝ) else if k = 2 then 1 else k)
    (w := fun k : ℕ => (k : ℝ))
    (by intro i j hi hj; interval_cases i <;> interval_cases j <;> norm_num [packed, rowMajor])
    (by norm_num)

end ALM
end Transformer
