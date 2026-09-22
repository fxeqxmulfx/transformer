/-
# Skipping the zeros, and the index the scan comes back with

The output projection is the one matrix `vm-rs/alm-model/src/linear.rs` does
not multiply densely.  The head is `vocab × d_model` — 915 × 38 and 85 % zero
— and `SparseHead::of` keeps only the entries that are not zero, in the order
the file lists them, so `argmax` touches a seventh of the words it otherwise
would.  `perf` puts 6 % of the `sudoku` run inside that loop.

The docstring says skipping them "is exact — adding `0.0 * x` to a finite
partial sum never changes it — so this is the same argmax, not an
approximation of it", and there are two claims there.

The first is that dropping a term with a zero coefficient does not move the
sum.  In `ℝ` that is `a + 0 * v = a` and needs no saying; in `f64` it is a
genuine hypothesis, and one with a boundary — `0.0 * v` is `NaN` when `v` is
infinite or `NaN`, and then the partial sum is destroyed rather than
preserved.  So `sparseFold_eq_rowFold` takes it as the hypothesis `hzero`,
quantified over the *entries of `x`* and not over all reals, which is exactly
the condition the runtime relies on and the docstring understates: it is the
input that has to be finite, not only the partial sum.

The second is that the scan returns the same index.  `SparseHead::argmax`
keeps the incumbent unless a score is *strictly* greater, which resolves a
tie towards the earlier row; `Tensor::argmax` and the C++ loop resolve it the
same way, and the reference traces depend on the choice.  `firstMax` is that
loop, `firstMax_le` says its answer is a maximum, and `firstMax_first` says
nothing before it attains that maximum — together, the first index attaining
it.  `firstMax_congr` is what carries the first claim into the second: rows
whose sparse scores agree with their dense scores are ranked identically.

Source: `vm-rs/alm-model/src/linear.rs`, `SparseHead::of` and
`SparseHead::argmax`.
-/

import Transformer.ALM.Dense

namespace Transformer
namespace ALM

/-! ### The row that skips its zeros -/

/-- One row of a compressed-sparse-rows product: the same fold as `rowFold`,
over the columns `keep` retains and in the same ascending order, which is the
order `SparseHead::of` pushes them in.

`keep` is a `Bool` and not a decision about `ℝ`, because the compiler makes
this choice once, off line, by testing `v != 0.0`. -/
def sparseFold (add mul : ℝ → ℝ → ℝ) (c x : ℕ → ℝ) (keep : ℕ → Bool) (cols : ℕ) : ℝ :=
  ((List.range cols).filter keep).foldl (fun s j => add s (mul (c j) (x j))) 0

/-- **The dropped terms were doing nothing.**  If every column `keep` discards
carries a zero coefficient, and if adding `mul 0 (x j)` to a partial sum
returns it unchanged, then the compressed row folds to what the full row folds
to — step for step, so in `f64` as well as in `ℝ`.

The second hypothesis is the one with content.  `add s (mul 0 v) = s` holds
for every finite `f64` `v` and fails when `v` is infinite or `NaN`, where
`0.0 * v` is `NaN`; the runtime is entitled to it because the inputs to the
head are finite activations.

Source: `vm-rs/alm-model/src/linear.rs`, `SparseHead`'s docstring. -/
theorem sparseFold_eq_rowFold (add mul : ℝ → ℝ → ℝ) (c x : ℕ → ℝ) (keep : ℕ → Bool) (cols : ℕ)
    (hdrop : ∀ j, keep j = false → c j = 0)
    (hzero : ∀ (s : ℝ) (j : ℕ), add s (mul 0 (x j)) = s) :
    sparseFold add mul c x keep cols = rowFold add mul c x cols := by
  induction cols with
  | zero => rfl
  | succ n ih =>
    rw [sparseFold, rowFold, List.range_succ, List.filter_append, List.foldl_append,
      List.foldl_append]
    rw [show ((List.range n).filter keep).foldl (fun s j => add s (mul (c j) (x j))) 0
        = (List.range n).foldl (fun s j => add s (mul (c j) (x j))) 0 from ih]
    cases hk : keep n with
    | false => simp [List.filter, hk, hdrop n hk, hzero]
    | true => simp [List.filter, hk]

/-- The hypotheses are satisfiable on a row that really is sparse: three
columns, the middle one zero and dropped, exact arithmetic supplying
`hzero`. -/
example (x : ℕ → ℝ) :
    sparseFold (· + ·) (· * ·) (fun j => if j = 1 then 0 else 1) x (fun j => j != 1) 3
      = rowFold (· + ·) (· * ·) (fun j => if j = 1 then 0 else 1) x 3 :=
  sparseFold_eq_rowFold _ _ _ _ _ _ (by intro j hj; simpa using hj) (by intro s j; ring)

/-! ### The index the scan settles on -/

/-- `SparseHead::argmax`'s loop over `n` rows: keep the incumbent unless a
later score is *strictly* greater.  `n = 0` returns `0`, as the Rust does with
`best` never assigned. -/
noncomputable def firstMax (f : ℕ → ℝ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => if f (firstMax f n) < f n then n else firstMax f n

/-- The answer is a row that exists, once there is one. -/
theorem firstMax_lt (f : ℕ → ℝ) {n : ℕ} (hn : 0 < n) : firstMax f n < n := by
  induction n with
  | zero => omega
  | succ m ih =>
    rw [firstMax]
    split
    · omega
    · rcases Nat.eq_zero_or_pos m with hm | hm
      · rw [hm, firstMax]; omega
      · exact Nat.lt_succ_of_lt (ih hm)

/-- The hypothesis is satisfiable: a head with rows at all. -/
example (f : ℕ → ℝ) : firstMax f 3 < 3 := firstMax_lt f (by norm_num)

/-- **It is a maximum.**  No row of the scan scores above the one it keeps. -/
theorem firstMax_le (f : ℕ → ℝ) {n j : ℕ} (hj : j < n) : f j ≤ f (firstMax f n) := by
  induction n with
  | zero => omega
  | succ m ih =>
    rw [firstMax]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hj with h | h
    · split
      · exact le_of_lt (lt_of_le_of_lt (ih h) ‹_›)
      · exact ih h
    · subst h
      split
      · exact le_rfl
      · exact not_lt.mp ‹_›

/-- **And it is the first one.**  Every earlier row scores strictly below it,
so a tie goes to the smaller index -- the resolution `Tensor::argmax` and the
C++ loop share, and the one the reference traces were generated under. -/
theorem firstMax_first (f : ℕ → ℝ) {n j : ℕ} (hj : j < firstMax f n) :
    f j < f (firstMax f n) := by
  induction n with
  | zero => rw [firstMax] at hj; omega
  | succ m ih =>
    rw [firstMax] at hj ⊢
    split at hj
    · rw [ite_eq_left ‹_›]
      exact lt_of_le_of_lt (firstMax_le f hj) ‹_›
    · rw [ite_eq_right ‹_›]
      exact ih hj

/-- The hypothesis of `firstMax_first` is satisfiable, at a head with a
genuine tie in it: rows `1` and `2` score `1` and row `0` scores `0`, so the
scan keeps row `1`, and row `0` is an earlier row below it. -/
example : (fun i : ℕ => if i = 0 then (0 : ℝ) else 1) 0
    < (fun i : ℕ => if i = 0 then (0 : ℝ) else 1)
      (firstMax (fun i => if i = 0 then (0 : ℝ) else 1) 3) :=
  firstMax_first (fun i : ℕ => if i = 0 then (0 : ℝ) else 1) (n := 3) (j := 0)
    (by norm_num [firstMax])

/-! ### Which makes the compressed head the same head -/

/-- Rows that score alike are ranked alike: the scan reads `f` only below
`n`. -/
theorem firstMax_congr {f g : ℕ → ℝ} {n : ℕ} (h : ∀ i < n, f i = g i) :
    firstMax f n = firstMax g n := by
  induction n with
  | zero => rfl
  | succ m ih =>
    have hm : ∀ i < m, f i = g i := fun i hi => h i (by omega)
    have hb : firstMax g m < m + 1 := by
      rcases Nat.eq_zero_or_pos m with h0 | h0
      · rw [h0, firstMax]; omega
      · exact Nat.lt_succ_of_lt (firstMax_lt g h0)
    rw [firstMax, firstMax, ih hm, h m (by omega), h (firstMax g m) hb]

/-- **The compressed head is the same head.**  If every row's kept columns
carry all of its nonzeros, the sparse scan and the dense scan return the same
index -- not a nearby one, and not one that depends on how the ties fell.

Source: `vm-rs/alm-model/src/linear.rs`, `SparseHead::argmax`. -/
theorem firstMax_sparse_eq (add mul : ℝ → ℝ → ℝ) (c : ℕ → ℕ → ℝ) (x : ℕ → ℝ)
    (keep : ℕ → ℕ → Bool) (rows cols : ℕ)
    (hdrop : ∀ i j, keep i j = false → c i j = 0)
    (hzero : ∀ (s : ℝ) (j : ℕ), add s (mul 0 (x j)) = s) :
    firstMax (fun i => sparseFold add mul (c i) x (keep i) cols) rows
      = firstMax (fun i => rowFold add mul (c i) x cols) rows :=
  firstMax_congr fun i _ => sparseFold_eq_rowFold add mul (c i) x (keep i) cols
    (hdrop i) hzero

/-- The hypotheses are satisfiable on a head with a zero in every row. -/
example (x : ℕ → ℝ) :
    firstMax (fun i => sparseFold (· + ·) (· * ·) (fun j => if j = i then 0 else 1) x
        (fun j => j != i) 3) 2
      = firstMax (fun i => rowFold (· + ·) (· * ·) (fun j => if j = i then 0 else 1) x 3) 2 :=
  firstMax_sparse_eq _ _ _ _ _ 2 3 (by intro i j hj; simpa using hj) (by intro s j; ring)

end ALM
end Transformer
