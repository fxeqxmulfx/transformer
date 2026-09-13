/-
# Why the binary search finds the maximum

`_HullCHT::argmax` in `transformer_vm/attention/hull2d_cht.h` (lines 203-215)
answers a query with one `lower_bound(x)` over the stored lines, falling back
to the last line when the search runs off the end.  The file names the
invariant this rests on but does not prove it (lines 95-98):

> The set ordering is by slope (`Line < Line`).
> `lower_bound(x)` relies on the invariant that `p` is increasing with slope
> among the *envelope* lines (maintained by insertion logic).

Here `p` is the breakpoint `interX` of `Transformer.ALM.Envelope`.  This
module proves that those two invariants are sufficient: under them, the index
`lower_bound` returns is a maximizer of the line value at `x`, the fallback
index `n` included.

The bridge is `lineEval_sub`: between two lines of different slope the gap is
the slope difference times the signed distance to their breakpoint, so a
single sign decides which of the two is on top.
-/

import Transformer.ALM.Envelope

namespace Transformer
namespace ALM

/-! ### Two lines -/

/-- The gap between two lines is the slope difference times the signed
distance from `x` to their breakpoint. -/
theorem lineEval_sub (l l' : ℝ × ℝ) (h : l.1 ≠ l'.1) (x : ℝ) :
    lineEval l' x - lineEval l x = (l'.1 - l.1) * (x - interX l l') := by
  unfold lineEval interX
  field_simp [sub_ne_zero_of_ne h]
  ring

/-- Past the breakpoint the steeper line is on top. -/
theorem lineEval_le_iff_interX_le {l l' : ℝ × ℝ} (h : l.1 < l'.1) (x : ℝ) :
    lineEval l x ≤ lineEval l' x ↔ interX l l' ≤ x := by
  rw [← sub_nonneg, lineEval_sub l l' (ne_of_lt h) x, ← sub_nonneg (a := x)]
  exact mul_nonneg_iff_of_pos_left (by linarith)

/-- Before the breakpoint the shallower line is on top. -/
theorem lineEval_le_iff_le_interX {l l' : ℝ × ℝ} (h : l.1 < l'.1) (x : ℝ) :
    lineEval l' x ≤ lineEval l x ↔ x ≤ interX l l' := by
  rw [← sub_nonneg, ← neg_sub (lineEval l' x), lineEval_sub l l' (ne_of_lt h) x,
    ← neg_mul, ← sub_nonneg (a := interX l l')]
  rw [show -(l'.1 - l.1) * (x - interX l l') = (l'.1 - l.1) * (interX l l' - x) by ring]
  exact mul_nonneg_iff_of_pos_left (by linarith)

/-! ### Walking along the envelope

`L` is the stored sequence of lines, indices `0 … n`, ordered by slope; the
breakpoint of the `k`-th line is `interX (L k) (L (k+1))`.
-/

variable {n : ℕ} (L : ℕ → ℝ × ℝ) (x : ℝ)

/-- Crossing a breakpoint that lies at or before `x` never decreases the line
value at `x`, so walking right while the breakpoints are behind `x` only
improves. -/
theorem lineEval_mono_right (hslope : ∀ j, j < n → (L j).1 < (L (j + 1)).1)
    (i j : ℕ) (hij : i ≤ j) :
    j ≤ n → (∀ k, i ≤ k → k < j → interX (L k) (L (k + 1)) ≤ x) →
      lineEval (L i) x ≤ lineEval (L j) x := by
  induction j, hij using Nat.le_induction with
  | base => intro _ _; exact le_rfl
  | succ j hij ih =>
    intro hj h
    have hjn : j < n := by omega
    have h1 : lineEval (L i) x ≤ lineEval (L j) x :=
      ih (by omega) fun k hk hk' => h k hk (by omega)
    have h2 : lineEval (L j) x ≤ lineEval (L (j + 1)) x :=
      (lineEval_le_iff_interX_le (hslope j hjn) x).mpr (h j hij (by omega))
    exact h1.trans h2

/-- Symmetrically, while the breakpoints are still ahead of `x`, walking right
only makes the line value at `x` worse. -/
theorem lineEval_anti_right (hslope : ∀ j, j < n → (L j).1 < (L (j + 1)).1)
    (i j : ℕ) (hij : i ≤ j) :
    j ≤ n → (∀ k, i ≤ k → k < j → x ≤ interX (L k) (L (k + 1))) →
      lineEval (L j) x ≤ lineEval (L i) x := by
  induction j, hij using Nat.le_induction with
  | base => intro _ _; exact le_rfl
  | succ j hij ih =>
    intro hj h
    have hjn : j < n := by omega
    have h1 : lineEval (L j) x ≤ lineEval (L i) x :=
      ih (by omega) fun k hk hk' => h k hk (by omega)
    have h2 : lineEval (L (j + 1)) x ≤ lineEval (L j) x :=
      (lineEval_le_iff_le_interX (hslope j hjn) x).mpr (h j hij (by omega))
    exact h2.trans h1

/-! ### The query -/

/-- **`lower_bound` returns a maximizer.**  `hslope` is the set ordering by
slope; `hbp` is the breakpoint invariant the file states; `hlt` and `hge`
together say that `i` is what `lower_bound(x)` yields — every earlier
breakpoint is strictly before `x`, and `i`'s own breakpoint is at or after it.
The fallback branch `it == end` is the case `i = n`, where `hge` is vacuous. -/
theorem lowerBound_isGreatest
    (hslope : ∀ j, j < n → (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → b < n → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1)))
    (i : ℕ) (hi : i ≤ n)
    (hlt : ∀ j, j < i → interX (L j) (L (j + 1)) < x)
    (hge : i < n → x ≤ interX (L i) (L (i + 1))) :
    ∀ j ≤ n, lineEval (L j) x ≤ lineEval (L i) x := by
  intro j hj
  rcases le_total j i with hji | hij
  · exact lineEval_mono_right L x hslope j i hji hi
      fun k _ hk' => (hlt k hk').le
  · refine lineEval_anti_right L x hslope i j hij hj fun k hk hk' => ?_
    have hkn : k < n := by omega
    have hin : i < n := by omega
    exact le_trans (hge hin) (hbp i k hk hkn)

/-- The hypotheses are satisfiable, and not only vacuously: two lines meeting
at the origin, queried at the origin, with `lower_bound` landing on the first
one. -/
example :
    let L : ℕ → ℝ × ℝ := fun k => if k = 0 then (0, 0) else (1, 0)
    (∀ j, j < 1 → (L j).1 < (L (j + 1)).1) ∧
      (∀ a b, a ≤ b → b < 1 → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1))) ∧
      (∀ j, j < 0 → interX (L j) (L (j + 1)) < (0 : ℝ)) ∧
      ((0 : ℕ) < 1 → (0 : ℝ) ≤ interX (L 0) (L (0 + 1))) := by
  intro L
  refine ⟨fun j hj => ?_, fun a b hab hb => ?_, fun j hj => absurd hj (by omega), fun _ => ?_⟩
  · interval_cases j
    norm_num [L]
  · interval_cases b
    interval_cases a
    exact le_rfl
  · norm_num [L, interX]

end ALM
end Transformer
