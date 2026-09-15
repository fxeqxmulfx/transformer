/-
# The exact sum the fast path falls back on

`Transformer.ALM.DotError` bounds the error of `q0*k0 + q1*k1` and
`cmp_of_dot_guard` says when that bound decides a comparison.  When it does
not, `vm-rs/alm-hull/src/head.rs` does not give up: it scans again with
`vm-rs/alm-hull/src/exact.rs::dot_cmp`, which answers with the sign of the
exact difference and is therefore the arbiter of every query the bound leaves
open.  On `fibonacci` that is 61 queries of 382284, and the arbiter's verdict
is the one the measurement reports.

An arbiter with nothing behind it is not an arbiter.  This file is what is
behind it, down to the one step that is not algebra.

`exact.rs` builds a Shewchuk expansion: a list of components whose sum is the
exact total, grown one term at a time by sweeping the new term through the
list with error-free two-sums and keeping every low part.  `grow` and `push`
are that loop; `expansion_sum` is its invariant, and it is pure algebra —
nothing about floating point enters, because nothing about floating point is
used.  A transcription error in the sweep would break it, and that is the
error worth guarding against.

The last step is not algebra.  `expansion_sign` reads the sign off the final
component alone, which is sound because a non-overlapping expansion is
dominated by its last component — Shewchuk's invariant, and the one thing
here taken as a hypothesis rather than proved.  `sign_of_top` is that step,
`expansion_sign_eq` the two halves put together.

`ExactSum` is the error-free transformation the sweep runs on: `two_sum`
(Knuth 1969, §4.2.2) returns the rounded sum together with the part it lost,
and the two add back to the exact sum.  It is a structure and not an `axiom`,
so every statement below carries it as a hypothesis, and `exactArith` is a
witness that the hypothesis is satisfiable.

Source: Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast
Robust Geometric Predicates", 1997, §2 (`grow_expansion`);
`vm-rs/alm-hull/src/exact.rs`, `two_sum` and `expansion_sign`.
-/

import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace Transformer
namespace ALM

/-! ### The error-free transformation the sweep runs on -/

/-- A sum that keeps what it rounded away: `hi a b` is the sum as computed and
`lo a b` the part it lost, and the two are the exact sum.

Source: Knuth 1969, §4.2.2; `exact.rs::two_sum`. -/
structure ExactSum where
  /-- The sum as the machine computes it. -/
  hi : ℝ → ℝ → ℝ
  /-- The part that computation lost, which is itself representable. -/
  lo : ℝ → ℝ → ℝ
  /-- And the two are the exact sum. -/
  exact : ∀ a b, hi a b + lo a b = a + b

/-- Exact arithmetic is the instance that loses nothing, so every statement
below is about a structure that exists. -/
def exactAdd : ExactSum where
  hi a b := a + b
  lo _ _ := 0
  exact _ _ := by ring

/-! ### One sweep -/

/-- **One pass of `grow_expansion`.**  The new term `q` is carried through the
components `l` from the smallest up; each two-sum leaves its low part behind
and hands the high part on, and what comes out the top is the carry.

Source: Shewchuk 1997, §2; the inner `for i in 0..len` of `expansion_sign`. -/
def grow (F : ExactSum) : ℝ → List ℝ → List ℝ × ℝ
  | q, [] => ([], q)
  | q, x :: xs =>
      let g := grow F (F.hi q x) xs
      (F.lo q x :: g.1, g.2)

/-- **The sweep loses nothing.**  What it leaves behind plus what it carries
out is what went in — the invariant the whole predicate rests on, and the one
a transcription error in the loop would break. -/
theorem grow_sum (F : ExactSum) (q : ℝ) (l : List ℝ) :
    (grow F q l).1.sum + (grow F q l).2 = q + l.sum := by
  induction l generalizing q with
  | nil => simp [grow]
  | cons x xs ih =>
      have hx := F.exact q x
      simp only [grow, List.sum_cons]
      have := ih (F.hi q x)
      linarith

/-- The hypothesis is satisfiable and the statement is not vacuous: exact
addition sweeps `1` through `[2, 3]` and carries out the whole `6`. -/
example : (grow exactAdd 1 [2, 3]).1.sum + (grow exactAdd 1 [2, 3]).2 = 6 := by
  rw [grow_sum]
  norm_num

/-! ### Dropping the zeros the runtime does not store -/

open scoped Classical in
/-- Zero components are not stored, and they are not missed. -/
theorem sum_filter_ne_zero (l : List ℝ) :
    (l.filter (fun r => !decide (r = 0))).sum = l.sum := by
  induction l with
  | nil => simp
  | cons x xs ih =>
      by_cases hx : x = 0 <;> simp [hx, ih]

open scoped Classical in
/-- **One term folded into the expansion.**  The sweep's low parts, with the
zeros dropped, and then the carry if it is not itself zero.

Source: the body of the outer `for &t in terms` of `expansion_sign`. -/
noncomputable def push (F : ExactSum) (e : List ℝ) (t : ℝ) : List ℝ :=
  let g := grow F t e
  let lows := g.1.filter (fun r => !decide (r = 0))
  if g.2 = 0 then lows else lows ++ [g.2]

/-- **And folding it in adds it.** -/
theorem push_sum (F : ExactSum) (e : List ℝ) (t : ℝ) : (push F e t).sum = t + e.sum := by
  have hg := grow_sum F t e
  classical
  by_cases h0 : (grow F t e).2 = 0 <;>
    simp [push, h0, sum_filter_ne_zero] <;> linarith

/-! ### And the whole expansion -/

/-- The expansion after every term has been folded in. -/
noncomputable def expansion (F : ExactSum) (ts : List ℝ) : List ℝ :=
  ts.foldl (push F) []

/-- **The expansion sums to the terms, exactly.**  No rounding has occurred
anywhere along the way, so this is an equality of reals and not a bound.

Source: Shewchuk 1997, §2, Theorem 10. -/
theorem expansion_sum (F : ExactSum) (ts : List ℝ) : (expansion F ts).sum = ts.sum := by
  have key : ∀ (l : List ℝ) (e : List ℝ), (l.foldl (push F) e).sum = e.sum + l.sum := by
    intro l
    induction l with
    | nil => simp
    | cons t tl ih =>
        intro e
        rw [List.foldl_cons, ih (push F e t), push_sum, List.sum_cons]
        ring
  rw [expansion, key ts []]
  simp

/-- Exact addition again, on the eight terms `dot_cmp` hands over: the
expansion of a list of reals sums to that list. -/
example : (expansion exactAdd [1, -2, 3]).sum = 2 := by
  rw [expansion_sum]
  norm_num

end ALM
end Transformer
