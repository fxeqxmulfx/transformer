/-
# Reading the sign off the top component

`Transformer.ALM.Expansion` proves what the sweep preserves: the components
sum to the terms, exactly, with nothing about floating point involved.  That
is the half a transcription error would break.  It is not yet the predicate,
because `exact.rs::expansion_sign` does not sum the components — it returns

    e[len - 1].partial_cmp(&0.0)

the sign of the last one alone, and an empty expansion as `Equal`.

Reading one component for the sign of all of them is sound exactly when the
last dominates the rest.  That is Shewchuk's non-overlapping invariant, and it
is the one step here that is a fact about binary floating point rather than
about algebra, so it enters as the hypothesis `|lows.sum| < |top|` and not as
a proof.  Everything else is discharged: `sign_of_top` is the reading,
`expansion_sign_eq` is it composed with `expansion_sum`, and
`expansion_nil_sum` is the empty case, which needs no hypothesis at all.

What the composition says is the contract `head.rs` relies on when it calls
the exact scan the arbiter of a query the error bound could not decide: the
sign the predicate returns is the sign of the exact sum of its terms.

Source: Shewchuk 1997, §2 (non-overlapping expansions) and the tail of
`vm-rs/alm-hull/src/exact.rs::expansion_sign`.
-/

import Transformer.ALM.Expansion

namespace Transformer
namespace ALM

/-! ### The empty expansion -/

/-- **Nothing left means nothing there.**  The runtime returns `Equal` on an
empty expansion, and it is entitled to: the components sum to the terms, so no
components means the terms cancelled exactly. -/
theorem expansion_nil_sum (F : ExactSum) {ts : List ℝ} (h : expansion F ts = []) :
    ts.sum = 0 := by
  have := expansion_sum F ts
  rw [h] at this
  simpa using this.symm

/-- The hypothesis is satisfiable: exact addition on no terms at all. -/
example : ([] : List ℝ).sum = 0 := expansion_nil_sum exactAdd (ts := []) rfl

/-! ### The top component decides -/

/-- **A dominated tail cannot change the sign.**  If the components below the
last sum to less than it in absolute value, the total has the last one's sign
— which is what makes reading `e[len - 1]` alone the right thing to do.

Source: Shewchuk 1997, §2; `expansion_sign`'s final `partial_cmp`. -/
theorem sign_of_top {lows : List ℝ} {top : ℝ} (h : |lows.sum| < |top|) :
    (0 < (lows ++ [top]).sum ↔ 0 < top) ∧ ((lows ++ [top]).sum < 0 ↔ top < 0) := by
  have hsum : (lows ++ [top]).sum = lows.sum + top := by simp
  obtain ⟨hlo, hhi⟩ := abs_lt.mp h
  rw [hsum]
  rcases le_or_gt top 0 with ht | ht
  · rw [abs_of_nonpos ht] at hlo hhi
    constructor <;> constructor <;> intro hc <;> linarith
  · rw [abs_of_pos ht] at hlo hhi
    constructor <;> constructor <;> intro hc <;> linarith

/-- The hypothesis is satisfiable and does real work: `[1, -2]` is dominated
by `10`, and the total is positive because `10` is. -/
example : (0 : ℝ) < ([1, -2] ++ [10] : List ℝ).sum ↔ (0 : ℝ) < 10 :=
  (sign_of_top (lows := [1, -2]) (top := 10) (by norm_num)).1

/-! ### Which is the predicate -/

/-- **`expansion_sign` returns the sign of the exact sum of its terms.**  The
components sum to the terms (`expansion_sum`, no rounding anywhere), and a
non-overlapping expansion is decided by its last component (`sign_of_top`),
so the one comparison the runtime performs at the end answers about the whole.

This is the contract `vm-rs/alm-hull/src/head.rs` leans on when it makes the
exact scan the arbiter of a query `Transformer.ALM.DotError.cmp_of_dot_guard`
could not decide: what comes back is the exact order, not a better estimate
of it.

Source: `exact.rs::expansion_sign`; Shewchuk 1997, §2. -/
theorem expansion_sign_eq (F : ExactSum) {ts lows : List ℝ} {top : ℝ}
    (hexp : expansion F ts = lows ++ [top]) (hdom : |lows.sum| < |top|) :
    (0 < ts.sum ↔ 0 < top) ∧ (ts.sum < 0 ↔ top < 0) := by
  have hsum : ts.sum = (lows ++ [top]).sum := by rw [← hexp, expansion_sum]
  rw [hsum]
  exact sign_of_top hdom

/-- The hypotheses hold together on a term list the machine could hand over:
exact addition leaves `[1, -2, 3]` as the single component `2`, which
dominates the empty tail, and the sum is positive because `2` is. -/
example : (0 : ℝ) < ([1, -2, 3] : List ℝ).sum ↔ (0 : ℝ) < 2 :=
  (expansion_sign_eq exactAdd (ts := [1, -2, 3]) (lows := []) (top := 2)
    (by norm_num [expansion, push, grow, exactAdd]) (by norm_num)).1

end ALM
end Transformer
