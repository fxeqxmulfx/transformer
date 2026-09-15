/-
# Why the cached breakpoints increase

`vm-rs/alm-hull/src/envelope.rs` states two invariants and rests both searches
on them: the slopes increase along the envelope, and so do the cached
breakpoints.  The first is the container's order and holds by construction.
The second is the one `Envelope::argmax` depends on -- it is why a tree ordered
by slope can be searched by breakpoint at all -- and until this file nothing
proved it.

The content is already in `Transformer.ALM.Envelope`, one adjacency at a time:
`interX_le_interX_iff` says the breakpoints at a triple are *out* of order
exactly when the middle line is redundant, which is the test both erase loops
fire on.  Negating it gives the invariant: the breakpoints are in order exactly
when no line of the triple is redundant, so a loop that erases every redundant
line leaves an ordered envelope behind.

`interX_lt_interX_widen_left` is the step the predecessor walk takes, and it is
the one that is not obvious.  The walk erases a left neighbour and then
recomputes the new line's breakpoint against the neighbour *beyond* it, without
re-running the test that decided to keep the new line in the first place.  That
is sound, and the reason is a witness point: at the crossing of the erased
neighbour with the right-hand line the new line is strictly above both, and the
line further left is below the erased one there, so it is below the new line
too.

Source: `vm-rs/alm-hull/src/envelope.rs`, `Envelope::add_line`, the predecessor
walk; `transformer_vm/attention/hull2d_cht.h` lines 122-192.
-/

import Transformer.ALM.HullCache

namespace Transformer
namespace ALM

/-! ### Where a line of smaller slope falls behind -/

/-- **Past their crossing the steeper line is ahead.**  Stated as the one
direction the witness argument below needs. -/
theorem lineEval_le_of_interX_le {l l' : ℝ × ℝ} (h : l.1 < l'.1) {x : ℝ}
    (hx : interX l l' ≤ x) : lineEval l x ≤ lineEval l' x := by
  have hd : l.1 - l'.1 < 0 := by linarith
  rw [interX, div_le_iff_of_neg hd] at hx
  unfold lineEval
  nlinarith [hx]

/-! ### The invariant, one triple at a time -/

/-- **The breakpoints at a triple are in order exactly when the middle line is
kept.**  This is `interX_le_interX_iff` negated, and it is the whole invariant:
an erase loop that drops every line failing the test leaves behind an envelope
on which the test fails nowhere, which is to say an ordered one. -/
theorem interX_lt_interX_iff {l₁ l₂ l₃ : ℝ × ℝ} (h₁₂ : l₁.1 < l₂.1) (h₂₃ : l₂.1 < l₃.1) :
    interX l₁ l₂ < interX l₂ l₃ ↔
      lineEval l₁ (interX l₁ l₃) < lineEval l₂ (interX l₁ l₃) := by
  rw [← not_le, ← not_le, not_iff_not]
  exact interX_le_interX_iff h₁₂ h₂₃

/-- **Erasing a left neighbour keeps the new line's own test passing.**  The
predecessor walk drops `B`, rebinds the left neighbour to `A` and recomputes
the breakpoint there; it never re-tests the new line `C` against the wider
pair.  It does not have to: `interX B D` is a point where `C` is strictly above
`B = D`, and `A` is below `B` there because `A` and `B` have already crossed,
so `C` is strictly above `max (A, D)` at that point and cannot be redundant.

The hypotheses are the two facts the walk has in hand -- `B` was on the
envelope between `A` and `D`, and `C` had just passed the test against `B` --
and neither mentions the erase condition, which the conclusion does not need. -/
theorem interX_lt_interX_widen_left {A B C D : ℝ × ℝ}
    (hAB : A.1 < B.1) (hBC : B.1 < C.1) (hCD : C.1 < D.1)
    (hB : interX A B < interX B D) (hC : interX B C < interX C D) :
    interX A C < interX C D := by
  have hAC : A.1 < C.1 := hAB.trans hBC
  have hBD : B.1 < D.1 := hBC.trans hCD
  set x := interX B D with hx
  have hBDx : lineEval B x = lineEval D x := lineEval_interX B D (ne_of_lt hBD)
  have hCB : lineEval B x < lineEval C x := (interX_lt_interX_iff hBC hCD).mp hC
  have hABx : lineEval A x ≤ lineEval B x := lineEval_le_of_interX_le hAB hB.le
  refine (interX_lt_interX_iff hAC hCD).mpr ?_
  by_contra hle
  rw [not_lt] at hle
  have hdom := (dominated_iff_le_at_interX hAC hCD).mpr hle x
  rcases max_cases (lineEval A x) (lineEval D x) with ⟨he, _⟩ | ⟨he, _⟩ <;>
    rw [he] at hdom <;> linarith

/-! ### The same statement about the cache the port keeps -/

/-- **And that is the comparison the code makes.**  `Transformer.ALM.HullCache`
has each node carrying `breakTo` against its successor, so the ordering of two
adjacent cached breakpoints is the ordering of two crossing points, which is
the erase test at that triple.  Nothing here reads a node's cache except
through the lines it was computed from. -/
theorem cached_lt_iff {a b c : HullNode} (hab : Cached a b) (hbc : Cached b c)
    (h₁₂ : a.1.1 < b.1.1) (h₂₃ : b.1.1 < c.1.1) :
    a.2 < b.2 ↔ lineEval a.1 (interX a.1 c.1) < lineEval b.1 (interX a.1 c.1) := by
  rw [hab, hbc]
  show ((interX a.1 b.1 : ℝ) : WithTop ℝ) < ((interX b.1 c.1 : ℝ) : WithTop ℝ) ↔ _
  rw [WithTop.coe_lt_coe]
  exact interX_lt_interX_iff h₁₂ h₂₃

/-! ### The hypotheses are satisfiable -/

/-- Four lines of increasing slope on which the widening step fires: the
parabolic lift at `k = 0, 1, 2, 3`, where no key is ever redundant and every
breakpoint test passes. -/
example :
    interX ((0 : ℝ), (0 : ℝ)) ((2 : ℝ), (-1 : ℝ)) < interX ((0 : ℝ), (0 : ℝ)) ((6 : ℝ), (-9 : ℝ)) ∧
      interX ((2 : ℝ), (-1 : ℝ)) ((4 : ℝ), (-4 : ℝ))
        < interX ((4 : ℝ), (-4 : ℝ)) ((6 : ℝ), (-9 : ℝ)) := by
  norm_num [interX]

/-- The crossing hypothesis of `lineEval_le_of_interX_le`, at the pair the
example above starts from. -/
example : interX ((0 : ℝ), (0 : ℝ)) ((2 : ℝ), (-1 : ℝ)) ≤ (1 : ℝ) := by norm_num [interX]

/-- The two cached adjacencies `cached_lt_iff` reads, on the envelope
`Transformer.ALM.the_two_line_cache` builds, with a third line beyond it. -/
example :
    Cached (((0 : ℝ), (0 : ℝ)), ((0 : ℝ) : WithTop ℝ))
        (((1 : ℝ), (0 : ℝ)), ((1 : ℝ) : WithTop ℝ)) ∧
      Cached (((1 : ℝ), (0 : ℝ)), ((1 : ℝ) : WithTop ℝ)) (((2 : ℝ), (-1 : ℝ)), (⊤ : WithTop ℝ)) := by
  constructor
  · show ((0 : ℝ) : WithTop ℝ) = ((interX ((0 : ℝ), (0 : ℝ)) ((1 : ℝ), (0 : ℝ)) : ℝ) : WithTop ℝ)
    norm_num [interX]
  · show ((1 : ℝ) : WithTop ℝ) = ((interX ((1 : ℝ), (0 : ℝ)) ((2 : ℝ), (-1 : ℝ)) : ℝ) : WithTop ℝ)
    norm_num [interX]

end ALM
end Transformer
