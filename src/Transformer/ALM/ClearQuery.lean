/-
# The cleared entry loses at every query the guard admits, and only there

`Transformer.ALM.ClearKey` proves the clearing mechanism sound at the query the
released model writes: `_to_2d_query` sets `qy = 1`, so subtracting `BIG` from
the key's ordinate subtracts `BIG` from the score, and `marked_sup'_eq_live`
takes it from there.  The hull path does not normalise the ordinate, and at a
general `qy` the marker arrives as `qy · BIG` — smaller, absent, or the wrong
sign.  The implementation already tests for that (`ClearGuard::dominated` in
`vm-rs/alm-hull/src/clearkey.rs` asks `2M < qy · B`); what it did not have is
the statement that the test is the right one.

Here it is, in the three pieces the code is built from:

* `dot_marked_query` — the marker passes into the score scaled by `qy`, which
  is `dot_marked` with the ordinate left free.
* `abs_dot_le_clearBound` — `ClearGuard` accumulates `max |kx|` and `max |ky|`
  at insert time and multiplies them out at query time, because the score is
  not known until the query is; this is that product bounding the score.
* `marked_sup'_eq_live_at_query` — the two composed into the guard's own test,
  and the licence for leaving the cleared entries out of a query.

And the two edges, which say the sign test is necessary rather than cautious:
`marker_vanishes_at_zero_ordinate` (at `qy = 0` the marker is not there at all,
and a cleared entry competes on its base score), and
`live_lt_cleared_of_added_marker` (at `qy < 0` the marker is *added*, and the
cleared entry beats every live one by the same margin that was supposed to sink
it).  `pos_of_margin` closes the circle: a query that passes the margin test has
`qy > 0` already, so the sign test in the code is there for the one thing the
reals do not have — a `NaN` ordinate, which fails a positive test and sends the
query to the cleared entries, the safe side.

Source: `todo3.md` §7; `transformer_vm/graph/core.py` lines 293-319;
`vm-rs/alm-hull/src/clearkey.rs`, `ClearGuard::bound` and `::dominated`.
-/

import Transformer.ALM.ClearKey

namespace Transformer
namespace ALM

/-! ### The marker at an ordinate that is not one -/

/-- **The marker arrives scaled by the query's ordinate.**  `dot_marked` is
this at `qy = 1`; every other query sees a marker of `qy · B`, which is what
makes the clearing argument a statement about the query and not only about the
keys. -/
theorem dot_marked_query (qx qy kx ky c B : ℝ) :
    dot (qx, qy) (kx, ky - c * B) = markedScore (qy * B) (dot (qx, qy) (kx, ky)) c := by
  unfold dot markedScore
  simp only []
  ring

/-- The bound `ClearGuard` carries: one number per coordinate, taken over every
entry the head holds, live and cleared alike, with the marker taken off the
cleared ones.  It is a product with the query rather than a bound on the score
because at insert time there is no query yet. -/
noncomputable def clearBound (qx qy Kx Ky : ℝ) : ℝ := |qx| * Kx + |qy| * Ky

/-- **And it bounds the score.**  This is the `M` of
`Transformer.ALM.ClearKey.marked_sup'_eq_live`, computed the way
`ClearGuard::bound` computes it. -/
theorem abs_dot_le_clearBound {qx qy kx ky Kx Ky : ℝ} (hx : |kx| ≤ Kx) (hy : |ky| ≤ Ky) :
    |dot (qx, qy) (kx, ky)| ≤ clearBound qx qy Kx Ky := by
  unfold dot clearBound
  calc |qx * kx + qy * ky| ≤ |qx * kx| + |qy * ky| := abs_add_le _ _
    _ = |qx| * |kx| + |qy| * |ky| := by rw [abs_mul, abs_mul]
    _ ≤ |qx| * Kx + |qy| * Ky := by gcongr

/-! ### The guard, as the code tests it -/

/-- **A query that clears the margin can leave the cleared entries out.**  The
hypothesis is `ClearGuard::dominated` written out: twice the bound below the
marker this query sees.  Everything else is the container's split into live
entries `L` and cleared entries `C`, and the requirement that some live entry
exists — which the code does not check and the caller owes. -/
theorem marked_sup'_eq_live_at_query {ι : Type*} [DecidableEq ι] {L C : Finset ι}
    {kx ky c : ι → ℝ} {qx qy B Kx Ky : ℝ} (hL : L.Nonempty) (hlive : ∀ i ∈ L, c i = 0)
    (hclear : ∀ i ∈ C, c i = 1) (hx : ∀ i ∈ L ∪ C, |kx i| ≤ Kx)
    (hy : ∀ i ∈ L ∪ C, |ky i| ≤ Ky) (hB : 2 * clearBound qx qy Kx Ky < qy * B) :
    (L ∪ C).sup' (hL.mono Finset.subset_union_left)
        (fun i => dot (qx, qy) (kx i, ky i - c i * B))
      = L.sup' hL (fun i => dot (qx, qy) (kx i, ky i)) := by
  simp only [dot_marked_query]
  exact marked_sup'_eq_live hL hlive hclear
    (fun i hi => abs_dot_le_clearBound (hx i hi) (hy i hi)) hB

/-- **And it has a positive ordinate, whether or not it is asked.**  A bound is
nonnegative and the marker is positive, so the margin test implies the sign
test.  In the code the sign test is kept regardless: it is written as `qy > 0`,
which a `NaN` ordinate fails, and a query that fails it consults the cleared
entries rather than skipping them. -/
theorem pos_of_margin {qy B M : ℝ} (hM : 0 ≤ M) (hB : 0 < B) (h : 2 * M < qy * B) : 0 < qy := by
  by_contra hc
  nlinarith [not_lt.mp hc]

/-! ### The two edges the guard refuses -/

/-- **At `qy = 0` there is no marker.**  The score does not read the ordinate at
all, so a cleared entry and a live one at the same abscissa carry the same
score: `marked_sup'_eq_live` has nothing to work with, and no margin can be
bought. -/
theorem marker_vanishes_at_zero_ordinate (B base c : ℝ) : markedScore (0 * B) base c = base := by
  unfold markedScore
  ring

/-- **At a negative ordinate the marker is added, and the cleared entry wins.**
Not narrowly: it wins by exactly the margin that was meant to sink it, against
every live entry at once.  That is a fact about the released representation, so
a container may not round it away — the guard has to refuse the query, and
`ClearGuard::dominated` does. -/
theorem live_lt_cleared_of_added_marker {qy B M bl bc : ℝ} (hl : |bl| ≤ M) (hc : |bc| ≤ M)
    (hB : 2 * M < -(qy * B)) : markedScore (qy * B) bl 0 < markedScore (qy * B) bc 1 := by
  have h1 := (abs_le.mp hl).2
  have h2 := (abs_le.mp hc).1
  unfold markedScore
  linarith

/-! ### The head the guard was written for -/

/-- The §4b head, whose keys sit at `3.4·10^8` and whose queries sit at
`8.6·10^9`: thirteen orders of margin under the marker, so its cleared entries
are never read.  The numbers are `alm-hull`'s own test
(`the_margin_holds_at_the_shipped_keys_and_fails_at_the_top_of_the_range`). -/
theorem the_section_4b_head_is_guarded :
    2 * clearBound 8589934503 1 673720322 113474768068945920 < 1 * (10 : ℝ) ^ 30 := by
  unfold clearBound
  rw [abs_of_nonneg (by norm_num : (0:ℝ) ≤ 8589934503), abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1)]
  norm_num

/-- And the top of the address space, where it is not: at a key of `2^52` the
score reaches `2^104`, which is past `BIG` itself, and the cleared entries have
to be consulted. -/
theorem the_top_of_the_range_is_not_guarded :
    ¬ 2 * clearBound 1 1 (2 ^ 53) (2 ^ 104) < 1 * (10 : ℝ) ^ 30 := by
  unfold clearBound
  rw [abs_of_nonneg (by norm_num : (0:ℝ) ≤ 1)]
  norm_num

/-! ### The hypotheses are satisfiable -/

/-- One live key and one cleared key at the §4b head's scale, queried at
`(8589934503, 1)`: the bounds hold and the margin clears. -/
example :
    |(673720322 : ℝ)| ≤ 673720322 ∧ |(-113474768068945920 : ℝ)| ≤ 113474768068945920 ∧
      2 * clearBound 8589934503 1 673720322 113474768068945920 < 1 * (10 : ℝ) ^ 30 := by
  refine ⟨by (rw [abs_of_nonneg]; norm_num), by rw [abs_of_nonpos] <;> norm_num,
    the_section_4b_head_is_guarded⟩

/-- `pos_of_margin` at the same numbers, which is where its conclusion is
already visible. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 10 ^ 30 ∧ 2 * (1 : ℝ) < 3 * 10 ^ 30 := by norm_num

/-- And the negative ordinate, at a marker of `10^30` and a range of one: the
cleared entry beats the live one by just under `10^30`. -/
example : |(1 : ℝ)| ≤ 1 ∧ |(-1 : ℝ)| ≤ 1 ∧ 2 * (1 : ℝ) < -(-1 * 10 ^ 30) := by
  norm_num

end ALM
end Transformer
