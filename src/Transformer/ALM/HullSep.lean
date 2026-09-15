/-
# How far apart the keys have to be, and what happens below that

`Transformer.ALM.HullNear` answers a query by the nearest key, and pays for it
with integrality: `sq_dist_gap_of_near_int` uses `z ≠ z'` only to get
`1 ≤ |z' - z|`.  On the shipped model that hypothesis is not free — the keys
are a matvec through the residual stream, and `alm-hull/src/lift.rs` counts
`576` of `137 389` of them non-integer on `hello` alone.  So the real question
is not whether a key is an integer but how close two keys can get, and the
answer should be a number.

It is `sqrt A`.  With the keys `sep` apart the window around a key widens or
narrows as `(sep^2 - A) / (2 * sep)` (`sq_dist_gap_of_sep`), which is
`HullNear`'s `(1 - A) / 2` at `sep = 1` and closes entirely at `sep^2 = A`.
Below that the nearest-key rule is not merely unproved, it is false:
`nearest_fails_of_close` puts two keys `d` apart with `d^2 < A` and the further
one wins *at the nearer one's own abscissa*, which is the one query
`Transformer.ALM.HullMark.lineEval_markKey_self` was about.

For the shipped `A = LATEST_ALPHA / log 2` the floor is between `0.657` and
`0.659` (`the_shipped_separation_floor`).  That is the threshold a live head has
to clear for the ordered-set reduction to be sound, and it is a quantity a run
can measure: the minimum distance between two keys in one head.

Source: `alm-compile/src/graph.rs` lines 231-264; `alm-hull/src/lift.rs`, whose
`noninteger` count is the hypothesis this file replaces.
-/

import Transformer.ALM.HullNear

namespace Transformer
namespace ALM

/-! ### The window as a function of the separation -/

/-- **A key `sep` from its neighbours owns a window of `(sep² - A) / (2·sep)`.**
The lead of the nearer key is `M² - 2uM` in the neighbour's offset `M` and the
query's offset `u`; over `|M| ≥ sep` that is least at `|M| = sep`, where it is
`sep² - 2|u|·sep`.  Asking it to exceed the offset spread `A` is asking
`|u| < (sep² - A) / (2·sep)`, and nothing else.

`Transformer.ALM.HullNear.sq_dist_gap_of_near_int` is this at `sep = 1`. -/
theorem sq_dist_gap_of_sep {A sep q k k' : ℝ} (hA0 : 0 ≤ A) (hsep0 : 0 < sep)
    (hsep : sep ≤ |k' - k|) (hq : |q - k| < (sep ^ 2 - A) / (2 * sep)) :
    A < (q - k') ^ 2 - (q - k) ^ 2 := by
  have h2s : (0 : ℝ) < 2 * sep := by linarith
  have hbound : 2 * sep * |q - k| < sep ^ 2 - A := by
    rw [show sep ^ 2 - A = 2 * sep * ((sep ^ 2 - A) / (2 * sep)) by field_simp]
    exact mul_lt_mul_of_pos_left hq h2s
  have hup : 2 * sep * (q - k) ≤ 2 * sep * |q - k| :=
    mul_le_mul_of_nonneg_left (le_abs_self _) h2s.le
  have hun : 2 * sep * (-(q - k)) ≤ 2 * sep * |q - k| :=
    mul_le_mul_of_nonneg_left (neg_le_abs _) h2s.le
  have hrw : (q - k') ^ 2 - (q - k) ^ 2 = (k' - k) ^ 2 - 2 * (q - k) * (k' - k) := by ring
  rw [hrw]
  rcases le_or_gt (k' - k) 0 with hle | hlt
  · rw [abs_of_nonpos hle] at hsep
    have hpos : (0 : ℝ) ≤ -(k' - k) + 2 * (q - k) := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_right hsep hpos]
  · rw [abs_of_pos hlt] at hsep
    have hpos : (0 : ℝ) ≤ (k' - k) - 2 * (q - k) := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_right hsep hpos]

/-- The hypotheses are satisfiable at a separation the integer argument does
not cover: keys `0.7` apart at the shipped spread, queried `0.1` off one of
them, whose window is `(0.49 - 0.44) / 1.4 = 0.0357`… which `0.1` is *outside*.
So the witness is a query `0.03` off instead, and the narrowing is the point. -/
example : (0.44 : ℝ) < (0.03 - 0.7) ^ 2 - (0.03 - 0) ^ 2 :=
  sq_dist_gap_of_sep (k := 0) (k' := 0.7) (sep := 0.7) (by norm_num) (by norm_num)
    (by rw [abs_of_pos] <;> norm_num) (by rw [abs_lt]; norm_num)

/-- **So the head answers with the nearest key at any separation clearing the
floor.**  `Transformer.ALM.HullNear.lineEval_markKey_lt_of_near` with the unit
replaced by a measurement. -/
theorem lineEval_markKey_lt_of_sep {A sep δ δ' q k k' : ℝ} (hδ0 : 0 ≤ δ)
    (hδ'0 : 0 ≤ δ') (hδ'A : δ' ≤ A) (hsep0 : 0 < sep) (hsep : sep ≤ |k' - k|)
    (hq : |q - k| < (sep ^ 2 - A) / (2 * sep)) :
    lineEval (markKey δ' k') q < lineEval (markKey δ k) q :=
  lineEval_markKey_lt_of_gap hδ0 hδ'A
    (sq_dist_gap_of_sep (le_trans hδ'0 hδ'A) hsep0 hsep hq)

/-- The hypotheses are satisfiable, and the conclusion is the comparison the
head performs: two keys `0.7` apart, the further one carrying the whole offset,
and the nearer one still winning. -/
example : lineEval (markKey 0.44 (0.7 : ℝ)) 0.03 < lineEval (markKey 0 (0 : ℝ)) 0.03 :=
  lineEval_markKey_lt_of_sep (A := 0.44) (sep := 0.7) le_rfl (by norm_num) le_rfl
    (by norm_num) (by rw [abs_of_pos] <;> norm_num) (by rw [abs_lt]; norm_num)

/-! ### And below the floor the rule is false -/

/-- **Two keys closer than `√A` and the nearer one loses at its own
abscissa.**  `HullMark.lineEval_markKey_self` says a key scores `k² + δ` at its
own key; a neighbour `d` away scores `k² - d² + δ'`, so a later write `d` away
with `d² < δ' - δ` outscores it there.  At `δ = 0`, `δ' = A` that is `d² < A`.

This is not a rounding effect and not an edge case of the search: there is no
query at which the nearer key is the answer while the further key is nearer to
it, so the ordered-set reduction is unsound below the floor, whatever the
container is implemented with. -/
theorem nearest_fails_of_close {A d : ℝ} (hd : d ^ 2 < A) (k : ℝ) :
    lineEval (markKey 0 k) k < lineEval (markKey A (k + d)) k := by
  rw [lineEval_markKey_eq, lineEval_markKey_eq]
  have : (k - (k + d)) ^ 2 = d ^ 2 := by ring
  rw [this]
  linarith

/-- And the loser is the nearer key, at distance zero. -/
theorem nearest_fails_of_close' {A d : ℝ} (hd0 : 0 < d) (hd : d ^ 2 < A) (k : ℝ) :
    |k - k| < |k - (k + d)| ∧
      lineEval (markKey 0 k) k < lineEval (markKey A (k + d)) k := by
  refine ⟨?_, nearest_fails_of_close hd k⟩
  rw [sub_self, abs_zero, show k - (k + d) = -d by ring, abs_neg, abs_of_pos hd0]
  exact hd0

/-- The hypotheses are satisfiable at the shipped spread: two keys `0.65`
apart, which is under the floor, and the later write wins at the earlier key's
own position. -/
example : |(0 : ℝ) - 0| < |(0 : ℝ) - (0 + 0.65)| ∧
    lineEval (markKey 0 (0 : ℝ)) 0 < lineEval (markKey 0.44 ((0 : ℝ) + 0.65)) 0 :=
  nearest_fails_of_close' (A := 0.44) (by norm_num) (by norm_num) 0

/-! ### Where the floor is on the shipped weights -/

/-- **The floor is between `0.657` and `0.659`.**  `LATEST_ALPHA / log 2` is
the offset spread (`HullMark.marked_sep_of_shipped`), and the separation a live
head must clear is its square root: at `0.657` `nearest_fails_of_close` already
applies, at `0.659` `sq_dist_gap_of_sep` gives a window and the head answers
with the nearest key.  Two thirds of a key step — which unit keys clear by half
again, and which nothing in the compiler guarantees of a key that came out of a
matvec.

Source: `todo3.md` §2a for `LATEST_ALPHA`; `alm-hull/src/lift.rs` for the
non-integer count this bounds. -/
theorem the_shipped_separation_floor :
    (0.657 : ℝ) ^ 2 < 0.3 / Real.log 2 ∧ (0.3 : ℝ) / Real.log 2 < 0.659 ^ 2 := by
  have hgt : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have hlt : Real.log 2 < 0.6931471808 := Real.log_two_lt_d9
  have hpos : (0 : ℝ) < Real.log 2 := by linarith
  constructor
  · rw [lt_div_iff₀ hpos]; nlinarith
  · rw [div_lt_iff₀ hpos]; nlinarith

/-- And a live head whose keys are a unit apart clears it with room: the unit
window of `HullNear` is `0.283`, and the floor is two thirds of a step. -/
example : (0.659 : ℝ) < 1 := by norm_num

end ALM
end Transformer
