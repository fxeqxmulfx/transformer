/-
# One key rounded twice

`Transformer.ALM.HullSep` says a head's keys must be `sqrt A` apart for the
nearest of them to be the answer, and `alm-hull/src/sep.rs` measured the
shipped heads against that.  Three of the hundred and thirty-three fail it —
and every failing pair is between half a step and three steps of the floating
grid wide, on integer keys between `4` and `888`.  Nothing measured lands
between three representable steps and the floor of `0.6578`.  So what the
measurement found is not two keys that collided but one key the projection
rounded two or three ways, and `HullSep.nearest_fails_of_close` is the wrong
reading of it: that theorem needs the *further* key to carry the larger
offset, and here "further" is a rounding error.

The right reading is below.  Two keys `u` apart with offsets `δ < δ'` differ in
score by `δ' - δ + 2u(q - k) - u²`, so the one with the larger offset wins over
an interval of radius `(δ' - δ) / (2u) - u / 2` about the key
(`twin_later_wins`).  When `u` is a rounding error and `δ' - δ` is the gap
between two positions' recency terms, that radius is enormous: at the widest
gap measured, `2.274e-13`, and the closest two positions of the longest run,
the radius passes `1400` while the keys involved stop at `888`
(`the_measured_twins_are_invisible`).

So the earlier rounding answers no query the run can put to it
(`twin_not_greatest`), and a container that erases it returns the same answers
as one that keeps it.  That is the licence the three heads need: they do not
satisfy `Transformer.ALM.HullMark.Marked`, their container holds two or three
lines where `build_card_eq_of_marked` counts one, and neither costs an answer.
The direction is the point — the offset rises with the position, so the
survivor is the later write, which is what `TieBreak::Latest` asked for.

Source: `alm-compile/src/graph.rs` lines 231-264; `alm-hull/src/sep.rs`, whose
`twins` count is what this file is about.
-/

import Transformer.ALM.HullSep

namespace Transformer
namespace ALM

/-! ### The later of two roundings wins, and over a wide interval -/

/-- **The score difference between two keys `u` apart.**  The quadratic terms
almost cancel, leaving the offset gap plus a term linear in the query — which
is why a rounding error in the key cannot outrun a genuine difference in the
offsets until the query is very far away. -/
theorem lineEval_markKey_sub_twin (δ δ' u k q : ℝ) :
    lineEval (markKey δ' (k + u)) q - lineEval (markKey δ k) q
      = δ' - δ + 2 * u * (q - k) - u ^ 2 := by
  rw [lineEval_markKey_eq, lineEval_markKey_eq]
  ring

/-- **So the larger offset wins out to a radius of `(δ' - δ) / (2u) - u / 2`.**
`u` is the gap between the two roundings and `δ' - δ` the gap between the two
positions' recency terms; the radius is the ratio, so a rounding error loses to
any offset difference at all over a range that grows as the error shrinks.

Note what is *not* assumed.  `u` may have either sign, so this covers a
rounding that fell short of the true key as well as one that overshot — the
survivor is picked by the offset and not by which side of the key it landed on.
And `δ < δ'` is not a hypothesis but a consequence: a radius wide enough to hold
a query at all is a radius exceeding `|u| / 2`, which already says the offset
gap beats the rounding. -/
theorem twin_later_wins {δ δ' u k q : ℝ} (hu : u ≠ 0)
    (hq : |q - k| < (δ' - δ) / (2 * |u|) - |u| / 2) :
    lineEval (markKey δ k) q < lineEval (markKey δ' (k + u)) q := by
  have hu0 : 0 < |u| := abs_pos.mpr hu
  have h2u : (0 : ℝ) < 2 * |u| := by linarith
  have hkey : 2 * |u| * |q - k| < δ' - δ - |u| ^ 2 := by
    have := mul_lt_mul_of_pos_left hq h2u
    rw [mul_sub, mul_div_cancel₀ _ (ne_of_gt h2u)] at this
    nlinarith
  have hlin : -(2 * |u| * |q - k|) ≤ 2 * u * (q - k) := by
    have := abs_le_abs_of_nonneg (abs_nonneg (2 * u * (q - k))) (le_refl _)
    have habs : |2 * u * (q - k)| = 2 * |u| * |q - k| := by
      rw [abs_mul, abs_mul, abs_two]
    linarith [neg_abs_le (2 * u * (q - k)), habs ▸ le_refl (2 * |u| * |q - k|)]
  have hsq : u ^ 2 = |u| ^ 2 := (sq_abs u).symm
  have := lineEval_markKey_sub_twin δ δ' u k q
  linarith

/-- The hypotheses are satisfiable at the measured scale: two roundings of the
key `4` one representable step apart, at an offset gap of a billionth, decided
at a query a thousand key steps away. -/
example : lineEval (markKey 0 (4 : ℝ)) 1004
    < lineEval (markKey 1e-9 ((4 : ℝ) + 8.88e-16)) 1004 :=
  twin_later_wins (δ := 0) (δ' := 1e-9) (u := 8.88e-16) (k := 4) (q := 1004) (by norm_num)
    (by rw [abs_of_pos, abs_of_pos] <;> norm_num)

/-- **And so the earlier rounding is not the answer.**  Stated as the negation
of what the container would have to hold for the extra line to matter: within
the radius there is no query at which the earlier of two roundings is a
maximizer, so erasing it changes no answer and keeping it changes no answer
either — it is dead weight and nothing more. -/
theorem twin_not_greatest {δ δ' u k q : ℝ} (hu : u ≠ 0)
    (hq : |q - k| < (δ' - δ) / (2 * |u|) - |u| / 2) {s : Finset (ℝ × ℝ)}
    (hmem : markKey δ' (k + u) ∈ s) :
    ¬ ∀ l ∈ s, lineEval l q ≤ lineEval (markKey δ k) q := fun h =>
  absurd (h _ hmem) (not_le.mpr (twin_later_wins hu hq))

/-- The hypotheses are satisfiable, on the two-line container the measurement
actually found: both roundings of `4` present, and the earlier one not the
maximizer. -/
example : ¬ ∀ l ∈ ({markKey 0 (4 : ℝ), markKey 1e-9 ((4 : ℝ) + 8.88e-16)} : Finset (ℝ × ℝ)),
    lineEval l 1004 ≤ lineEval (markKey 0 (4 : ℝ)) 1004 :=
  twin_not_greatest (δ := 0) (δ' := 1e-9) (u := 8.88e-16) (k := 4) (q := 1004) (by norm_num)
    (by rw [abs_of_pos, abs_of_pos] <;> norm_num) (by simp)

/-! ### At the measured numbers -/

/-- **The radius covers every key the run put to those heads.**  `2.274e-13` is
the widest gap `alm-hull/src/sep.rs` found between two roundings of one key,
over all six reference programs; `1.2e-9` is the recency gap between the two
closest positions of the longest run that produced it; `888` is the largest key
involved in any such pair.  The radius clears the keys by a factor of three.

So on the three heads that fail `Transformer.ALM.HullSep`'s floor, every pair
that fails it is covered here instead, and the failure costs no answer.

Source: `alm-hull/src/sep.rs`, `SepWitness::twins` and `worst_at`, measured on
`data/hello.txt` through `data/min_cost_matching.txt`. -/
theorem the_measured_twins_are_invisible :
    (888 : ℝ) < 1.2e-9 / (2 * 2.274e-13) - 2.274e-13 / 2 := by
  norm_num

/-- And the tighter of the two ends measured — the longest run, whose positions
crowd the recency terms closest together — clears its own keys by more still. -/
theorem the_longest_run_is_covered_too :
    (215 : ℝ) < 1.6e-10 / (2 * 5.684e-14) - 5.684e-14 / 2 := by
  norm_num

end ALM
end Transformer
