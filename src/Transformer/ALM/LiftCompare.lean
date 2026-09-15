/-
# The abscissa outlives the ordinate, and carries the whole comparison

`Transformer.ALM.ScoreWall` puts the wall in the stored coordinate: past
`⌈√(2^53)⌉ = 94 906 266` the ordinate `-k²` of a key is not a binary64 number
(`key_coord_unstorable`), the unit that separated two scores is gone before any
dot product runs (`minimal_margin_unstorable`), and no wider accumulator
recovers it.  What that file leaves open is what the head is supposed to do
about it, and its own answer — a wider embedding — is a change to the weights.

There is a cheaper one, and this file is it.  The key reaches the head as the
point `(2k, -k² + δ)`, and the *abscissa* is `2k`: an integer of one more bit
than `k`, so it is stored exactly to `|k| ≤ 2^52`
(`the_abscissa_and_its_half_are_both_storable`) long past the magnitude at which
the ordinate stops being stored at all (`the_section_4b_key` exhibits both at
one key).  And for a live key the ordinate is not independent information — it
is `-k²` plus a recency term `δ` the compiler adds, bounded by `LATEST_ALPHA /
log 2 < 1/2` (`the_shipped_spread`).  So the head need not read the ordinate: it
can recover `k` from the abscissa and rebuild the comparison from `k`.

`upper_lt_iff` and `lower_lt_iff` are that rebuilt comparison, and they say it
is the *same* comparison — not an approximation of it.  Against an integer
query the score of a marked key splits as an integer plus an offset
(`dot_markKey_upper`, `dot_markKey_lower`), and `lt_iff` is the one inequality
that makes the split decide: two offsets under a half cannot bridge a unit, so
the integer part orders the keys and the offset is consulted only where the
integers tie.  That is `HullNear` without its window hypothesis — `HullNear`
buys a strict win by putting the query inside `(1 - A)/2` of a key, which needs
the query to be near *some* key; here the query is an integer and the
comparison is total, ties included, which is what a container has to answer.

The integers it runs on are squared distances, and `sq_dist_le` bounds them:
at `|k|, |q| ≤ 2^52` a squared distance is at most `2^106`, which is `i128` with
twenty-one bits to spare, so the comparison the machine makes is exact
arithmetic and not floating point at all.  The wall moves from `2^26.5` to
`2^52`, in the same two dimensions and at the same weights.

Source: `todo3.md` §4, §4a, §4b; `vm-rs/alm-hull/src/liftkey.rs`
(`LiftKey::of`, `UnitQuery::cmp`, `KEY_LIMIT`).
-/

import Transformer.ALM.HullMark
import Transformer.ALM.ScoreWall

namespace Transformer
namespace ALM
namespace LiftCompare

/-! ### What survives the store -/

/-- **The abscissa is an integer of one more bit, and both it and the integer
under it are representable.**  `embed_key` emits `2k`, so recovering `k` is a
halving, and a halving of a representable number is exact: the two share a
significand.  The bound `2^52` is `KEY_LIMIT`, and it is one bit short of the
`2^53` a binary64 significand holds precisely because the stored number is
`2k`. -/
theorem the_abscissa_and_its_half_are_both_storable {k : ℤ} (hk : |k| ≤ 2 ^ 52) :
    IsBinary 53 ((2 * k : ℤ) : ℝ) ∧ IsBinary 53 ((k : ℤ) : ℝ) := by
  have hlt : |k| < 2 ^ 53 := lt_of_le_of_lt hk (by norm_num)
  exact ⟨⟨k, 1, hlt, by push_cast; ring⟩, ⟨k, 0, hlt, by simp⟩⟩

/-- The hypothesis is satisfiable at the largest key it admits. -/
example : IsBinary 53 ((2 * (2 ^ 52 : ℤ) : ℤ) : ℝ) :=
  (the_abscissa_and_its_half_are_both_storable (k := 2 ^ 52) (by norm_num)).1

/-- **And at the key `todo3.md` §4b names, the two parts of the point part
company.**  `0x20202020 = 673 720 322` is the abscissa the shipped model
reaches, so the key is `336 860 161`; its square is twenty-six binades past
`2^53` and is not stored (`ScoreWall.key_coord_unstorable`), while the abscissa
is stored exactly.  The head is handed a wrong ordinate and a right abscissa,
which is why reading the key off the abscissa is not a refinement of the old
path but a different one. -/
theorem the_section_4b_key :
    ¬ IsBinary 53 ((-((336860161 : ℤ) ^ 2) : ℤ) : ℝ) ∧
      IsBinary 53 ((2 * 336860161 : ℤ) : ℝ) :=
  ⟨key_coord_unstorable ⟨168430080, by norm_num⟩ (by norm_num),
    (the_abscissa_and_its_half_are_both_storable (k := 336860161) (by norm_num)).1⟩

/-! ### An integer gap is not bridged by two offsets under a half -/

/-- **The integer part decides.**  Two offsets bounded by `A < 1/2` differ by
less than one, and distinct integers differ by at least one, so the integer
order is the order of the sum.  This is the inequality `not_eraseStep_of_marked`
spends to keep every key in the container, spent here on the comparison
instead. -/
theorem lt_of_int_lt {A δ δ' : ℝ} {n n' : ℤ} (hA : A < 1 / 2) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) (h : n < n') : (n : ℝ) + δ < (n' : ℝ) + δ' := by
  have hz : (n : ℝ) + 1 ≤ (n' : ℝ) := by exact_mod_cast Int.add_one_le_iff.mpr h
  obtain ⟨_, h2⟩ := abs_le.mp hδ
  obtain ⟨h1', _⟩ := abs_le.mp hδ'
  linarith

/-- The hypotheses are satisfiable at the shipped spread and at its extremes:
the offsets `±0.3 / log 2` against the keys `0` and `1`. -/
example : ((0 : ℤ) : ℝ) + 0.3 / Real.log 2 < ((1 : ℤ) : ℝ) + -(0.3 / Real.log 2) := by
  refine lt_of_int_lt (A := 0.3 / Real.log 2) ?_ (le_of_eq (abs_of_nonneg ?_)) ?_ (by decide)
  · rw [div_lt_div_iff₀ (Real.log_pos (by norm_num)) (by norm_num)]
    linarith [lt_trans (by norm_num : (0.6 : ℝ) < 0.6931471803) Real.log_two_gt_d9]
  · exact div_nonneg (by norm_num) (Real.log_nonneg (by norm_num))
  · rw [abs_neg, abs_of_nonneg (div_nonneg (by norm_num) (Real.log_nonneg (by norm_num)))]

/-- **So the comparison is lexicographic.**  An integer part, and the offset
read only where the integers are equal: exactly the two branches of
`UnitQuery::cmp`, and the reason the second one can be a plain float compare
with nothing proved about its precision — it is never reached unless the
first tied. -/
theorem lt_iff {A δ δ' : ℝ} {n n' : ℤ} (hA : A < 1 / 2) (hδ : |δ| ≤ A) (hδ' : |δ'| ≤ A) :
    (n : ℝ) + δ < (n' : ℝ) + δ' ↔ n < n' ∨ (n = n' ∧ δ < δ') := by
  constructor
  · intro h
    rcases lt_trichotomy n n' with hlt | heq | hgt
    · exact Or.inl hlt
    · subst heq; exact Or.inr ⟨rfl, by linarith⟩
    · exact absurd (lt_of_int_lt hA hδ' hδ hgt) (by linarith)
  · rintro (hlt | ⟨rfl, hd⟩)
    · exact lt_of_int_lt hA hδ hδ' hlt
    · linarith

/-- The hypotheses are satisfiable, and the equal-integer branch is reachable:
the same key at two offsets is ordered by the offsets alone. -/
example : ((7 : ℤ) : ℝ) + 0 < ((7 : ℤ) : ℝ) + 0.25 :=
  (lt_iff (A := 0.25) (by norm_num) (by norm_num) (by norm_num)).mpr
    (Or.inr ⟨rfl, by norm_num⟩)

/-! ### The score of a marked key at an integer query -/

/-- **The upper envelope.**  Against `(q, 1)` a marked key scores
`2kq - k² + δ = (q² - (k - q)²) + δ`, and at an integer key and an integer query
the bracket is an integer: the split `lt_iff` wants, with the squared distance
inside it. -/
theorem dot_markKey_upper (δ : ℝ) (z q : ℤ) :
    dot ((q : ℝ), 1) (markKey δ (z : ℝ)) = ((q ^ 2 - (z - q) ^ 2 : ℤ) : ℝ) + δ := by
  unfold dot markKey
  push_cast
  ring

/-- **The lower envelope.**  Against `(q, -1)` the sign of the ordinate flips
and the score is `2kq + k² - δ = ((k + q)² - q²) - δ`: the same split with the
squared distance to `-q` and the offset reversed, which is why `UnitQuery::cmp`
negates both halves rather than one. -/
theorem dot_markKey_lower (δ : ℝ) (z q : ℤ) :
    dot ((q : ℝ), -1) (markKey δ (z : ℝ)) = (((z + q) ^ 2 - q ^ 2 : ℤ) : ℝ) + -δ := by
  unfold dot markKey
  push_cast
  ring

/-! ### And that split is the whole comparison -/

/-- **The rebuilt comparison, upper envelope.**  The key at least squared
distance from the query wins, and the recency offsets decide only a tie in
squared distance.  Nothing here is rounded: the left side is the score the head
would compute and the right side is a comparison of integers.  This is
`UnitQuery::cmp` at `upper = true`. -/
theorem upper_lt_iff {A δ δ' : ℝ} {z z' q : ℤ} (hA : A < 1 / 2) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) :
    dot ((q : ℝ), 1) (markKey δ (z : ℝ)) < dot ((q : ℝ), 1) (markKey δ' (z' : ℝ)) ↔
      (z' - q) ^ 2 < (z - q) ^ 2 ∨ ((z - q) ^ 2 = (z' - q) ^ 2 ∧ δ < δ') := by
  rw [dot_markKey_upper, dot_markKey_upper, lt_iff hA hδ hδ', sub_lt_sub_iff_left,
    sub_right_inj]

/-- **The rebuilt comparison, lower envelope.**  Here the score is convex in the
key, so the maximum is the key *furthest* from `-q` and the larger offset loses.
This is `UnitQuery::cmp` at `upper = false`, and the half the hull keeps two
lines of at any length (`HullLower`). -/
theorem lower_lt_iff {A δ δ' : ℝ} {z z' q : ℤ} (hA : A < 1 / 2) (hδ : |δ| ≤ A)
    (hδ' : |δ'| ≤ A) :
    dot ((q : ℝ), -1) (markKey δ (z : ℝ)) < dot ((q : ℝ), -1) (markKey δ' (z' : ℝ)) ↔
      (z + q) ^ 2 < (z' + q) ^ 2 ∨ ((z + q) ^ 2 = (z' + q) ^ 2 ∧ δ' < δ) := by
  rw [dot_markKey_lower, dot_markKey_lower, lt_iff hA (abs_neg δ ▸ hδ) (abs_neg δ' ▸ hδ'),
    sub_lt_sub_iff_right, sub_left_inj, neg_lt_neg_iff]

/-- The hypotheses are satisfiable at the shipped spread, and both branches of
both envelopes are reachable: at the query `0` the key `0` beats the key `1`
above and loses to it below. -/
example : dot (((0 : ℤ) : ℝ), 1) (markKey 0 ((1 : ℤ) : ℝ)) <
      dot (((0 : ℤ) : ℝ), 1) (markKey 0 ((0 : ℤ) : ℝ)) ∧
    dot (((0 : ℤ) : ℝ), -1) (markKey 0 ((0 : ℤ) : ℝ)) <
      dot (((0 : ℤ) : ℝ), -1) (markKey 0 ((1 : ℤ) : ℝ)) :=
  ⟨(upper_lt_iff (A := 0) (by norm_num) (by norm_num) (by norm_num)).mpr
      (Or.inl (by decide)),
    (lower_lt_iff (A := 0) (by norm_num) (by norm_num) (by norm_num)).mpr
      (Or.inl (by decide))⟩

/-! ### The width it runs in, and the wall it moves -/

/-- **A squared distance at the new limit is `2^106`.**  `KEY_LIMIT = 2^52`
bounds the key and `on_the_grid` bounds the query the same way, so `k ∓ q` is at
most `2^53` and its square at most `2^106` — twenty-one bits inside a signed
`i128`, which is what lets `UnitQuery::cmp` compute it and not estimate it. -/
theorem sq_dist_le {z q : ℤ} (hz : |z| ≤ 2 ^ 52) (hq : |q| ≤ 2 ^ 52) :
    (z - q) ^ 2 ≤ 2 ^ 106 ∧ (z + q) ^ 2 ≤ 2 ^ 106 := by
  obtain ⟨h1, h2⟩ := abs_le.mp hz
  obtain ⟨h3, h4⟩ := abs_le.mp hq
  constructor <;> nlinarith

/-- And `2^106` is not near the edge of the width: twenty bits of headroom are
left over the `i128` maximum, so no pair of keys the abscissa can hold comes
near overflowing the comparison. -/
theorem sq_dist_fits : (2 : ℤ) ^ 106 * 2 ^ 20 < 2 ^ 127 - 1 := by norm_num

/-- The hypotheses of `sq_dist_le` are satisfiable at the extreme pair. -/
example : ((2 ^ 52 : ℤ) + 2 ^ 52) ^ 2 ≤ 2 ^ 106 :=
  (sq_dist_le (z := 2 ^ 52) (q := 2 ^ 52) (by norm_num) (by norm_num)).2

/-- **The shipped offset spread clears a half.**  `LATEST_ALPHA = 0.3` and the
recency term is `0.3 · inv_log_pos p ∈ [0, 0.3 / log 2]`
(`HullMark.marked_sep_of_shipped` bounds it by one; this is the same quantity
bounded by a half), so the shipped model discharges `hA` in every theorem above.
`LiftKey::of` tests `|δ| ≤ MARK_SPREAD` with `MARK_SPREAD = 0.3 / log 2`, and
that test is exactly this hypothesis. -/
theorem the_shipped_spread : (0.3 : ℝ) / Real.log 2 < 1 / 2 := by
  have h : (0.6 : ℝ) < Real.log 2 :=
    lt_trans (by norm_num) Real.log_two_gt_d9
  rw [div_lt_div_iff₀ (by linarith) (by norm_num)]
  linarith

/-- **So the query §4b loses is answered.**  At the shipped key `336 860 161`,
whose ordinate is not stored (`the_section_4b_key`), the query equal to the key
puts that key strictly ahead of its neighbour whatever the two recency offsets
are — because the comparison is `0 < 1` between two squared distances, and the
offsets cannot reach across it.  The measured failure of §4b is an inversion of
these two, so this is the defect closed rather than bounded. -/
theorem the_section_4b_query {A δ δ' : ℝ} (hA : A < 1 / 2) (hδ : |δ| ≤ A) (hδ' : |δ'| ≤ A) :
    dot (((336860161 : ℤ) : ℝ), 1) (markKey δ' ((336860162 : ℤ) : ℝ)) <
      dot (((336860161 : ℤ) : ℝ), 1) (markKey δ ((336860161 : ℤ) : ℝ)) :=
  (upper_lt_iff hA hδ' hδ).mpr (Or.inl (by decide))

/-- The hypotheses are satisfiable at the spread the shipped weights emit. -/
example : dot (((336860161 : ℤ) : ℝ), 1) (markKey (0.3 / Real.log 2) ((336860162 : ℤ) : ℝ)) <
    dot (((336860161 : ℤ) : ℝ), 1) (markKey 0 ((336860161 : ℤ) : ℝ)) :=
  the_section_4b_query the_shipped_spread
    (by rw [abs_zero]; exact div_nonneg (by norm_num) (Real.log_nonneg (by norm_num)))
    (le_of_eq (abs_of_nonneg (div_nonneg (by norm_num) (Real.log_nonneg (by norm_num)))))

end LiftCompare
end ALM
end Transformer
