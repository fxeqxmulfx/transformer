/-
# The wall is in the representation, not in the arithmetic

`Transformer.ALM.FloatGrid` discharges exactness from a condition an
implementation can be checked against: a score routine that lands on the
integer grid and errs by less than a unit is exact there (`fp_eval_exact_of_grid`,
hypothesis `S.δ < 1`).  It never says where that condition stops holding, and
`todo3.md` §4 is that it stops at a nameable place: the score `2qk - k²` is an
integer, binary64 holds integers exactly only below `2^53`, so exact retrieval
needs `n < √(2^53) = 94 906 266`.

The point of this file is *which* step fails.  It is not the arithmetic.  Above
`2^p` every representable number is an even integer (`isBinary_even`), so two
distinct ones are at least two apart (`two_le_dist_of_isBinary`) — while
`Transformer.ALM.LatestWindow.one_le_sScore_sub` says distinct integer keys are
exactly one apart in score.  The separation the whole development rests on is
destroyed when the coordinate is *stored*, before any dot product runs, which is
why §4 measures the same first failure for rounded and for exact scoring:

    rounded float64 scoring                first failure at q = 94 906 266
    exact scoring on the same stored pts   first failure at q = 94 906 266

`the_first_unstorable_key` is that boundary as a decided fact, and
`the_missing_unit` is the pair of integers §4 names — `9 007 199 515 875 289`
stored as `9 007 199 515 875 288`, the lost unit being exactly the gap that made
the neighbour lose.  So no wider accumulator recovers it: Shewchuk expansions,
an 80-bit `long double`, `f128` all arrive after the loss.  Only a wider
embedding moves the wall, and that is a change to the weights.

`IsBinary p` is deliberately more generous than any real format — significand
below `2^p`, exponent unrestricted — so what it cannot hold, binary64 cannot
hold either.

Source: `todo3.md` §4, §4a and §4b; measurement and test:
`vm-rs/alm-hull/src/exact.rs::the_wall_is_the_key_coordinate_not_the_arithmetic`,
guard: `vm-rs/alm-hull/src/grid.rs`.
-/

import Transformer.ALM.LatestWindow

namespace Transformer
namespace ALM

/-! ### What a binary format can hold -/

/-- Representable with `p` significand bits: a signed significand below `2 ^ p`
times a power of two.  The exponent is unrestricted, so this contains every
binary floating-point format of `p` significand bits — what it misses, binary64
misses too. -/
def IsBinary (p : ℕ) (x : ℝ) : Prop := ∃ m e : ℤ, |m| < 2 ^ p ∧ x = (m : ℝ) * (2 : ℝ) ^ e

/-- Below the wall there is nothing to prove: an integer of magnitude under
`2 ^ p` is its own significand. -/
lemma isBinary_intCast {p : ℕ} {z : ℤ} (hz : |z| < 2 ^ p) : IsBinary p (z : ℝ) :=
  ⟨z, 0, hz, by simp⟩

/-- **Above the wall every representable number is an even integer.**  A
significand below `2 ^ p` can only reach past `2 ^ p` by an exponent of at least
one, and that factor of two is left in the result.  This is the whole of §4:
past that magnitude the format has no odd numbers at all, so a quantity that
needs one is lost on the way in. -/
lemma isBinary_even {p : ℕ} {x : ℝ} (h : IsBinary p x) (hx : (2 : ℝ) ^ p < |x|) :
    ∃ w : ℤ, x = 2 * (w : ℝ) := by
  obtain ⟨m, e, hm, rfl⟩ := h
  have hmR : |(m : ℝ)| < (2 : ℝ) ^ p := by exact_mod_cast hm
  have he : 1 ≤ e := by
    by_contra hcon
    push Not at hcon
    have he0 : e ≤ 0 := by omega
    have h2 : (2 : ℝ) ^ e ≤ 1 := zpow_le_one_of_nonpos₀ (by norm_num) he0
    have hpos : (0 : ℝ) < (2 : ℝ) ^ e := zpow_pos (by norm_num) e
    have : |(m : ℝ) * (2 : ℝ) ^ e| ≤ |(m : ℝ)| := by
      rw [abs_mul, abs_of_pos hpos]
      nlinarith [abs_nonneg ((m : ℝ))]
    linarith
  obtain ⟨n, hn⟩ : ∃ n : ℕ, e = (n : ℤ) + 1 := ⟨(e - 1).toNat, by omega⟩
  refine ⟨m * 2 ^ n, ?_⟩
  rw [hn, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), zpow_natCast]
  push_cast
  ring

/-- **So an odd integer past the wall is not representable.**  The coordinate
`-k²` of an odd key is one, and `ky` is where the compiler puts it. -/
lemma not_isBinary_odd {p : ℕ} {z : ℤ} (hodd : Odd z) (hz : (2 : ℤ) ^ p < |z|) :
    ¬ IsBinary p (z : ℝ) := by
  intro h
  have hzR : (2 : ℝ) ^ p < |(z : ℝ)| := by exact_mod_cast hz
  obtain ⟨w, hw⟩ := isBinary_even h hzR
  have : z = 2 * w := by exact_mod_cast hw
  rcases hodd with ⟨t, ht⟩
  omega

/-- **And two distinct representables past the wall are at least two apart.**
Both are even integers, so the unit gap that separates distinct integer keys
(`one_le_sScore_sub`) has nowhere to land. -/
lemma two_le_dist_of_isBinary {p : ℕ} {x y : ℝ} (hx : IsBinary p x) (hy : IsBinary p y)
    (hx' : (2 : ℝ) ^ p < |x|) (hy' : (2 : ℝ) ^ p < |y|) (hne : x ≠ y) : 2 ≤ |x - y| := by
  obtain ⟨u, rfl⟩ := isBinary_even hx hx'
  obtain ⟨v, rfl⟩ := isBinary_even hy hy'
  have huv : u ≠ v := by
    intro h; exact hne (by rw [h])
  have h1 : 1 ≤ |u - v| := Int.one_le_abs (sub_ne_zero_of_ne huv)
  have : (1 : ℝ) ≤ |((u - v : ℤ) : ℝ)| := by exact_mod_cast h1
  have hrw : 2 * (u : ℝ) - 2 * (v : ℝ) = 2 * ((u - v : ℤ) : ℝ) := by push_cast; ring
  rw [hrw, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
  linarith

/-- **The unit gap cannot be stored.**  Of two consecutive integers past the
wall, at least one is not representable at all — one of them is odd.  So a
winner that beat its runner-up by the minimum possible margin arrives at the
head already tied with it, whatever arithmetic is applied afterwards. -/
theorem unit_gap_unstorable {p : ℕ} {z : ℤ} (hz : (2 : ℤ) ^ p < z) :
    ¬ IsBinary p (z : ℝ) ∨ ¬ IsBinary p ((z + 1 : ℤ) : ℝ) := by
  have hz' : (0 : ℤ) < z := lt_of_le_of_lt (by positivity) hz
  rcases Int.even_or_odd z with he | ho
  · refine Or.inr (not_isBinary_odd ?_ ?_)
    · rcases he with ⟨t, ht⟩; exact ⟨t, by omega⟩
    · rw [abs_of_pos (by omega)]; omega
  · exact Or.inl (not_isBinary_odd ho (by rw [abs_of_pos hz']; exact hz))

/-- **The key coordinate itself is what goes.**  The compiler stores `-k²` in
`ky`; for an odd key past the wall that is an odd integer past the wall, so it
is rounded on the way in and every later reader sees the rounded one.  This is
§4's "the coordinate `-k²` stops being an exact double", stated for every
width. -/
theorem key_coord_unstorable {p : ℕ} {k : ℤ} (hodd : Odd k) (hk : (2 : ℤ) ^ p < k ^ 2) :
    ¬ IsBinary p ((-(k ^ 2) : ℤ) : ℝ) := by
  have hpos : (0 : ℤ) < k ^ 2 := lt_of_le_of_lt (by positivity) hk
  rcases hodd with ⟨t, ht⟩
  refine not_isBinary_odd ⟨-(2 * t ^ 2 + 2 * t + 1), by subst ht; ring⟩ ?_
  rw [abs_of_neg (by linarith : (-(k ^ 2) : ℤ) < 0)]
  linarith

/-! ### And the head loses exactly the gap it runs on -/

/-- The score as the integer it is: `Transformer.ALM.sScore` lands in `ℤ` at an
integer query, which is what `one_le_sScore_sub` counts in. -/
def iScore (q k : ℤ) : ℤ := 2 * k * q - k ^ 2

lemma sScore_eq_iScore (q k : ℤ) : sScore q k = ((iScore q k : ℤ) : ℝ) := by
  simp only [sScore, iScore]; push_cast; ring

/-- **So past the wall the separation the search runs on is not stored.**
`Transformer.ALM.LatestWindow.one_le_sScore_sub` says distinct integer keys are
at least a unit apart in score, and a unit is all a pair need be: for a pair
that close past `2 ^ p`, at least one of the two scores is not representable.
The comparison the search then makes is between one score and a rounding of the
other, which is why §4 finds the same first failure whether the dot product is
rounded or exact -- the loss is upstream of both. -/
theorem minimal_margin_unstorable {p : ℕ} {q j k : ℤ}
    (hlt : sScore q j < sScore q k) (hmin : sScore q k ≤ sScore q j + 1)
    (hz : (2 : ℤ) ^ p < iScore q j) :
    ¬ IsBinary p (sScore q j) ∨ ¬ IsBinary p (sScore q k) := by
  have heq : sScore q k = sScore q j + 1 := le_antisymm hmin (one_le_sScore_sub hlt)
  rw [sScore_eq_iScore, sScore_eq_iScore] at heq ⊢
  have hz' : iScore q k = iScore q j + 1 := by exact_mod_cast heq
  rw [hz']
  exact unit_gap_unstorable hz

/-- The hypotheses are satisfiable: two keys one apart, at the query that puts
their scores one apart, and a width they have already passed. -/
example : sScore 3 2 < sScore 3 3 ∧ sScore 3 3 ≤ sScore 3 2 + 1 ∧
    (2 : ℤ) ^ 2 < iScore 3 2 := by
  refine ⟨?_, ?_, ?_⟩ <;> norm_num [sScore, iScore]

/-! ### Where the wall is -/

/-- `94 906 266 = ⌈√(2^53)⌉`: the last key whose square binary64 still reaches,
and the first whose square it does not. -/
theorem the_wall : (94906265 : ℤ) ^ 2 < 2 ^ 53 ∧ (2 : ℤ) ^ 53 < 94906266 ^ 2 := by
  constructor <;> norm_num

/-- **And the first key whose coordinate cannot be stored is `94 906 267`.**  Not
`94 906 266`: that one is even, so its square is a multiple of four and survives.
The wall is reached by the first *odd* key past `√(2^53)`, which is why §4 puts
the loss of the coordinate one above the loss of the bound. -/
theorem the_first_unstorable_key : ¬ IsBinary 53 ((-((94906267 : ℤ) ^ 2) : ℤ) : ℝ) :=
  key_coord_unstorable ⟨47453133, by norm_num⟩ (by norm_num)

/-- **And the unit it loses is the one that decided the query.**  §4 reports the
stored coordinate of `94 906 267` as `9 007 199 515 875 288` against a true
`9 007 199 515 875 289`.  Both numbers are here: the true one is the square, it
is odd, and the value one below it is even — the nearest thing the format has,
one unit away, which is exactly the margin that made the neighbouring key lose.
No wider accumulator recovers it, because the loss happened before the
accumulator was reached. -/
theorem the_missing_unit :
    (94906267 : ℤ) ^ 2 = 9007199515875289 ∧ Odd ((94906267 : ℤ) ^ 2) ∧
      IsBinary 53 ((9007199515875288 : ℤ) : ℝ) := by
  refine ⟨by norm_num, ⟨4503599757937644, by norm_num⟩, ?_⟩
  exact ⟨1125899939484411, 3, by norm_num, by norm_num⟩

end ALM
end Transformer
