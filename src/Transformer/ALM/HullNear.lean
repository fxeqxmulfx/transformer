/-
# The live head answers with the key nearest the query

`Transformer.ALM.HullMark` settles what the container *holds* — every key
inserted, because the recency offset is too small to make an erase fire.  It
says nothing about what the container *answers*, and everything it proves about
the offset is proved at one query, `x = k`, the abscissa of the key itself.
That left the reduction the machine actually wants unlicensed: a lookup is an
argmax over a general query, and the claim that it returns the nearest key is
the claim that turns the hull into an ordered set of integers.

Here it is.  Completing the square,

    lineEval (markKey d k) q = q^2 - (q - k)^2 + d,

so the query's own term cancels out of every comparison and the head is
minimising `(q - k)^2 - d`: squared distance, tilted by the offset.
`lineEval_markKey_lt_of_gap` is that read as a margin — a key wins whenever its
squared-distance lead exceeds the whole offset spread `A` — and it needs no
integrality at all.

`sq_dist_gap_of_near_int` is where the integrality is spent.  For integer keys
the lead of the nearest key over any other is at least `1 - 2|q - z|`, so a
query within `(1 - A) / 2` of a key present in the head is answered by that key
and by nothing else (`lineEval_markKey_lt_of_near`, `isGreatest_of_near`).  For
the shipped `A = LATEST_ALPHA / log 2` the window is `0.283`
(`the_shipped_window`), and `ALM.QueryScale` measures the machine's queries at
`2e-6` off an integer, five orders inside it.

What is *not* claimed is the converse, and the converse is false: at the
midpoint of two keys the squared distances are equal and the offsets decide,
which is exactly what the compiler added them for.  `the_window_is_needed`
exhibits that, so the window above is not slack to be optimised away.

Source: `alm-compile/src/graph.rs` lines 231-269 (`embed_key`, `embed_query`);
`alm-hull/src/head.rs`, `argmax`.
-/

import Transformer.ALM.HullMark

namespace Transformer
namespace ALM

/-! ### The score is a squared distance -/

/-- **Completing the square.**  Against the query `(q, 1)` that `embed_query`
emits, a marked key scores `q² - (q - k)² + δ`.  The first term is the same for
every key in the head, so an argmax over the container is an argmin of
`(q - k)² - δ`. -/
theorem lineEval_markKey_eq (δ k q : ℝ) :
    lineEval (markKey δ k) q = q ^ 2 - (q - k) ^ 2 + δ := by
  unfold lineEval markKey
  ring

/-- **A squared-distance lead wider than the offset spread decides the
query.**  No integrality, no separation between keys: whatever the two offsets
are inside `[0, A]`, a key closer to the query by more than `A` in squared
distance beats the other one.  This is the whole comparison the head performs,
with the query's own term already cancelled. -/
theorem lineEval_markKey_lt_of_gap {A δ δ' q k k' : ℝ} (hδ0 : 0 ≤ δ) (hδ'A : δ' ≤ A)
    (hgap : A < (q - k') ^ 2 - (q - k) ^ 2) :
    lineEval (markKey δ' k') q < lineEval (markKey δ k) q := by
  rw [lineEval_markKey_eq, lineEval_markKey_eq]
  linarith

/-- The hypotheses are satisfiable, and with room to spare: the keys `0` and
`3` at the query `0`, whose squared-distance lead is `9`. -/
example : lineEval (markKey (0.4 : ℝ) 3) 0 < lineEval (markKey (0 : ℝ) 0) 0 :=
  lineEval_markKey_lt_of_gap (A := 0.4) le_rfl le_rfl (by norm_num)

/-! ### And for integer keys the lead comes from the query's position -/

/-- **The nearest integer key leads by more than the offset spread.**  Writing
`M = z' - z` and `u = q - z`, the lead is `M² - 2uM`, which is at least
`|M| - 2|u| ≥ 1 - 2|u|` — so a query within `(1 - A) / 2` of `z` puts every
other integer key more than `A` behind.  The unit separation of the keys is
what is being spent, exactly as in `not_eraseStep_of_marked`, but now at every
query rather than at the key's own abscissa.  `A < 1` is not a hypothesis
because it is already one: an offset spread of a unit or more leaves the window
`(1 - A) / 2` empty, and `hq` says a query is in it. -/
theorem sq_dist_gap_of_near_int {A q : ℝ} {z z' : ℤ} (hA0 : 0 ≤ A)
    (hne : z ≠ z') (hq : |q - (z : ℝ)| < (1 - A) / 2) :
    A < (q - (z' : ℝ)) ^ 2 - (q - (z : ℝ)) ^ 2 := by
  obtain ⟨h1, h2⟩ := abs_lt.mp hq
  have hM1 : (1 : ℝ) ≤ |(z' : ℝ) - (z : ℝ)| := by
    have hz : (1 : ℤ) ≤ |z' - z| := Int.one_le_abs (sub_ne_zero_of_ne (Ne.symm hne))
    have := (Int.cast_le (R := ℝ)).mpr hz
    rwa [Int.cast_abs, Int.cast_sub, Int.cast_one] at this
  have hrw : (q - (z' : ℝ)) ^ 2 - (q - (z : ℝ)) ^ 2
      = ((z' : ℝ) - (z : ℝ)) ^ 2 - 2 * (q - (z : ℝ)) * ((z' : ℝ) - (z : ℝ)) := by ring
  rw [hrw]
  rcases le_or_gt ((z' : ℝ) - (z : ℝ)) 0 with hle | hlt
  · rw [abs_of_nonpos hle] at hM1
    have hpos : (0 : ℝ) ≤ -((z' : ℝ) - (z : ℝ)) + 2 * (q - (z : ℝ)) := by linarith
    nlinarith [mul_le_mul_of_nonneg_right hM1 hpos]
  · rw [abs_of_pos hlt] at hM1
    have hpos : (0 : ℝ) ≤ ((z' : ℝ) - (z : ℝ)) - 2 * (q - (z : ℝ)) := by linarith
    nlinarith [mul_le_mul_of_nonneg_right hM1 hpos]

/-- The hypotheses are satisfiable at the shipped spread, and on both sides:
the query `0.2` against the keys `0` and `1`, and against `-1`. -/
example : (0.44 : ℝ) < (0.2 - ((1 : ℤ) : ℝ)) ^ 2 - (0.2 - ((0 : ℤ) : ℝ)) ^ 2 ∧
    (0.44 : ℝ) < (0.2 - ((-1 : ℤ) : ℝ)) ^ 2 - (0.2 - ((0 : ℤ) : ℝ)) ^ 2 := by
  constructor <;>
    · refine sq_dist_gap_of_near_int (by norm_num) (by decide) ?_
      rw [abs_lt]; norm_num

/-- **So the nearest key wins outright.**  A query inside the window of a key
the head holds is answered by that key, strictly, against every other integer
key whatever its offset. -/
theorem lineEval_markKey_lt_of_near {A δ δ' q : ℝ} {z z' : ℤ}
    (hδ0 : 0 ≤ δ) (hδ'0 : 0 ≤ δ') (hδ'A : δ' ≤ A) (hne : z ≠ z')
    (hq : |q - (z : ℝ)| < (1 - A) / 2) :
    lineEval (markKey δ' (z' : ℝ)) q < lineEval (markKey δ (z : ℝ)) q :=
  lineEval_markKey_lt_of_gap hδ0 hδ'A
    (sq_dist_gap_of_near_int (le_trans hδ'0 hδ'A) hne hq)

/-- **And "nearest" is literal.**  Under the same window the winner is the key
at least squared distance, so the ordered search a sorted key set supports —
predecessor and successor of the query — is searching for the right thing. -/
theorem abs_lt_abs_of_near {A q : ℝ} {z z' : ℤ} (hA0 : 0 ≤ A)
    (hne : z ≠ z') (hq : |q - (z : ℝ)| < (1 - A) / 2) :
    |q - (z : ℝ)| < |q - (z' : ℝ)| :=
  sq_lt_sq.mp (by linarith [sq_dist_gap_of_near_int hA0 hne hq])

/-! ### Over the whole container -/

/-- The offsets of a point that is in a marked family are the ones the family
bounds: the key determines the point, so there is nothing to choose. -/
theorem Marked.offset_mem {A : ℝ} {s : Finset (ℝ × ℝ)} (hs : Marked A s) {δ : ℝ} {z : ℤ}
    (hmem : markKey δ (z : ℝ) ∈ s) : 0 ≤ δ ∧ δ ≤ A := by
  obtain ⟨z₀, δ₀, h0, hA, heq⟩ := hs.perturbed _ hmem
  have hz : ((z₀ : ℤ) : ℝ) = ((z : ℤ) : ℝ) := by
    have := congrArg Prod.fst heq
    simp only [markKey_fst] at this
    linarith
  have hd := congrArg Prod.snd heq
  simp only [markKey, hz] at hd
  exact ⟨by linarith, by linarith⟩

/-- **The container's answer, over the whole container.**  On a live head — no
clear marker, integer keys, offsets inside `[0, A]` with `A < 1` — a query
within `(1 - A) / 2` of a key the head holds is answered by that key.  This is
the statement `hullProbe` and `argmax` need and that `HullMark` stopped short
of: not that the container holds every key, but that finding the nearest one is
finding the maximiser.  The hull, its breakpoints and its exact predicates are
not consulted anywhere in the proof — a sorted set of keys and a predecessor
search would answer identically. -/
theorem isGreatest_of_near {A : ℝ} {s : Finset (ℝ × ℝ)} (hs : Marked A s)
    {δ q : ℝ} {z : ℤ} (hmem : markKey δ (z : ℝ) ∈ s)
    (hq : |q - (z : ℝ)| < (1 - A) / 2) :
    ∀ l ∈ s, lineEval l q ≤ lineEval (markKey δ (z : ℝ)) q := by
  obtain ⟨hδ0, -⟩ := hs.offset_mem hmem
  intro l hl
  obtain ⟨z', δ', hδ'0, hδ'A, rfl⟩ := hs.perturbed l hl
  rcases eq_or_ne z' z with rfl | hne
  · rw [hs.onePerKey _ hl _ hmem (by rw [markKey_fst, markKey_fst])]
  · exact (lineEval_markKey_lt_of_near hδ0 hδ'0 hδ'A (Ne.symm hne) hq).le

/-- The hypotheses are satisfiable together: the two-key family of
`Transformer.ALM.HullMark`, queried at `0.2`, which is inside the window of
the key `0` at the shipped spread. -/
example : ∀ l ∈ ({markKey 0 ((0 : ℤ) : ℝ), markKey 0.4 ((1 : ℤ) : ℝ)} : Finset (ℝ × ℝ)),
    lineEval l 0.2 ≤ lineEval (markKey 0 ((0 : ℤ) : ℝ)) 0.2 := by
  refine isGreatest_of_near (A := 0.4) ⟨?_, ?_⟩ (by simp) (by rw [abs_lt]; norm_num)
  · intro l hl
    simp only [Finset.mem_insert, Finset.mem_singleton] at hl
    rcases hl with rfl | rfl
    exacts [⟨0, 0, le_rfl, by norm_num, rfl⟩, ⟨1, 0.4, by norm_num, le_rfl, rfl⟩]
  · intro l hl l' hl' hf
    simp only [Finset.mem_insert, Finset.mem_singleton] at hl hl'
    rcases hl with rfl | rfl <;> rcases hl' with rfl | rfl <;>
      first | rfl | (exfalso; simp only [markKey_fst] at hf; norm_num at hf)

/-! ### The window, and that it is not slack -/

/-- **The shipped window is `0.283`.**  `LATEST_ALPHA / log 2` is the whole
range of the recency offset (`marked_sep_of_shipped`), so `(1 - A) / 2` is how
far off a key a query may sit and still be answered by it.  `ALM.QueryScale`
and `alm-hull/src/query.rs` measure the machine's queries at `1.9e-6` off an
integer: five orders inside this. -/
theorem the_shipped_window : (0.283 : ℝ) < (1 - 0.3 / Real.log 2) / 2 := by
  have hlog : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have hpos : (0 : ℝ) < Real.log 2 := by linarith
  have : (0.3 : ℝ) / Real.log 2 < 0.4329 := by
    rw [div_lt_iff₀ hpos]; linarith
  linarith

/-- **And the window is not slack.**  At the midpoint of two keys the squared
distances are equal and the offsets decide, so the later write wins — which is
what `LATEST_ALPHA` was added to do.  `0.5` is outside the window of `0` at
`A = 0.4`, and there the nearest-key rule genuinely fails: the two keys are
equidistant and the key `1` wins on its offset alone. -/
theorem the_window_is_needed :
    ¬ |(0.5 : ℝ) - ((0 : ℤ) : ℝ)| < (1 - 0.4) / 2 ∧
      lineEval (markKey 0 ((0 : ℤ) : ℝ)) 0.5 < lineEval (markKey 0.4 ((1 : ℤ) : ℝ)) 0.5 := by
  constructor
  · rw [abs_lt]; push Not; intro _; norm_num
  · unfold lineEval markKey; norm_num

end ALM
end Transformer
