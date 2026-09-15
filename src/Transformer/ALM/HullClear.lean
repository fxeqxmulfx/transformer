/-
# The marker erases the entry it marks

`Transformer.ALM.ClearKey` prices the other term `_to_2d_key` adds to the
intercept: `ky = ky - clear · BIG` with `BIG = 10^30`
(`transformer_vm/graph/core.py`, line 314, ported as
`vm-rs/alm-compile/src/graph.rs::embed_key`).  It proves the marker passes into
the score undiminished and that a cleared entry loses to every live one, so the
head answers as if the cleared entries had never been inserted
(`marked_sup'_eq_live`).

That is a statement about the answer.  This one is about the container, and it
goes the other way from `Transformer.ALM.HullMark`.  The recency term is small
enough that no erase rule fires; the marker is not, and `eraseStep_of_clearKey`
says so: a cleared key with a live key on either side of it is at or below the
higher of the two at *every* query, which is exactly the hypothesis of
`EraseStep`.  So the erase loops do run on a head that clears, and
`HullCover`'s count — one line per key inserted — is an upper bound there and
not an equality.  `not_marked_of_clearKey` is the same fact stated against the
predicate: a container holding a cleared line is not a `Marked` family for any
offset bound below the marker, so none of `Transformer.ALM.HullMark` applies to
it.

The bracket the erase needs is `(k - k')² ≤ BIG - δ`, and
`clear_bracket_of_shipped` discharges it for the shipped constants with
thirteen orders of magnitude to spare: the widest bracket two storable keys can
make is `(2·10^8)² = 4·10^16`, against a marker of `10^30`.  Which is the
same margin `cleared_lt_live` asks for, read as a geometric condition instead
of a bound on the score range.

Source: `transformer_vm/graph/core.py` lines 293-319; `todo3.md` §7.
-/

import Transformer.ALM.HullMark

namespace Transformer
namespace ALM

/-! ### The cleared key -/

/-- The 2D key of a cleared entry: the lift of `k`, raised by the recency term
`δ` and dropped by the marker `B`. -/
def clearKey (B δ k : ℝ) : ℝ × ℝ := (2 * k, -k ^ 2 + δ - B)

/-- The marker moves the intercept and nothing else, so a cleared entry still
occupies its key's slope. -/
theorem clearKey_fst (B δ k : ℝ) : (clearKey B δ k).1 = 2 * k := rfl

/-- With no marker it is the key of a live entry, so the two families differ in
`B` alone. -/
@[simp] theorem clearKey_zero (δ k : ℝ) : clearKey 0 δ k = markKey δ k := by
  unfold clearKey markKey; norm_num

/-! ### One live key covers it on one side -/

/-- **A live key is above a cleared one wherever the marker outweighs their
separation at the query.**  The difference of the two lines at `x` is
`(x - k')² - (x - k)²` against `B - δ + δ'`, so the marker buys coverage over
a range that widens with it. -/
theorem lineEval_clearKey_le {B δ δ' k k' x : ℝ} (hδ' : 0 ≤ δ')
    (h : (x - k') ^ 2 - (x - k) ^ 2 ≤ B - δ) :
    lineEval (clearKey B δ k) x ≤ lineEval (markKey δ' k') x := by
  unfold lineEval clearKey markKey
  nlinarith

/-- And on its own side of the cleared key the separation never exceeds the
gap between the two keys, however far the query runs. -/
theorem sq_sub_le_of_le {k k' x : ℝ} (hk : k' ≤ k) (hx : x ≤ k) :
    (x - k') ^ 2 - (x - k) ^ 2 ≤ (k - k') ^ 2 := by
  nlinarith

/-- The hypotheses are satisfiable: the live key `0` covers the cleared key `1`
at the query `0`, with a marker of `1`. -/
example : lineEval (clearKey 1 0 (1 : ℝ)) 0 ≤ lineEval (markKey 0 (0 : ℝ)) 0 :=
  lineEval_clearKey_le (by norm_num) (by norm_num)

/-! ### Two of them erase it -/

/-- **The marker erases the entry it marks.**  A cleared key with a live key on
either side is at or below the higher of the two at every query, so the erase
relation of `Transformer.ALM.HullPrune` applies to it — the loops of `add_line`
do run on a head that clears, and the container it leaves is smaller than the
set of lines inserted into it. -/
theorem eraseStep_of_clearKey {s : Finset (ℝ × ℝ)} {B δ δ₁ δ₂ k k₁ k₂ : ℝ}
    (hl : clearKey B δ k ∈ s)
    (h₁ : markKey δ₁ k₁ ∈ s.erase (clearKey B δ k))
    (h₂ : markKey δ₂ k₂ ∈ s.erase (clearKey B δ k))
    (hδ₁ : 0 ≤ δ₁) (hδ₂ : 0 ≤ δ₂) (hk₁ : k₁ ≤ k) (hk₂ : k ≤ k₂)
    (hB₁ : (k - k₁) ^ 2 ≤ B - δ) (hB₂ : (k₂ - k) ^ 2 ≤ B - δ) :
    EraseStep s (s.erase (clearKey B δ k)) := by
  refine ⟨clearKey B δ k, hl, rfl, fun x => ?_⟩
  rcases le_total x k with hx | hx
  · exact ⟨markKey δ₁ k₁, h₁,
      lineEval_clearKey_le hδ₁ (le_trans (sq_sub_le_of_le hk₁ hx) hB₁)⟩
  · refine ⟨markKey δ₂ k₂, h₂, lineEval_clearKey_le hδ₂ (le_trans ?_ hB₂)⟩
    have := sq_sub_le_of_le (k := -k) (k' := -k₂) (x := -x) (by linarith) (by linarith)
    nlinarith [this]

/-- The hypotheses are satisfiable, on the smallest family that has a bracket:
the cleared key `1` between the live keys `0` and `2`, with a marker of `1`. -/
example : EraseStep ({clearKey 1 0 (1 : ℝ), markKey 0 (0 : ℝ), markKey 0 (2 : ℝ)} : Finset (ℝ × ℝ))
    (({clearKey 1 0 (1 : ℝ), markKey 0 (0 : ℝ), markKey 0 (2 : ℝ)} : Finset (ℝ × ℝ)).erase
      (clearKey 1 0 (1 : ℝ))) := by
  refine eraseStep_of_clearKey (k₁ := 0) (k₂ := 2) (δ₁ := 0) (δ₂ := 0) (by simp) ?_ ?_
    (le_refl 0) (le_refl 0) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · simp only [Finset.mem_erase, Finset.mem_insert, Finset.mem_singleton]
    exact ⟨by simp [clearKey, markKey], by norm_num⟩
  · simp only [Finset.mem_erase, Finset.mem_insert, Finset.mem_singleton]
    exact ⟨by simp [clearKey, markKey], by norm_num⟩

/-! ### So `HullMark` does not describe a head that clears -/

/-- **A container holding a cleared line is not a `Marked` family.**  The
slope names the key, so the cleared line can only be the perturbed lift of its
own key, and its offset is `δ - B`, which is below zero as soon as the marker
exceeds the offset bound.  Every theorem of `Transformer.ALM.HullMark` is
therefore silent about such a container, and `eraseStep_of_clearKey` says why
it has to be. -/
theorem not_marked_of_clearKey {A B δ k : ℝ} {s : Finset (ℝ × ℝ)}
    (hmem : clearKey B δ k ∈ s) (hB : A < B - δ) : ¬ Marked A s := by
  intro hs
  obtain ⟨z, δ', hδ'0, hδ'A, heq⟩ := hs.perturbed _ hmem
  have hk : (z : ℝ) = k := by
    have := congrArg Prod.fst heq
    simp only [clearKey_fst, markKey_fst] at this
    linarith
  have hb := congrArg Prod.snd heq
  simp only [clearKey, markKey, hk] at hb
  linarith

/-- The hypotheses are satisfiable: the cleared key `1` with a marker of `1`
lies in a family no offset bound below `1` can describe. -/
example : ¬ Marked 0.4 ({clearKey 1 0 (1 : ℝ)} : Finset (ℝ × ℝ)) :=
  not_marked_of_clearKey (B := 1) (δ := 0) (k := 1) (by simp) (by norm_num)

/-! ### And the shipped marker clears the bracket -/

/-- **`BIG` is wider than any bracket two storable keys can make.**  A key past
`Transformer.ALM.ScoreWall`'s `2^26.5 < 10^8` is not stored exactly in the
first place, so the widest separation the erase test has to cover is
`(2·10^8)² = 4·10^16`, against `BIG = 10^30` — the same margin
`ClearKey.cleared_lt_live` asks of the score range, as a condition on the
keys. -/
theorem clear_bracket_of_shipped {δ k k' : ℝ} (hδ : δ ≤ 1)
    (hk : |k| ≤ 10 ^ 8) (hk' : |k'| ≤ 10 ^ 8) : (k - k') ^ 2 ≤ 10 ^ 30 - δ := by
  obtain ⟨hk1, hk2⟩ := abs_le.mp hk
  obtain ⟨hk'1, hk'2⟩ := abs_le.mp hk'
  nlinarith

/-- The hypotheses are satisfiable at the shipped offset bound and at a key the
machine really stores: `0.3/log 2 < 1`, and the wall itself. -/
example : (0.3 : ℝ) / Real.log 2 ≤ 1 ∧ |(94906266 : ℝ)| ≤ 10 ^ 8 :=
  ⟨le_of_lt marked_sep_of_shipped, by rw [abs_of_nonneg (by norm_num)]; norm_num⟩

end ALM
end Transformer
