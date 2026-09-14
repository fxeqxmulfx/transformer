/-
# The cleared entry loses, and that is the only reason it is harmless

`_to_2d_key` subtracts `BIG = 1e30` from the key's intercept when a clear flag
is set (`transformer_vm/graph/core.py:314`), and the resulting point goes into
the same hull as every live key.  `todo3.md` §7 names this as the one place §0's
grid argument does not reach: `HARD_K` was off the integer grid by a scale, the
marker is off it by thirty orders of magnitude, so a cleared entry satisfies no
hypothesis of `Transformer.ALM.FloatGrid` — and the invariant that makes the
container correct anyway is nowhere stated or checked.

Here it is, and it is domination rather than arithmetic.  The query's second
coordinate is `1` (`_to_2d_query`), so the marker passes from the key
coordinate into the score unchanged (`dot_marked`); with every live score
inside `±M` and `2M < BIG` a marked entry is below every live one at every
query (`cleared_lt_live`), and the head answers exactly as if the cleared
entries had never been inserted (`marked_sup'_eq_live`).  That is the
unwritten invariant, in the form the erase loops need it.

What it costs is stated too.  `the_marker_costs_the_grid`: of two live scores a
unit apart — the separation `Transformer.ALM.ScoreWall` and the whole
tie-breaking argument run on — at most one survives the subtraction as a
binary64 number, because past `10^30` the representables are `2^47` apart.  So
the marker does not merely leave the grid, it destroys the unit that the grid
was there to preserve; only the fact that nothing ever reads a cleared score
back keeps that from mattering.

Source: `todo3.md` §7; `transformer_vm/graph/core.py` lines 8 and 293-319.
-/

import Transformer.ALM.ScoreWall
import Transformer.ALM.Duality

namespace Transformer
namespace ALM

/-! ### The marker passes into the score -/

/-- The score of an entry whose intercept carries `-c·B`: the flag `c` is `0`
for a live entry and `1` for a cleared one. -/
noncomputable def markedScore (B base c : ℝ) : ℝ := base - c * B

/-- **And it arrives there undiminished.**  `_to_2d_query` sets `qy = 1`, so
subtracting `BIG` from `ky` subtracts `BIG` from the score itself — there is no
scale in between to soften it. -/
theorem dot_marked (qx kx ky c B : ℝ) :
    dot (qx, 1) (kx, ky - c * B) = markedScore B (dot (qx, 1) (kx, ky)) c := by
  unfold dot markedScore
  simp only []
  ring

/-! ### So a cleared entry loses to every live one -/

/-- **The marked entry is below the live one.**  With both underlying scores
inside `±M` and the marker wider than the whole range, a cleared entry cannot
win, at any query and against any live competitor.  This is the invariant
`add_line` needs and `hull2d_cht.h` does not state. -/
theorem cleared_lt_live {B M bl bc : ℝ} (hl : |bl| ≤ M) (hc : |bc| ≤ M) (hB : 2 * M < B) :
    markedScore B bc 1 < markedScore B bl 0 := by
  have h1 := (abs_le.mp hl).1
  have h2 := (abs_le.mp hc).2
  unfold markedScore
  linarith

/-- **So the head answers as if the cleared entries were not there.**  Over any
container split into live entries `L` and cleared entries `C`, the highest
marked score is the highest live score: the whole clearing mechanism is a
domination argument, and none of `Transformer.ALM.FloatGrid` is used or
needed. -/
theorem marked_sup'_eq_live {ι : Type*} [DecidableEq ι] {L C : Finset ι} {base c : ι → ℝ}
    {B M : ℝ} (hL : L.Nonempty) (hlive : ∀ i ∈ L, c i = 0) (hclear : ∀ i ∈ C, c i = 1)
    (hbase : ∀ i ∈ L ∪ C, |base i| ≤ M) (hB : 2 * M < B) :
    (L ∪ C).sup' (hL.mono Finset.subset_union_left) (fun i => markedScore B (base i) (c i))
      = L.sup' hL (fun i => base i) := by
  have hLdup := hL
  obtain ⟨j, hj⟩ := hLdup
  have hjU : j ∈ L ∪ C := Finset.mem_union_left _ hj
  refine le_antisymm (Finset.sup'_le _ _ fun i hi => ?_) (Finset.sup'_le _ _ fun i hi => ?_)
  · rcases Finset.mem_union.mp hi with hiL | hiC
    · rw [hlive i hiL]
      simp only [markedScore, zero_mul, sub_zero]
      exact Finset.le_sup' (fun i => base i) hiL
    · rw [hclear i hiC]
      refine le_trans (cleared_lt_live (B := B) (hbase j hjU) (hbase i hi) hB).le ?_
      simp only [markedScore, zero_mul, sub_zero]
      exact Finset.le_sup' (fun i => base i) hj
  · have h := Finset.le_sup' (fun i => markedScore B (base i) (c i))
      (Finset.mem_union_left C hi)
    rw [hlive i hi] at h
    simp only [markedScore, zero_mul, sub_zero] at h
    exact h

/-! ### And what the subtraction costs -/

/-- Past the wall two consecutive integers cannot both be stored, on either
side of zero — `Transformer.ALM.ScoreWall.unit_gap_unstorable` read through
`|·|`, since the marker sends the score to where it is large and negative. -/
lemma unit_gap_unstorable_abs {p : ℕ} {z : ℤ} (hz : (2 : ℤ) ^ p < |z|)
    (hz1 : (2 : ℤ) ^ p < |z + 1|) :
    ¬ IsBinary p (z : ℝ) ∨ ¬ IsBinary p ((z + 1 : ℤ) : ℝ) := by
  rcases Int.even_or_odd z with he | ho
  · refine Or.inr (not_isBinary_odd ?_ hz1)
    rcases he with ⟨t, ht⟩
    exact ⟨t, by omega⟩
  · exact Or.inl (not_isBinary_odd ho hz)

/-- **The marker destroys the unit it was added to.**  Of two live scores one
apart, at most one survives the subtraction of `BIG` as a binary64 number: near
`10^30` the representables are `2^47` apart, and the separation the search and
the tie test both run on is gone.  Harmless only because a cleared score is
never compared against anything but a live one, which
`marked_sup'_eq_live` settles without arithmetic. -/
theorem the_marker_costs_the_grid {z : ℤ} (hz : |z| ≤ 2 ^ 53) :
    ¬ IsBinary 53 ((z - 10 ^ 30 : ℤ) : ℝ) ∨ ¬ IsBinary 53 ((z + 1 - 10 ^ 30 : ℤ) : ℝ) := by
  obtain ⟨h1, h2⟩ := abs_le.mp hz
  have hrw : z + 1 - 10 ^ 30 = (z - 10 ^ 30) + 1 := by ring
  rw [hrw]
  refine unit_gap_unstorable_abs ?_ ?_ <;>
    · rw [abs_of_nonpos (by norm_num at h2 ⊢; omega)]
      norm_num at h1 ⊢
      omega

/-! ### The hypotheses are satisfiable -/

/-- One live entry and one cleared entry, both scoring inside `±1`, with a
marker of `3`: the cleared one loses and the container answers `1`. -/
example :
    (∀ i ∈ ({0} : Finset ℕ), (fun i : ℕ => if i = 0 then (0 : ℝ) else 1) i = 0) ∧
    (∀ i ∈ ({1} : Finset ℕ), (fun i : ℕ => if i = 0 then (0 : ℝ) else 1) i = 1) ∧
    (∀ i ∈ ({0} : Finset ℕ) ∪ ({1} : Finset ℕ), |(fun _ : ℕ => (1 : ℝ)) i| ≤ 1) ∧
    2 * (1 : ℝ) < 3 := by
  refine ⟨?_, ?_, ?_, by norm_num⟩ <;> intro i hi <;> simp_all

/-- And a score the marker takes off the grid: `0`, the score of the key that
sits at the query. -/
example : |(0 : ℤ)| ≤ 2 ^ 53 := by norm_num

end ALM
end Transformer
