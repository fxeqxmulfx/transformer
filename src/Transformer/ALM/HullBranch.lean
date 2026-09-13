/-
# The three branches of `query`, and why one of them needs no walk

`Transformer.ALM.HullLines` runs `lower_bound` over the lines the code stores.
`query` (`transformer_vm/attention/hull2d_cht.h`, lines 255-315) dispatches on
the sign of `qy` before it gets there, and `Transformer.ALM.Duality` reduces
the three cases one at a time without ever searching in them:

* `planar_bsearch_of_pos` — `qy > 0`: the upper hull, queried at `m = qx/qy`.
* `planar_bsearch_of_neg` — `qy < 0`: the code negates `(m, b)` and asks the
  *lower* hull, which is the same search run on the negated family.  This is
  the branch `isGreatest_dot_iff_of_neg` reduced and nothing used.
* `planar_argmax_of_snd_eq_zero`, `planar_argmax_of_snd_eq_zero_neg` —
  `qy == 0`: the degenerate branch the code answers at `±INF`, where the
  maximizer is an extreme slope and no search happens at all.

The degenerate branch also does something the other two do not: it calls
`it->meta.resolve(tb, out)` on a single line, with neither merge loop.  That
is sound only if nothing can tie there, and nothing said so.
`planar_argmax_unique_of_snd_eq_zero` and its mirror do: with `qy = 0` and
`qx ≠ 0` the score is a strictly monotone function of the slope, so the
extreme line beats every other line strictly and the tie set is a singleton.
The asymmetry in the code is a theorem about the geometry, not an oversight.

Source: `hull2d_cht.h`, lines 255-315.
-/

import Transformer.ALM.HullLines

namespace Transformer
namespace ALM

/-! ### The three branches of `query` -/

/-- **`qy > 0`.**  The upper hull, queried at `m = qx / qy`: the line the
binary search lands on is a planar argmax, at logarithmic cost.  Source:
`hull2d_cht.h`, lines 11-17 and 255-315. -/
theorem planar_bsearch_of_pos (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : 0 < q.2) (n : ℕ)
    (hslope : ∀ j, (L j).1 < (L (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (L a) (L (a + 1)) ≤ interX (L b) (L (b + 1))) :
    let p : ℕ → Bool := fun j => decide (q.1 / q.2 ≤ interX (L j) (L (j + 1)))
    (∀ j ≤ n, dot q (L j) ≤ dot q (L (bsearch p 0 n))) ∧ bcount n ≤ Nat.log 2 n + 1 := by
  intro p
  exact ⟨fun j hj => (dot_le_dot_iff_of_pos hq _ _).mpr
    ((bsearch_lines_isGreatest L (q.1 / q.2) n hslope hbp).1 j hj), bcount_le_log n⟩

/-- **`qy < 0`.**  Here the code pushes `(-m, -b)` into a second hull and
queries that one.  `N` is what the lower hull stores, so the keys of the
instance are `-N j`; the very same search over `N` returns a planar argmax of
the original keys, because negation reverses the order twice.  Source:
`hull2d_cht.h`, lines 11-17 and 255-315. -/
theorem planar_bsearch_of_neg (N : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : q.2 < 0) (n : ℕ)
    (hslope : ∀ j, (N j).1 < (N (j + 1)).1)
    (hbp : ∀ a b, a ≤ b → interX (N a) (N (a + 1)) ≤ interX (N b) (N (b + 1))) :
    let p : ℕ → Bool := fun j => decide (q.1 / q.2 ≤ interX (N j) (N (j + 1)))
    (∀ j ≤ n, dot q (-N j) ≤ dot q (-N (bsearch p 0 n))) ∧ bcount n ≤ Nat.log 2 n + 1 := by
  intro p
  refine ⟨fun j hj => ?_, bcount_le_log n⟩
  have hmax := (bsearch_lines_isGreatest N (q.1 / q.2) n hslope hbp).1 j hj
  rw [dot_le_dot_iff_of_neg hq, lineEval_neg, lineEval_neg, neg_le_neg_iff]
  exact hmax

/-- The mirror of `dot_le_dot_iff_of_snd_eq_zero`: with `qy = 0` and `qx < 0`
the planar argmax is the *smallest* slope. -/
theorem dot_le_dot_iff_of_snd_eq_zero_neg {q : ℝ × ℝ} (hq : q.2 = 0) (hq1 : q.1 < 0)
    (k k' : ℝ × ℝ) : dot q k ≤ dot q k' ↔ k'.1 ≤ k.1 := by
  rw [dot_of_snd_eq_zero q k hq, dot_of_snd_eq_zero q k' hq]
  exact mul_le_mul_left_of_neg hq1

/-- **`qy == 0`, `qx > 0`.**  The intercept drops out, so the argmax is the
last stored line — the code's query at `+INF`, answered with no search.
Source: `hull2d_cht.h`, lines 255-315. -/
theorem planar_argmax_of_snd_eq_zero (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : q.2 = 0)
    (hq1 : 0 < q.1) (n : ℕ) (hslope : ∀ j, (L j).1 < (L (j + 1)).1) :
    ∀ j ≤ n, dot q (L j) ≤ dot q (L n) := by
  intro j hj
  rw [dot_le_dot_iff_of_snd_eq_zero hq hq1]
  exact le_of_step_lt (n := n) (K := fun i => (L i).1) (fun i _ => hslope i) hj le_rfl

/-- **`qy == 0`, `qx < 0`.**  The same branch at `-INF`: the argmax is the
first stored line.  Source: `hull2d_cht.h`, lines 255-315. -/
theorem planar_argmax_of_snd_eq_zero_neg (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : q.2 = 0)
    (hq1 : q.1 < 0) (n : ℕ) (hslope : ∀ j, (L j).1 < (L (j + 1)).1) :
    ∀ j ≤ n, dot q (L j) ≤ dot q (L 0) := by
  intro j hj
  rw [dot_le_dot_iff_of_snd_eq_zero_neg hq hq1]
  exact le_of_step_lt (n := n) (K := fun i => (L i).1) (fun i _ => hslope i)
    (Nat.zero_le j) hj

/-! ### And in that branch nothing ties -/

/-- **`qy == 0`, `qx > 0`: the winner is alone.**  The score is `qx` times the
slope, and the slopes increase strictly, so every earlier line is strictly
worse.  This is why the branch resolves a single `meta` and runs neither merge
loop.  Source: `hull2d_cht.h`, lines 255-265. -/
theorem planar_argmax_unique_of_snd_eq_zero (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ) (hq : q.2 = 0)
    (hq1 : 0 < q.1) (n : ℕ) (hslope : ∀ j, (L j).1 < (L (j + 1)).1) :
    ∀ j < n, dot q (L j) < dot q (L n) := by
  intro j hj
  rw [dot_of_snd_eq_zero q (L j) hq, dot_of_snd_eq_zero q (L n) hq]
  refine mul_lt_mul_of_pos_left ?_ hq1
  exact lt_of_lt_of_le (hslope j)
    (le_of_step_lt (n := n) (K := fun i => (L i).1) (fun i _ => hslope i) hj le_rfl)

/-- **`qy == 0`, `qx < 0`: the same, at the other end.**  Source:
`hull2d_cht.h`, lines 255-265. -/
theorem planar_argmax_unique_of_snd_eq_zero_neg (L : ℕ → ℝ × ℝ) (q : ℝ × ℝ)
    (hq : q.2 = 0) (hq1 : q.1 < 0) (n : ℕ) (hslope : ∀ j, (L j).1 < (L (j + 1)).1) :
    ∀ j, 0 < j → j ≤ n → dot q (L j) < dot q (L 0) := by
  intro j hj0 _
  rw [dot_of_snd_eq_zero q (L j) hq, dot_of_snd_eq_zero q (L 0) hq]
  refine mul_lt_mul_of_neg_left ?_ hq1
  exact lt_of_le_of_lt
    (le_of_step_lt (n := j - 1) (K := fun i => (L i).1) (fun i _ => hslope i)
      (Nat.zero_le _) le_rfl)
    (by simpa [Nat.sub_add_cancel hj0] using hslope (j - 1))

/-! ### The hypotheses are satisfiable -/


/-- Satisfiable for a query with positive second coordinate. -/
example : (0 : ℝ) < ((5, 2) : ℝ × ℝ).2 ∧
    (∀ j, (parabLine j).1 < (parabLine (j + 1)).1) ∧
      ∀ a b, a ≤ b → interX (parabLine a) (parabLine (a + 1))
        ≤ interX (parabLine b) (parabLine (b + 1)) :=
  ⟨by norm_num, parabLine_slope, parabLine_bp⟩

/-- And for one with negative second coordinate, the lower-hull branch. -/
example : ((5, -2) : ℝ × ℝ).2 < 0 ∧
    (∀ j, (parabLine j).1 < (parabLine (j + 1)).1) ∧
      ∀ a b, a ≤ b → interX (parabLine a) (parabLine (a + 1))
        ≤ interX (parabLine b) (parabLine (b + 1)) :=
  ⟨by norm_num, parabLine_slope, parabLine_bp⟩

/-- The degenerate branch is reachable in both directions, and the two sides
really disagree: at `q = (1, 0)` the last line wins, at `q = (-1, 0)` the
first. -/
example : ((1, 0) : ℝ × ℝ).2 = 0 ∧ (0 : ℝ) < ((1, 0) : ℝ × ℝ).1 ∧
    ((-1, 0) : ℝ × ℝ).2 = 0 ∧ ((-1, 0) : ℝ × ℝ).1 < 0 ∧
    dot (1, 0) (parabLine 0) < dot (1, 0) (parabLine 1) ∧
    dot (-1, 0) (parabLine 1) < dot (-1, 0) (parabLine 0) := by
  refine ⟨rfl, by norm_num, rfl, by norm_num, ?_, ?_⟩ <;>
    simp only [dot, parabLine, liftKey] <;> norm_num

end ALM
end Transformer
