/-
# The scale the compiler puts on a query, and the division that takes it off

`todo3.md` §0: the compiler multiplies every hard-attention query by
`HARD_K · √2 = 1.41 · 10^10`, a softmax temperature left on a path that takes an
argmax.  `vm-rs/alm-model/src/cache.rs::on_the_grid` divides it back out before
the query reaches a head, and its docstring gives the reason in one line —
`argmax_k ⟨q, k⟩` is invariant under `q ↦ q / s` — and then relies on it
everywhere, because every float statement in this development is written at
`qy = 1`.

Nothing here had said that line.  This file says it, and says what the two ends
of the division are.

`dot_scaleQuery_int` is the first end: at the compiler's query the head's score
is `σ · sScore q k`, which is the exact expression `Transformer.ALM.GuardSep`
and `Transformer.ALM.ScoreGap` are stated over, so their `σ` is the model's
scale and not a convenience.  `onTheGrid_scaleQuery` is the other: dividing by
`|qy|` returns the lifted query `(q, 1)` on the nose, and
`onTheGrid_score_isInt` is what that buys — the score is an integer again,
which is the grid hypothesis `ALM.FloatGrid.fp_exact_of_grid` needs and the
only thing the scale ever took away.

Between them, `order_scale_invariant` and `maximizers_scale_invariant`: a
positive rescaling of the query changes no comparison and no set of maximizers,
so the division is free.  Free in exact arithmetic — the division itself
rounds, and `rounded_normalization_keeps_the_winner` is the price, which
`Transformer.ALM.DriftMargin` had already bounded and which is paid in the
margin rather than in exactness.

Source: `todo3.md` §0 and §8; `vm-rs/alm-model/src/cache.rs`, `on_the_grid`.
-/

import Transformer.ALM.DriftMargin
import Transformer.ALM.Duality

namespace Transformer
namespace ALM

/-! ### The query the compiler actually emits -/

/-- The lifted query with the hard-attention scale still on it: `(σq, σ)`,
which is what `graph/core.py` hands the head and what `gap.rs`'s own test
builds as `[7s, s]`. -/
def scaleQuery (σ q : ℝ) : ℝ × ℝ := (σ * q, σ)

@[simp] theorem scaleQuery_one (q : ℝ) : scaleQuery 1 q = liftQuery q := by
  unfold scaleQuery liftQuery
  norm_num

/-- **The scaled score is the scaled `sScore`.**  So the `σ · sScore q k` that
`Transformer.ALM.GuardSep` and `Transformer.ALM.ScoreGap` state their
separation over is the score the head computes at the compiler's own query,
not a normalization chosen to make the algebra work. -/
theorem dot_scaleQuery_int (σ : ℝ) (q k : ℤ) :
    dot (scaleQuery σ (q : ℝ)) (liftKey (k : ℝ)) = σ * sScore q k := by
  unfold dot scaleQuery liftKey sScore
  ring

/-! ### A positive rescaling decides nothing -/

/-- Scaling the query scales every score by the same factor. -/
theorem dot_smul_query (c : ℝ) (q k : ℝ × ℝ) :
    dot (c * q.1, c * q.2) k = c * dot q k := by
  unfold dot
  ring

/-- **And so changes no comparison.**  This is `on_the_grid`'s one-line
justification, which the rest of the float development takes for granted every
time it writes a query as `(q, 1)`. -/
theorem order_scale_invariant {c : ℝ} (hc : 0 < c) (q k k' : ℝ × ℝ) :
    dot (c * q.1, c * q.2) k ≤ dot (c * q.1, c * q.2) k' ↔ dot q k ≤ dot q k' := by
  rw [dot_smul_query, dot_smul_query]
  exact ⟨fun h => le_of_mul_le_mul_left h hc, fun h => mul_le_mul_of_nonneg_left h hc.le⟩

open scoped Classical in
/-- **Nor any set of maximizers.**  Not merely the winner: the whole tie set the
merge walk collects is the same before and after the division, so the payload
the head returns is too. -/
theorem maximizers_scale_invariant {c : ℝ} (hc : 0 < c) (q : ℝ × ℝ)
    (S : Finset (ℝ × ℝ)) :
    (S.filter fun k => ∀ k' ∈ S, dot (c * q.1, c * q.2) k' ≤ dot (c * q.1, c * q.2) k)
      = S.filter fun k => ∀ k' ∈ S, dot q k' ≤ dot q k := by
  ext k
  simp only [Finset.mem_filter, and_congr_right_iff]
  exact fun _ => forall_congr' fun k' =>
    forall_congr' fun _ => order_scale_invariant hc q k' k

/-! ### What the division returns -/

/-- `on_the_grid`: divide both coordinates by `|qy|`, and leave the query alone
when that is zero — a zero second coordinate is the degenerate branch, where
every key ties and the tie-break answers. -/
noncomputable def onTheGrid (q : ℝ × ℝ) : ℝ × ℝ :=
  if q.2 = 0 then q else (q.1 / |q.2|, q.2 / |q.2|)

/-- **The division inverts the scale exactly.**  Not approximately: for a
positive scale the normalized query is the lifted `(q, 1)` itself, which is why
every statement at `qy = 1` is a statement about the shipped model. -/
theorem onTheGrid_scaleQuery {σ q : ℝ} (hσ : 0 < σ) :
    onTheGrid (scaleQuery σ q) = liftQuery q := by
  have hne : σ ≠ 0 := ne_of_gt hσ
  unfold onTheGrid scaleQuery liftQuery
  rw [if_neg hne]
  simp only [abs_of_pos hσ, Prod.mk.injEq]
  refine ⟨by field_simp, by field_simp⟩

/-- A normalized query is a positive rescaling of the original, so it compares
keys the same way. -/
theorem onTheGrid_preserves_order {q : ℝ × ℝ} (hq : q.2 ≠ 0) (k k' : ℝ × ℝ) :
    dot (onTheGrid q) k ≤ dot (onTheGrid q) k' ↔ dot q k ≤ dot q k' := by
  have hpos : 0 < 1 / |q.2| := by positivity
  have hrw : onTheGrid q = ((1 / |q.2|) * q.1, (1 / |q.2|) * q.2) := by
    unfold onTheGrid
    rw [if_neg hq]
    rw [one_div, inv_mul_eq_div, inv_mul_eq_div]
  rw [hrw]
  exact order_scale_invariant hpos q k k'

/-- **And that is the whole of what the scale had cost.**  Back at unit scale
the score of an integer key at an integer query is an integer again, which is
the grid hypothesis `ALM.FloatGrid.fp_exact_of_grid` asks for and the one thing
`HARD_K · √2` was destroying. -/
theorem onTheGrid_score_isInt {σ : ℝ} (hσ : 0 < σ) (q k : ℤ) :
    ∃ z : ℤ, dot (onTheGrid (scaleQuery σ (q : ℝ))) (liftKey (k : ℝ)) = (z : ℝ) := by
  refine ⟨2 * k * q - k ^ 2, ?_⟩
  rw [onTheGrid_scaleQuery hσ, dot_lift_int]
  unfold sScore
  push_cast
  ring

/-! ### And what the division itself costs -/

/-- The lifted score at a real query is `qScore`, the drifted score
`Transformer.ALM.DriftMargin` is stated over. -/
theorem dot_liftQuery_eq_qScore (r : ℝ) (k : ℤ) :
    dot (liftQuery r) (liftKey (k : ℝ)) = qScore r k := by
  unfold dot liftQuery liftKey qScore
  ring

/-- **The division rounds, and the margin absorbs it.**  `on_the_grid` is the
only rounding `cache.rs` introduces, so the query that reaches the head is
within some `ε` of the integer one rather than equal to it.  While the keys
stay inside `4Kε < 1` the winner is unchanged — the cost is paid out of the
unit gap between distinct keys and not out of exactness, which is
`Transformer.ALM.DriftMargin`'s whole point, here attached to the division that
causes it. -/
theorem rounded_normalization_keeps_the_winner {r : ℝ} {q₀ b : ℤ} {K ε : ℝ}
    {S : Finset ℤ} (hr : |r - (q₀ : ℝ)| ≤ ε) (hb : b ∈ S)
    (hK : ∀ k ∈ S, |(k : ℝ)| ≤ K) (hmargin : 4 * K * ε < 1)
    (hwin : ∀ k ∈ S, k ≠ b → sScore q₀ k < sScore q₀ b) :
    ∀ k ∈ S, k ≠ b →
      dot (liftQuery r) (liftKey (k : ℝ)) < dot (liftQuery r) (liftKey (b : ℝ)) := by
  intro k hk hne
  rw [dot_liftQuery_eq_qScore, dot_liftQuery_eq_qScore]
  exact winner_survives_drift hr hb hK hmargin hwin k hk hne

/-! ### The hypotheses are satisfiable -/

/-- The shipped scale, on `gap.rs`'s own query: `σ = √2 · 10^10` at `q = 7`
scores `σ · sScore 7 k`, and the division returns `(7, 1)`.  These are the
hypotheses of `dot_scaleQuery_int`, `onTheGrid_scaleQuery`,
`onTheGrid_preserves_order` and `onTheGrid_score_isInt`. -/
example : (0 : ℝ) < 14142135623.730951 ∧
    onTheGrid (scaleQuery 14142135623.730951 (7 : ℝ)) = liftQuery 7 ∧
    dot (scaleQuery 14142135623.730951 ((7 : ℤ) : ℝ)) (liftKey ((7 : ℤ) : ℝ))
      = 14142135623.730951 * sScore 7 7 := by
  refine ⟨by norm_num, onTheGrid_scaleQuery (by norm_num), dot_scaleQuery_int _ 7 7⟩

/-- And the rounded division that still lands on the right key: the query `0`
over the keys `{0, 1}`, where `1` wins and a drift of `10⁻⁹` against keys
bounded by `100` is well inside the margin.  These are the hypotheses of
`rounded_normalization_keeps_the_winner`. -/
example : |(0 : ℝ) - ((1 : ℤ) : ℝ)| ≤ 1 ∧ (1 : ℤ) ∈ ({0, 1} : Finset ℤ) ∧
    (∀ k ∈ ({0, 1} : Finset ℤ), |(k : ℝ)| ≤ 100) ∧ 4 * 100 * (1e-9 : ℝ) < 1 := by
  refine ⟨by norm_num, by decide, fun k hk => ?_, by norm_num⟩
  fin_cases hk <;> norm_num

end ALM
end Transformer
