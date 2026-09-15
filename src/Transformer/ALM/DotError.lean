/-
# Three roundings, not one, and the cancellation between them

`Transformer.ALM.GuardSep.cmp_of_guard` takes the runtime's guard to the
comparison it licenses, and it models the price of the arithmetic as
`hδ₁ hδ₂ : δ ≤ ulpOf p E / 2` — each score within half a spacing of exact,
which is what one correctly rounded operation costs.

The machine does not spend one operation.  `alm-hull/src/head.rs` evaluates
`q0 * k0 + q1 * k1`: two products and a sum, three roundings, and the bound
they add up to is relative to the two products and not to their sum.  On the
keys the compiler emits that distinction is the whole story.  A marked key is
`(2k, -k² + δ)` (`Transformer.ALM.HullMark.markKey`), so a query near its own
abscissa forms `2k²` and `-k²` and answers `k²`: the terms are three times the
result, and their rounding is carried into it undiminished.  Measuring the
error as `ulp(score)` understates it by exactly that factor.

`dot_error_le` is the bound — `u(2 + u)(|a| + |b|)`, the standard two-term
model — and `cmp_of_dot_guard` is `Transformer.ALM.FloatHull.cmp_of_sep`
discharged from it, so the runtime's check is `2·u(2 + u)·T < |a - b|` against
the true gap rather than `ulp(score) < |qy|` against an assumed one.
`dotTerms_markKey_self` is the factor of three.

This is what `alm-hull/src/gap.rs::score_error_bound` computes and what
`ScoreGaps::unresolved` counts the failures of.  Read this way the shipped
model leaves queries uncovered where the `ulp` reading said it did not: on
`fibonacci` the worst ratio is `3.86` and 61 of 382284 queries have a gap that
does not clear the rounding at all.  Uncovered is not wrong -- the guard is
sufficient and not necessary, as the examples below show -- and rescanning
those 61 with `alm-hull/src/exact.rs`'s exact dot product upholds every one of
them.  What the theorem does not reach, the runtime still gets right.

Source: `todo3.md` §4a; `vm-rs/alm-hull/src/head.rs`, `BruteAttentionHead::query`;
Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1.
-/

import Transformer.ALM.FloatHull
import Transformer.ALM.HullMark

namespace Transformer
namespace ALM

/-! ### The size the rounding is relative to -/

/-- The two products of a two-term dot product, in absolute value.  This is
the quantity a relative error bound on `a + b` is relative to, and it is not
`|a + b|`: the sum can cancel and the products cannot. -/
def dotTerms (a b : ℝ) : ℝ := |a| + |b|

/-- `dotTerms` is a bound on the exact sum, and in general a strict one. -/
theorem abs_add_le_dotTerms (a b : ℝ) : |a + b| ≤ dotTerms a b := abs_add_le a b

/-- **Two products and a sum.**  With every operation correctly rounded to a
relative `u`, the computed `s` sits within `u(2 + u)` of the exact `a + b`
*relative to the terms*: one `u` for the two products, one for the sum, and
the `u²` because the sum is rounded after the products already grew.

Source: Higham, §3.1, applied to `q0 * k0 + q1 * k1`. -/
theorem dot_error_le {u a b p q s : ℝ} (hu : 0 ≤ u)
    (hp : |p - a| ≤ u * |a|) (hq : |q - b| ≤ u * |b|)
    (hs : |s - (p + q)| ≤ u * |p + q|) :
    |s - (a + b)| ≤ u * (2 + u) * dotTerms a b := by
  have hpa : |p| ≤ |a| + u * |a| := by linarith [abs_sub_abs_le_abs_sub p a]
  have hqb : |q| ≤ |b| + u * |b| := by linarith [abs_sub_abs_le_abs_sub q b]
  have hsum : |p + q| ≤ (1 + u) * (|a| + |b|) := by
    have hexp : (1 + u) * (|a| + |b|) = (|a| + u * |a|) + (|b| + u * |b|) := by ring
    rw [hexp]
    linarith [abs_add_le p q]
  have hgrow : u * |p + q| ≤ u * ((1 + u) * (|a| + |b|)) :=
    mul_le_mul_of_nonneg_left hsum hu
  have hsplit : |s - (a + b)| ≤ |s - (p + q)| + |(p - a) + (q - b)| := by
    have hrw : s - (a + b) = (s - (p + q)) + ((p - a) + (q - b)) := by ring
    rw [hrw]
    exact abs_add_le _ _
  have hterms : |(p - a) + (q - b)| ≤ u * |a| + u * |b| := by
    have := abs_add_le (p - a) (q - b)
    linarith
  have hfin : u * ((1 + u) * (|a| + |b|)) + (u * |a| + u * |b|)
      = u * (2 + u) * (|a| + |b|) := by ring
  unfold dotTerms
  linarith

/-- The hypotheses are satisfiable, and by the exact arithmetic that rounds
nothing: `u = 0` makes all three bounds `0 ≤ 0`. -/
example : |(3 : ℝ) - (1 + 2)| ≤ 0 * (2 + 0) * dotTerms 1 2 :=
  dot_error_le (u := 0) (a := 1) (b := 2) (p := 1) (q := 2) (s := 3)
    le_rfl (by norm_num) (by norm_num) (by norm_num)

/-! ### Which is the hypothesis the comparison needs -/

/-- **A comparison of two dot products, decided correctly.**  Both scores
carry the error of `dot_error_le` against a common bound `T` on their terms,
so the two errors together are `2·u(2 + u)·T`; a gap wider than that decides
the comparison exactly.  This is the guard `alm-hull/src/gap.rs` reports:
`2 · score_error_bound(terms) / (best - second) < 1`.

`Transformer.ALM.GuardSep.cmp_of_guard` is the same conclusion from the
shipped check, and the two differ in both arguments — that one reads the
error as `ulpOf p E / 2` and the separation as the assumed `σ`. -/
theorem cmp_of_dot_guard {u a b a' b' T : ℝ}
    (ha : |a' - a| ≤ u * (2 + u) * T) (hb : |b' - b| ≤ u * (2 + u) * T)
    (hguard : 2 * (u * (2 + u) * T) < |a - b|) :
    a' ≤ b' ↔ a ≤ b :=
  cmp_of_sep ha hb (by linarith)

/-- The hypotheses are satisfiable at a rounding the machine actually does:
`u = 2⁻⁵³`, terms of `3`, and two scores a whole unit apart. -/
example : ∀ a' b' : ℝ,
    |a' - 1| ≤ 1 / 2 ^ (53 : ℕ) * (2 + 1 / 2 ^ (53 : ℕ)) * 3 →
    |b' - 2| ≤ 1 / 2 ^ (53 : ℕ) * (2 + 1 / 2 ^ (53 : ℕ)) * 3 →
    (a' ≤ b' ↔ (1 : ℝ) ≤ 2) :=
  fun _ _ ha hb =>
    cmp_of_dot_guard ha hb (by rw [show |(1 : ℝ) - 2| = 1 by norm_num]; norm_num)

/-- **And the hypothesis cannot be dropped.**  At `u = 1/4` and terms of `1`
the bound is `9/16`, and two errors of exactly that size carry `a = 0` and
`b = 1` past each other: the computed comparison is the reverse of the exact
one.  So a gap that fails to clear twice the bound really is undecided. -/
example : |(9 / 16 : ℝ) - 0| ≤ 1 / 4 * (2 + 1 / 4) * 1 ∧
    |(7 / 16 : ℝ) - 1| ≤ 1 / 4 * (2 + 1 / 4) * 1 ∧
    ¬ ((9 / 16 : ℝ) ≤ 7 / 16 ↔ (0 : ℝ) ≤ 1) := by
  norm_num

/-- **Undecided is not wrong.**  The same bound, the same gap, and errors that
the bound permits but does not force: the comparison comes out right anyway.
The guard is sufficient and not necessary, which is why
`alm-hull/src/head.rs` answers an unresolved query by rescanning with an exact
dot product rather than by calling it a failure -- and why all 61 unresolved
queries on `fibonacci` are upheld. -/
example : |(1 / 10 : ℝ) - 0| ≤ 1 / 4 * (2 + 1 / 4) * 1 ∧
    |(9 / 10 : ℝ) - 1| ≤ 1 / 4 * (2 + 1 / 4) * 1 ∧
    ¬ 2 * ((1 : ℝ) / 4 * (2 + 1 / 4) * 1) < |(0 : ℝ) - 1| ∧
    ((1 / 10 : ℝ) ≤ 9 / 10 ↔ (0 : ℝ) ≤ 1) := by
  norm_num

/-! ### And why the factor is three and not one -/

/-- **The cancellation, on the keys the compiler emits.**  A marked key
queried at its own abscissa forms the products `2k·k` and `-k² + δ` and
answers `k² + δ`.  At `δ = 0` the terms are `3k²` and the answer is `k²`: the
error the dot product carries is three times what the spacing at the answer
suggests, and `ulp(score)` cannot see the difference.

Source: `alm-hull/src/head.rs`, `BruteAttentionHead::query`. -/
theorem dotTerms_markKey_self (k : ℝ) :
    dotTerms (2 * k * k) (-k ^ 2) = 3 * |lineEval (markKey 0 k) k| := by
  have hval : lineEval (markKey 0 k) k = k ^ 2 := by
    unfold lineEval markKey; ring
  have h2 : |2 * k * k| = 2 * k ^ 2 := by
    rw [show 2 * k * k = 2 * k ^ 2 by ring, abs_of_nonneg (by positivity)]
  have h1 : |(-k ^ 2 : ℝ)| = k ^ 2 := by
    rw [abs_neg, abs_of_nonneg (by positivity)]
  rw [hval, abs_of_nonneg (by positivity : (0 : ℝ) ≤ k ^ 2)]
  unfold dotTerms
  rw [h2, h1]
  ring

/-- And so the bound really is three times the `ulp`-sized one: at `k = 2`
the score is `4` and the terms are `12`. -/
example : dotTerms (2 * 2 * 2) (-(2 : ℝ) ^ 2) = 12 := by
  rw [dotTerms_markKey_self 2, show lineEval (markKey 0 (2 : ℝ)) 2 = 4 by
    unfold lineEval markKey; norm_num]
  norm_num

end ALM
end Transformer
