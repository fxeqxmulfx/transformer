/-
# The eight terms the arbiter is handed

`Transformer.ALM.Expansion` and `Transformer.ALM.ExpansionSign` take a list of
terms to the sign of its exact sum.  This file says which list, and what that
sign then means.

`alm-hull/src/exact.rs::dot_cmp` splits each of the four products with
`two_prod` -- an FMA and a subtraction, so `hi + lo = a * b` with no error at
all -- and hands `expansion_sign` the eight pieces, the second key's four
negated.  `crossTerms` is that list and `crossTerms_sum` is the only thing
about it that matters: the pieces sum to `⟪q, a⟫ - ⟪q, b⟫`, exactly, however
the splitting was done.  `dot_cmp_eq` composes it with `expansion_sign_eq`.

The composition is the contract `head.rs` leans on.  When
`Transformer.ALM.DotError.cmp_of_dot_guard` cannot license a comparison --
which on `fibonacci` happens on 61 of 382284 queries -- the head rescans with
`dot_cmp`, and what comes back has to be the exact order rather than a better
estimate of it, or the rescan decides nothing.  It is the exact order, down to
the one step that is not algebra: that the expansion is non-overlapping, which
`ExpansionSign` takes as a hypothesis and Shewchuk proves about binary
floating point.

The exactness of `two_prod` enters the same way, as the field `ExactProd.exact`
rather than as a proof: it is Dekker's theorem, and with an FMA it is one
instruction and a rounding-error identity, neither of which this repository's
real-valued model of floating point can state.

Source: `vm-rs/alm-hull/src/exact.rs`, `two_prod` and `dot_cmp`; Dekker 1971,
§5; Shewchuk 1997, §2; Ogita-Rump-Oishi 2005, §3.
-/

import Transformer.ALM.ExpansionSign
import Transformer.ALM.Duality

namespace Transformer
namespace ALM

/-! ### Splitting a product -/

/-- **What `two_prod` returns.**  A rounded product and the error it dropped,
which together are the product exactly.  With an FMA that is one instruction:
`p = fl(a * b)` and `e = fma(a, b, -p)`, and `e` is representable.

Only the identity is assumed here; which pair of floats realizes it is
Dekker's theorem and a statement about representability, not about ℝ.

Source: Dekker 1971, §5; `exact.rs::two_prod`. -/
structure ExactProd where
  /-- The rounded product. -/
  hi : ℝ → ℝ → ℝ
  /-- The error it dropped. -/
  lo : ℝ → ℝ → ℝ
  /-- Together they are the product, with nothing lost. -/
  exact : ∀ a b, hi a b + lo a b = a * b

/-- Real multiplication splits trivially: it drops nothing.  This is not the
runtime's `two_prod`, it is the witness that the contract is satisfiable. -/
def exactMul : ExactProd where
  hi a b := a * b
  lo _ _ := 0
  exact _ _ := by ring

/-! ### The list `dot_cmp` builds -/

/-- **The eight terms of `dot_cmp`.**  Four products, each split in two, the
loser's four negated -- in the order the runtime pushes them.

Source: `exact.rs::dot_cmp`. -/
def crossTerms (P : ExactProd) (q a b : ℝ × ℝ) : List ℝ :=
  [P.hi q.1 a.1, P.lo q.1 a.1, P.hi q.2 a.2, P.lo q.2 a.2,
    -P.hi q.1 b.1, -P.lo q.1 b.1, -P.hi q.2 b.2, -P.lo q.2 b.2]

/-- **They sum to the difference of the scores, exactly.**  Nothing about the
splitting is used beyond `ExactProd.exact`, which is the point: the arbiter is
handed the comparison it is meant to settle, not an approximation of it. -/
theorem crossTerms_sum (P : ExactProd) (q a b : ℝ × ℝ) :
    (crossTerms P q a b).sum = dot q a - dot q b := by
  have h1 := P.exact q.1 a.1
  have h2 := P.exact q.2 a.2
  have h3 := P.exact q.1 b.1
  have h4 := P.exact q.2 b.2
  simp only [crossTerms, dot, List.sum_cons, List.sum_nil]
  linarith

/-- The same difference read on the line the hull actually queries, through
`Transformer.ALM.dot_eq_mul_lineEval`: the arbiter settles the planar
comparison, and the envelope's order is that one rescaled by `q.2`. -/
theorem crossTerms_sum_lineEval (P : ExactProd) {q : ℝ × ℝ} (hq : q.2 ≠ 0) (a b : ℝ × ℝ) :
    (crossTerms P q a b).sum
      = q.2 * (lineEval a (q.1 / q.2) - lineEval b (q.1 / q.2)) := by
  rw [crossTerms_sum, dot_eq_mul_lineEval q a hq, dot_eq_mul_lineEval q b hq]
  ring

/-- The hypothesis is satisfiable: the queries the hull is asked have a
nonzero second coordinate, which is what makes the reduction available. -/
example : (crossTerms exactMul ((3 : ℝ), (1 : ℝ)) (2, 0) (1, 0)).sum
    = (1 : ℝ) * (lineEval (2, 0) (3 / 1) - lineEval (1, 0) (3 / 1)) :=
  crossTerms_sum_lineEval exactMul (q := (3, 1)) one_ne_zero (2, 0) (1, 0)

/-! ### Which is the arbiter -/

/-- **`dot_cmp` returns the exact comparison.**  The eight terms sum to the
difference of the scores (`crossTerms_sum`), the expansion preserves that sum
(`expansion_sum`), and its last component carries the sign (`sign_of_top`), so
the single comparison at the end of `expansion_sign` answers about `⟪q, a⟫`
against `⟪q, b⟫` and about nothing else.

This is what `head.rs` needs from the rescan: not a sharper bound, a verdict.

Source: `exact.rs::dot_cmp`; Shewchuk 1997, §2. -/
theorem dot_cmp_eq (F : ExactSum) (P : ExactProd) {q a b : ℝ × ℝ} {lows : List ℝ} {top : ℝ}
    (hexp : expansion F (crossTerms P q a b) = lows ++ [top]) (hdom : |lows.sum| < |top|) :
    (dot q b < dot q a ↔ 0 < top) ∧ (dot q a < dot q b ↔ top < 0) := by
  obtain ⟨hpos, hneg⟩ := expansion_sign_eq F hexp hdom
  rw [crossTerms_sum] at hpos hneg
  constructor
  · rw [← hpos]; constructor <;> intro h <;> linarith
  · rw [← hneg]; constructor <;> intro h <;> linarith

/-- The hypotheses hold together, and the verdict they give is the true one:
`q = (1, 0)` scores `a = (2, 0)` above `b = (1, 0)`, the expansion is the
single component `1`, and it dominates the empty tail below it. -/
example : dot (1, 0) (1, 0) < dot (1, 0) (2, 0) ↔ (0 : ℝ) < 1 :=
  (dot_cmp_eq exactAdd exactMul (q := (1, 0)) (a := (2, 0)) (b := (1, 0))
    (lows := []) (top := 1)
    (by norm_num [expansion, push, grow, exactAdd, exactMul, crossTerms]) (by norm_num)).1

/-- **An empty expansion is a tie, and the runtime's `Equal` is right.**  No
components left means the eight terms cancelled exactly, which for these eight
terms means the two keys score the same. -/
theorem dot_cmp_tie (F : ExactSum) (P : ExactProd) {q a b : ℝ × ℝ}
    (hexp : expansion F (crossTerms P q a b) = []) : dot q a = dot q b := by
  have := expansion_nil_sum F hexp
  rw [crossTerms_sum] at this
  linarith

/-- The hypothesis is satisfiable, on the tie it is meant to report: a key
compared against itself. -/
example : dot (1, 1) (2, 3) = dot (1, 1) (2, 3) :=
  dot_cmp_tie exactAdd exactMul (q := (1, 1)) (a := (2, 3)) (b := (2, 3))
    (by norm_num [expansion, push, grow, exactAdd, exactMul, crossTerms])

end ALM
end Transformer
