/-
# The breakpoint test, both ways it is answered

`alm-hull/src/exact.rs::cross_sign` decides `a * b - c * d` against zero, and
it is not a rescan: every breakpoint comparison in the hull goes through it, so
unlike `dot_cmp` it is on the path the shipped machine takes.  It answers
twice over.

The fast path subtracts the two rounded products and trusts the result when it
is far enough from zero to carry its own error -- `|det| > 2·eps·(|p| + |q|)`,
four operations.  `cross_filter_sound` is that test: with the error model of
`Transformer.ALM.DotError.dot_error_le` the computed difference is within
`u(2 + u)` of the exact one relative to the terms, the terms are within
`1/(1 - u)` of the rounded products the runtime actually measures against, and
`2·eps = 4u` clears the product of the two for every `u ≤ 2/5`.  Binary64's
`u = 2^-53` clears it by a factor of nearly two.

The slow path is the expansion, and `cross_sign_eq` is it: the four terms
`two_prod` produces sum to `a * b - c * d` exactly, so
`Transformer.ALM.ExpansionSign.expansion_sign_eq` gives the true sign.  It runs
where the filter gives up, which is where the determinant nearly cancels.

Both answer the same question, which is `cross_paths_agree`: the docstring's
"the filter decides who computes it, not what it is" is the conjunction of the
two, and not a separate claim.

Source: `vm-rs/alm-hull/src/exact.rs`, `CROSS_FILTER` and `cross_sign`;
Shewchuk 1997, §3; Higham, §3.1.
-/

import Transformer.ALM.ExactDot
import Transformer.ALM.DotError

namespace Transformer
namespace ALM

/-! ### A difference that carries its own error -/

/-- **An error smaller than the value has the value's sign.**  The one step the
filter is: if the computed `det` is further from zero than it is from the exact
`e`, then `e` is on the same side of zero as `det` -- and `det = 0` is
impossible, since nothing is strictly closer to `det` than `det`. -/
theorem sign_of_error_lt {det e : ℝ} (h : |det - e| < |det|) :
    (0 < det ↔ 0 < e) ∧ (det < 0 ↔ e < 0) := by
  obtain ⟨hlo, hhi⟩ := abs_lt.mp h
  rcases lt_trichotomy det 0 with hd | hd | hd
  · rw [abs_of_neg hd] at hlo hhi
    constructor <;> constructor <;> intro hc <;> linarith
  · rw [hd] at h
    simp at h
    exact absurd h (abs_nonneg _).not_gt
  · rw [abs_of_pos hd] at hlo hhi
    constructor <;> constructor <;> intro hc <;> linarith

/-- **The fast path is sound.**  `p` and `q` are the rounded products, `det`
their rounded difference, and the runtime compares `|det|` against
`4u(|p| + |q|)` -- its `CROSS_FILTER` is `2 * eps` and `eps = 2u`.  When the
comparison passes, the sign of `det` is the sign of `a * b - c * d`.

The margin is where `u ≤ 2/5` comes from: the exact terms exceed the measured
ones by at most `1/(1 - u)`, so the bound to clear is `u(2 + u)/(1 - u)`, and
`4u` clears it exactly when `5u ≤ 2`.

Source: `exact.rs::CROSS_FILTER`; Higham, §3.1. -/
theorem cross_filter_sound {u a b c d p q det : ℝ} (hu : 0 ≤ u) (hu' : 5 * u ≤ 2)
    (hp : |p - a * b| ≤ u * |a * b|) (hq : |q - c * d| ≤ u * |c * d|)
    (hdet : |det - (p - q)| ≤ u * |p - q|)
    (hfilter : 4 * u * (|p| + |q|) < |det|) :
    (0 < det ↔ 0 < a * b - c * d) ∧ (det < 0 ↔ a * b - c * d < 0) := by
  -- The two-term model, with the second term entering negated.
  have hq' : |(-q) - (-(c * d))| ≤ u * |(-(c * d))| := by
    have hneg : (-q) - (-(c * d)) = -(q - c * d) := by ring
    rw [hneg, abs_neg, abs_neg]
    exact hq
  have hs : |det - (p + -q)| ≤ u * |p + -q| := by simpa [sub_eq_add_neg] using hdet
  have herr := dot_error_le (u := u) (a := a * b) (b := -(c * d)) hu hp hq' hs
  rw [dotTerms, abs_neg] at herr
  have herr' : |det - (a * b - c * d)| ≤ u * (2 + u) * (|a * b| + |c * d|) := by
    have hrw : det - (a * b - c * d) = det - (a * b + -(c * d)) := by ring
    rw [hrw]
    exact herr
  -- The terms the bound is relative to, against the products the test measures.
  have hpa : (1 - u) * |a * b| ≤ |p| := by
    have h := abs_sub_abs_le_abs_sub (a * b) p
    rw [abs_sub_comm] at h
    nlinarith
  have hqc : (1 - u) * |c * d| ≤ |q| := by
    have h := abs_sub_abs_le_abs_sub (c * d) q
    rw [abs_sub_comm] at h
    nlinarith
  have hT : (1 - u) * (|a * b| + |c * d|) ≤ |p| + |q| := by nlinarith
  have hscale : 4 * u * ((1 - u) * (|a * b| + |c * d|)) ≤ 4 * u * (|p| + |q|) :=
    mul_le_mul_of_nonneg_left hT (by linarith)
  -- `u(2 + u) ≤ 4u(1 - u)` is `5u ≤ 2`, on terms that are nonnegative.
  have hterms : (0 : ℝ) ≤ |a * b| + |c * d| := by positivity
  have hcoef : u * (2 + u) * (|a * b| + |c * d|)
      ≤ 4 * u * ((1 - u) * (|a * b| + |c * d|)) := by
    nlinarith [mul_nonneg (mul_nonneg hu (by linarith : (0 : ℝ) ≤ 2 - 5 * u)) hterms]
  exact sign_of_error_lt (by linarith)

/-- The hypotheses are satisfiable with the rounding doing real work: at
`u = 1/8` the exact determinant is `1`, the product rounds up to `17/16`, and
the filter still passes, since `4u` of the measured magnitude is `17/32`. -/
example : (0 : ℝ) < 17 / 16 ↔ (0 : ℝ) < 1 * 1 - 0 * 0 :=
  (cross_filter_sound (u := 1 / 8) (a := 1) (b := 1) (c := 0) (d := 0)
    (p := 17 / 16) (q := 0) (det := 17 / 16) (by norm_num) (by norm_num)
    (by norm_num [abs_of_nonneg]) (by norm_num) (by norm_num)
    (by norm_num [abs_of_nonneg])).1

/-! ### The path the filter hands over to -/

/-- **The four terms of the expansion branch.**  Two products, each split by
`two_prod`, the second negated -- in the order `cross_sign` pushes them.

Source: `exact.rs::cross_sign`. -/
def crossProdTerms (P : ExactProd) (a b c d : ℝ) : List ℝ :=
  [P.hi a b, P.lo a b, -P.hi c d, -P.lo c d]

/-- They sum to the determinant, exactly. -/
theorem crossProdTerms_sum (P : ExactProd) (a b c d : ℝ) :
    (crossProdTerms P a b c d).sum = a * b - c * d := by
  have h1 := P.exact a b
  have h2 := P.exact c d
  simp only [crossProdTerms, List.sum_cons, List.sum_nil]
  linarith

/-- **The slow path is exact.**  Same composition as `dot_cmp_eq`, on four
terms instead of eight: the expansion preserves their sum and its last
component carries the sign, so what `cross_sign` returns on the branch the
filter could not take is the sign of `a * b - c * d` itself.

Source: `exact.rs::cross_sign`; Shewchuk 1997, §2. -/
theorem cross_sign_eq (F : ExactSum) (P : ExactProd) {a b c d : ℝ} {lows : List ℝ} {top : ℝ}
    (hexp : expansion F (crossProdTerms P a b c d) = lows ++ [top])
    (hdom : |lows.sum| < |top|) :
    (0 < a * b - c * d ↔ 0 < top) ∧ (a * b - c * d < 0 ↔ top < 0) := by
  have h := expansion_sign_eq F hexp hdom
  rwa [crossProdTerms_sum] at h

/-- The hypotheses hold together: `2 * 1 - 1 * 1` leaves the single component
`1`, which dominates the empty tail below it. -/
example : (0 : ℝ) < 2 * 1 - 1 * 1 ↔ (0 : ℝ) < 1 :=
  (cross_sign_eq exactAdd exactMul (a := 2) (b := 1) (c := 1) (d := 1)
    (lows := []) (top := 1)
    (by norm_num [expansion, push, grow, exactAdd, exactMul, crossProdTerms])
    (by norm_num)).1

/-- **The filter decides who computes the sign, not what it is.**  Both
branches of `cross_sign` report the sign of `a * b - c * d`, so on any input
where both are available they report the same thing -- which is what makes the
fast path an optimization rather than an approximation. -/
theorem cross_paths_agree {u a b c d p q det : ℝ} (hu : 0 ≤ u) (hu' : 5 * u ≤ 2)
    (hp : |p - a * b| ≤ u * |a * b|) (hq : |q - c * d| ≤ u * |c * d|)
    (hdet : |det - (p - q)| ≤ u * |p - q|)
    (hfilter : 4 * u * (|p| + |q|) < |det|)
    (F : ExactSum) (P : ExactProd) {lows : List ℝ} {top : ℝ}
    (hexp : expansion F (crossProdTerms P a b c d) = lows ++ [top])
    (hdom : |lows.sum| < |top|) :
    (0 < det ↔ 0 < top) ∧ (det < 0 ↔ top < 0) := by
  obtain ⟨hf1, hf2⟩ := cross_filter_sound hu hu' hp hq hdet hfilter
  obtain ⟨he1, he2⟩ := cross_sign_eq F P hexp hdom
  exact ⟨hf1.trans he1, hf2.trans he2⟩

/-- Both sets of hypotheses at once, on one input: the rounded `det = 17/16`
passes the filter, the expansion of the exact terms leaves `1` on top, and the
two agree about the sign because both are the sign of `1 * 1 - 0 * 0`. -/
example : (0 : ℝ) < 17 / 16 ↔ (0 : ℝ) < 1 :=
  (cross_paths_agree (u := 1 / 8) (a := 1) (b := 1) (c := 0) (d := 0)
    (p := 17 / 16) (q := 0) (det := 17 / 16) (by norm_num) (by norm_num)
    (by norm_num [abs_of_nonneg]) (by norm_num) (by norm_num)
    (by norm_num [abs_of_nonneg]) exactAdd exactMul (lows := []) (top := 1)
    (by norm_num [expansion, push, grow, exactAdd, exactMul, crossProdTerms])
    (by norm_num)).1

end ALM
end Transformer
