module

public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Tactic

/-!
# Explicit two-coordinate quadratic bound

Source: `review/convex/moments.md`, CM108. The scalar formula is proved before
applying it to the actual complete complex column Gram matrix.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound

/-- CM108: the explicit larger quadratic energy value. -/
noncomputable def top (a d q : ℝ) : ℝ :=
  (a + d + Real.sqrt ((a - d) ^ 2 + 4 * q ^ 2)) / 2

/-- CM108: the full real discriminant is nonnegative, including zero. -/
theorem discriminant_nonneg (a d q : ℝ) : 0 ≤ (a - d) ^ 2 + 4 * q ^ 2 := by
  nlinarith [sq_nonneg (a - d), sq_nonneg q]

/-- CM108: the explicit bound dominates the first diagonal coefficient. -/
theorem top_ge_left (a d q : ℝ) : a ≤ top a d q := by
  have h : a - d ≤ Real.sqrt ((a - d) ^ 2 + 4 * q ^ 2) :=
    Real.le_sqrt_of_sq_le (by nlinarith [sq_nonneg q])
  unfold top
  linarith

/-- CM108: the explicit bound dominates the second diagonal coefficient. -/
theorem top_ge_right (a d q : ℝ) : d ≤ top a d q := by
  have h : d - a ≤ Real.sqrt ((a - d) ^ 2 + 4 * q ^ 2) :=
    Real.le_sqrt_of_sq_le (by nlinarith [sq_nonneg q])
  unfold top
  linarith

/-- CM108: nonnegative first Gram diagonal gives a nonnegative energy bound. -/
theorem top_nonneg (a d q : ℝ) (ha : 0 ≤ a) : 0 ≤ top a d q :=
  ha.trans (top_ge_left a d q)

/-- CM108: the two actual diagonal gaps have the exact cross-energy product. -/
theorem gap_product (a d q : ℝ) :
    (top a d q - a) * (top a d q - d) = q ^ 2 := by
  have h := Real.sq_sqrt (discriminant_nonneg a d q)
  unfold top
  nlinarith

/-- CM108: a nonnegative two-coordinate quadratic form controls its full cross term. -/
theorem weighted_cross (p r q x y : ℝ) (hp : 0 ≤ p) (hr : 0 ≤ r)
    (hqr : q ^ 2 = p * r) : 2 * q * x * y ≤ p * x ^ 2 + r * y ^ 2 := by
  rcases eq_or_lt_of_le hp with hp | hp
  · have hpz : p = 0 := hp.symm
    have hqz : q = 0 := by rw [hpz] at hqr; nlinarith
    simpa only [hpz, hqz, mul_zero, zero_mul, zero_add] using
      mul_nonneg hr (sq_nonneg y)
  · by_contra h
    have hn := mul_neg_of_pos_of_neg hp (sub_neg.mpr (lt_of_not_ge h))
    have he := congrArg (fun t : ℝ => t * y ^ 2) hqr
    nlinarith [sq_nonneg (p * x - q * y)]

example : 0 ≤ top 1 1 1 := top_nonneg 1 1 1 (by norm_num)

example (x y : ℝ) : 2 * (1 : ℝ) * x * y ≤ 1 * x ^ 2 + 1 * y ^ 2 :=
  weighted_cross 1 1 1 x y (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Hilbert.MatrixBound
