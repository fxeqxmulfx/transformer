/-
# Polynomials against `e^{2z}` for `lemma:beta1`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, Appendix A, proof of
`lemma:beta1`, which uses that a few functions `√z p(z) e^{-z}` are bounded,
"straightforward to verify".  `Perspective.SyncKernelSecond` reduces its bounds
on `f_β''` to four such facts, squared:

* `beta1_upper_tail`: `2z(2z - 3)² ≤ (4/25) e^{2z}` for `z ≥ 3/2`;
* `beta1_upper_middle`: `25(3z - 2)³ ≤ 27(3z - 1 - z²) z e^{2z}` on `[1, 11/5]`;
* `beta1_lower_start`: `2z(4 - 5z + z²)² ≤ (64/25) e^{2z}` on `[0, 2/5]`;
* `beta1_lower_middle`: `2z(3 - 2z)² ≤ (64/25) e^{2z}` on `[0, 3/2]`.

Each one replaces `e^{2z}` by a Taylor polynomial, shifted by `3` or `2` where
`z` is large, using `e > 2.7182818283` (`Real.exp_one_gt_d9`).  What is left is
a polynomial inequality that `nlinarith` proves.
-/

import Mathlib.Analysis.Complex.ExponentialBounds

namespace Transformer
namespace Perspective

/-- `e³ > 20`, from `e > 2.7182818283`. -/
theorem exp_three_gt_twenty : (20 : ℝ) < Real.exp 3 := by
  have h := Real.exp_one_gt_d9
  have h3 : Real.exp 3 = Real.exp 1 * Real.exp 1 * Real.exp 1 := by
    rw [← Real.exp_add, ← Real.exp_add]; norm_num
  rw [h3]
  nlinarith [mul_pos (Real.exp_pos 1) (Real.exp_pos 1)]

/-- `e² > 7.38`, from `e > 2.7182818283`. -/
theorem exp_two_gt_d2 : (369 / 50 : ℝ) < Real.exp 2 := by
  have h := Real.exp_one_gt_d9
  have h2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
    rw [← Real.exp_add]; norm_num
  rw [h2]
  nlinarith

/-- The Taylor polynomial of degree `5` bounds `exp` from below on `[0, ∞)`:
`Real.sum_le_exp_of_nonneg` written out. -/
theorem taylor_five_le_exp {y : ℝ} (hy : 0 ≤ y) :
    1 + y + y ^ 2 / 2 + y ^ 3 / 6 + y ^ 4 / 24 + y ^ 5 / 120 ≤ Real.exp y := by
  have h := Real.sum_le_exp_of_nonneg hy 6
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial] at h
  norm_num at h
  linarith

/-- `2z(2z - 3)² ≤ (4/25) e^{2z}` for `z ≥ 3/2`: with `s = 2z - 3`,
`(s + 3)s² ≤ (16/5)(1 + s + ⋯ + s⁵/120) ≤ (4/25) e³ e^s`.  The ratio peaks at
`0.88`, near `z = 2.7`. -/
theorem beta1_upper_tail {z : ℝ} (hz : 3 / 2 ≤ z) :
    2 * z * (2 * z - 3) ^ 2 ≤ 4 / 25 * Real.exp (2 * z) := by
  set s := 2 * z - 3 with hs
  have hs0 : 0 ≤ s := by linarith
  have hexp : Real.exp (2 * z) = Real.exp 3 * Real.exp s := by
    rw [← Real.exp_add]; congr 1; linarith
  have hT := taylor_five_le_exp hs0
  have hpoly : (s + 3) * s ^ 2 ≤
      16 / 5 * (1 + s + s ^ 2 / 2 + s ^ 3 / 6 + s ^ 4 / 24 + s ^ 5 / 120) := by
    nlinarith [sq_nonneg (s - 12 / 5), mul_nonneg hs0 (sq_nonneg (s - 12 / 5)),
      mul_nonneg (mul_nonneg hs0 hs0) (sq_nonneg (s - 12 / 5)), sq_nonneg (s ^ 2 - 6),
      mul_nonneg hs0 (sq_nonneg (s ^ 2 - 6)), mul_nonneg (pow_nonneg hs0 3) hs0]
  have h2z : 2 * z = s + 3 := by linarith
  rw [hexp, h2z]
  have hpos : 0 ≤ 1 + s + s ^ 2 / 2 + s ^ 3 / 6 + s ^ 4 / 24 + s ^ 5 / 120 := by positivity
  nlinarith [mul_le_mul exp_three_gt_twenty.le hT hpos (Real.exp_pos 3).le]

/-- `25(3z - 2)³ ≤ 27(3z - 1 - z²) z e^{2z}` for `1 ≤ z ≤ 11/5`, through
`e^{2z} = e² e^{2(z - 1)}`.  The ratio peaks at `0.66`, at `z = 11/5`. -/
theorem beta1_upper_middle {z : ℝ} (hz : 1 ≤ z) (hz' : z ≤ 11 / 5) :
    25 * (3 * z - 2) ^ 3 ≤ 27 * (3 * z - 1 - z ^ 2) * z * Real.exp (2 * z) := by
  set s := z - 1 with hs
  have hs0 : 0 ≤ s := by linarith
  have hs1 : s ≤ 6 / 5 := by linarith
  have hexp : Real.exp (2 * z) = Real.exp 2 * Real.exp (2 * s) := by
    rw [← Real.exp_add]; congr 1; linarith
  have hpoly : 25 * (1 + 3 * s) ^ 3 ≤ 27 * (1 + s - s ^ 2) * (1 + s) * (369 / 50) *
      (1 + 2 * s + (2 * s) ^ 2 / 2 + (2 * s) ^ 3 / 6) := by
    nlinarith [mul_nonneg hs0 (sub_nonneg.2 hs1),
      mul_nonneg (mul_nonneg hs0 hs0) (sub_nonneg.2 hs1),
      mul_nonneg (pow_nonneg hs0 3) (sub_nonneg.2 hs1), pow_nonneg hs0 3, pow_nonneg hs0 4,
      mul_nonneg (pow_nonneg hs0 4) (sub_nonneg.2 hs1)]
  have hz1 : z = 1 + s := by linarith
  have hr : 0 ≤ 27 * (1 + s - s ^ 2) * (1 + s) := by nlinarith
  have h2s : (0 : ℝ) ≤ 2 * s := by linarith
  have hT : 1 + 2 * s + (2 * s) ^ 2 / 2 + (2 * s) ^ 3 / 6 ≤ Real.exp (2 * s) := by
    nlinarith [taylor_five_le_exp h2s, pow_nonneg h2s 4, pow_nonneg h2s 5]
  have hprod : (369 / 50) * (1 + 2 * s + (2 * s) ^ 2 / 2 + (2 * s) ^ 3 / 6) ≤
      Real.exp 2 * Real.exp (2 * s) :=
    mul_le_mul exp_two_gt_d2.le hT (by positivity) (Real.exp_pos 2).le
  rw [hexp, hz1]
  nlinarith [mul_le_mul_of_nonneg_left hprod hr]

/-- `2z(4 - 5z + z²)² ≤ (64/25) e^{2z}` for `0 ≤ z ≤ 2/5`.  The ratio peaks at
`0.968`, near `z = 0.20`. -/
theorem beta1_lower_start {z : ℝ} (hz : 0 ≤ z) (hz' : z ≤ 2 / 5) :
    2 * z * (4 - 5 * z + z ^ 2) ^ 2 ≤ 64 / 25 * Real.exp (2 * z) := by
  have h2z : (0 : ℝ) ≤ 2 * z := by linarith
  have hpoly : 2 * z * (4 - 5 * z + z ^ 2) ^ 2 ≤
      64 / 25 * (1 + 2 * z + (2 * z) ^ 2 / 2 + (2 * z) ^ 3 / 6) := by
    nlinarith [sq_nonneg (z - 9 / 50), mul_nonneg hz (sq_nonneg (z - 9 / 50)),
      mul_nonneg (sub_nonneg.2 hz') (sq_nonneg (z - 9 / 50)), mul_nonneg hz (sub_nonneg.2 hz'),
      mul_nonneg (mul_nonneg hz (sub_nonneg.2 hz')) (sq_nonneg (z - 9 / 50))]
  nlinarith [taylor_five_le_exp h2z, pow_nonneg h2z 4, pow_nonneg h2z 5]

/-- `2z(3 - 2z)² ≤ (64/25) e^{2z}` for `0 ≤ z ≤ 3/2`.  The ratio peaks at
`0.744`, near `z = 0.28`. -/
theorem beta1_lower_middle {z : ℝ} (hz : 0 ≤ z) (hz' : z ≤ 3 / 2) :
    2 * z * (3 - 2 * z) ^ 2 ≤ 64 / 25 * Real.exp (2 * z) := by
  have h2z : (0 : ℝ) ≤ 2 * z := by linarith
  have hpoly : 2 * z * (3 - 2 * z) ^ 2 ≤ 64 / 25 * (1 + 2 * z + (2 * z) ^ 2 / 2) := by
    nlinarith [sq_nonneg (z - 1 / 4), mul_nonneg hz (sq_nonneg (z - 1 / 4)),
      mul_nonneg (sub_nonneg.2 hz') (sq_nonneg (z - 1 / 4)), mul_nonneg hz (sub_nonneg.2 hz')]
  nlinarith [taylor_five_le_exp h2z, pow_nonneg h2z 3, pow_nonneg h2z 4, pow_nonneg h2z 5]

/-- The hypotheses of the four bounds above are satisfiable. -/
example : 2 * 2 * (2 * 2 - 3 : ℝ) ^ 2 ≤ 4 / 25 * Real.exp (2 * 2) ∧
    25 * (3 * 2 - 2 : ℝ) ^ 3 ≤ 27 * (3 * 2 - 1 - 2 ^ 2) * 2 * Real.exp (2 * 2) ∧
    2 * (1 / 5) * (4 - 5 * (1 / 5) + (1 / 5) ^ 2 : ℝ) ^ 2 ≤ 64 / 25 * Real.exp (2 * (1 / 5)) ∧
    2 * 1 * (3 - 2 * 1 : ℝ) ^ 2 ≤ 64 / 25 * Real.exp (2 * 1) :=
  ⟨beta1_upper_tail (by norm_num), beta1_upper_middle (by norm_num) (by norm_num),
    beta1_lower_start (by norm_num) (by norm_num), beta1_lower_middle (by norm_num) (by norm_num)⟩

/-- `taylor_five_le_exp` applies at `y = 1`. -/
example : 1 + 1 + 1 ^ 2 / 2 + 1 ^ 3 / 6 + 1 ^ 4 / 24 + 1 ^ 5 / 120 ≤ Real.exp 1 :=
  taylor_five_le_exp zero_le_one

end Perspective
end Transformer
