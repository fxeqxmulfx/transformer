/-
# Causal attention — The elementary estimates behind the interaction window

Five inequalities with no `h` or `g` in them, pulled out of the proof of
`lem:interaction` so that it reads as the Gaussian argument it is:

* `exp_neg_le_four_div_sq`  — `e^{-y} ≤ 4/y²`, the polynomial decay the
  large-`b` estimate needs;
* `exp_neg_27_32_le_half`   — `e^{-27/32} ≤ 1/2`, its small-`b` counterpart;
* `lin_lower_of_small`, `const_lower_of_small` — the loss from the cubic term
  of the sine and from a Gaussian factor near the origin;
* `quartic_exponent_le`     — how much of the quadratic exponent the quartic
  correction is allowed to eat.

Source: arXiv:2411.04990v2, §B, the arithmetic of `lem:interaction`.
-/

import Transformer.Basic

open Real

namespace Transformer
namespace Causal

/-- `e^{-y} ≤ 4/y²` for `y > 0`: already the quadratic term of
`e^{y} = (e^{y/2})² ≥ (1 + y/2)²` dominates. -/
theorem exp_neg_le_four_div_sq {y : ℝ} (hy : 0 < y) : Real.exp (-y) ≤ 4 / y ^ 2 := by
  have hhalf := Real.add_one_le_exp (y / 2)
  have hsplit : Real.exp y = Real.exp (y / 2) * Real.exp (y / 2) := by
    rw [← Real.exp_add]; ring_nf
  have hq : y ^ 2 / 4 ≤ Real.exp y := by
    rw [hsplit]; nlinarith [Real.exp_pos (y / 2)]
  have hpos : 0 < Real.exp (-y) := Real.exp_pos _
  have hprod : Real.exp (-y) * Real.exp y = 1 := by rw [← Real.exp_add]; simp
  rw [le_div_iff₀ (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_left hq hpos.le]

/-- The hypothesis is satisfiable: `y = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- `e^{-27/32} ≤ 1/2`, the numeric heart of the first interaction
inequality: `e^{27/32} = (e^{27/64})² ≥ (91/64)² > 2`. -/
theorem exp_neg_27_32_le_half : Real.exp (-(27 / 32 : ℝ)) ≤ 1 / 2 := by
  have hsplit : Real.exp (27 / 32 : ℝ) = Real.exp (27 / 64) * Real.exp (27 / 64) := by
    rw [← Real.exp_add]; norm_num
  have h2 : (2 : ℝ) ≤ Real.exp (27 / 32) := by
    have h := Real.add_one_le_exp (27 / 64 : ℝ)
    rw [hsplit]; nlinarith [Real.exp_pos (27 / 64 : ℝ)]
  have hprod : Real.exp (-(27 / 32 : ℝ)) * Real.exp (27 / 32) = 1 := by
    rw [← Real.exp_add]; norm_num
  nlinarith [Real.exp_pos (-(27 / 32 : ℝ))]

/-- The elementary estimate behind `h(εβ^{-1/2}) > 0.99 εβ^{-1/2}`: for
`0 < u` with `u² ≤ 1/1000`, the cubic correction of the sine and a Gaussian
factor `e ≥ 199/200` together cost less than one per cent. -/
theorem lin_lower_of_small {u e : ℝ} (hu : 0 < u) (hu2 : u ^ 2 ≤ 1 / 1000)
    (he : 199 / 200 ≤ e) : 99 / 100 * u ≤ e * (u - u ^ 3 / 6) := by
  have hcube : u * u ^ 2 ≤ u * (1 / 1000) := mul_le_mul_of_nonneg_left hu2 hu.le
  have hpos3 : (0 : ℝ) ≤ u - u ^ 3 / 6 := by nlinarith
  calc 99 / 100 * u ≤ 199 / 200 * (u - u ^ 3 / 6) := by nlinarith
    _ ≤ e * (u - u ^ 3 / 6) := mul_le_mul_of_nonneg_right he hpos3

/-- The hypotheses of `lin_lower_of_small` are satisfiable: `u = 1/100`,
`e = 1`. -/
example : (0 : ℝ) < 1 / 100 ∧ ((1 : ℝ) / 100) ^ 2 ≤ 1 / 1000 ∧ (199 : ℝ) / 200 ≤ 1 := by
  norm_num

/-- The elementary estimate behind `g(εβ^{-1/2}) > 0.9`: the two quadratic
corrections and a Gaussian factor `e ≥ 199/200` together cost less than a
tenth. -/
theorem const_lower_of_small {v w e : ℝ} (hv2 : v ≤ 1 / 1000)
    (hw2 : w ≤ 1 / 100) (he : 199 / 200 ≤ e) :
    9 / 10 ≤ e * (1 - v / 2 - w) := by
  have hinner : (0 : ℝ) ≤ 1 - v / 2 - w := by linarith
  calc (9 : ℝ) / 10 ≤ 199 / 200 * (1 - v / 2 - w) := by linarith
    _ ≤ e * (1 - v / 2 - w) := mul_le_mul_of_nonneg_right he hinner

/-- The hypotheses of `const_lower_of_small` are satisfiable: `v = w = 0`,
`e = 1`. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ (0 : ℝ) ≤ 1 / 100 ∧ (199 : ℝ) / 200 ≤ 1 := by
  norm_num

/-- The exponent of the Gaussian upper bound at `(b+1)β^{-1/2}`: writing
`q = (b+1)²` and `v = q β^{-1}`, the quartic correction `qv/24` eats at most
`242/1944` of the quadratic term `q/2`, leaving `365/972 q`. -/
theorem quartic_exponent_le {q v : ℝ} (hq : 0 ≤ q) (hv : v ≤ 242 / 81) :
    -(q / 2) + q * v / 24 ≤ -(365 / 972 * q) := by
  nlinarith

/-- The hypotheses of `quartic_exponent_le` are satisfiable: `q = v = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 242 / 81 := by norm_num

end Causal
end Transformer
