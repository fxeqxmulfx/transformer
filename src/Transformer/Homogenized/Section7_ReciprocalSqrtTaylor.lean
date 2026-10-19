module

public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Tactic

/-!
# Explicit reciprocal-square-root remainders for the delta method

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the Taylor-Lagrange
expansion of `g(a,b,c) = a / sqrt(bc)`. Controlling its remainder
requires keeping the two squared query norms away from zero before
integrating. This module proves the deterministic scalar estimates
for that step, with explicit constants and an actual cubic error.

The quadratic polynomial is `1 - u/2 + 3u²/8`. Its difference from
`1/sqrt(1+u)` factors exactly after setting `t = sqrt(1+u)`.
On the larger domain `s ≥ 1/4`, the scalar error is at most
`4 |s-1|³`. For norm deviations of absolute value at most `1/2`,
both the reciprocal square root and its polynomial have absolute
value at most two. Multiplying the two estimates then gives an
explicit cubic bound for the product of the reciprocal square roots.

No probabilistic remainder, Gaussian moment bound, or exceptional-set
estimate is assumed. Those are separate steps when integrating this
pointwise expansion for the original Gaussian product matrix.
-/

@[expose] public section

open Real

namespace Transformer.Homogenized

/-- The reciprocal-square-root error has an exact cubic factor in
`sqrt(s)-1`. Source: arXiv:2604.01978v1, §7.3,
the Taylor-Lagrange step in `lem:delta_method`. -/
theorem reciprocal_sqrt_taylor_identity (s : ℝ) (hs : 0 < s) :
    8 * Real.sqrt s * ((Real.sqrt s)⁻¹ - (1 - (s - 1) / 2 + 3 * (s - 1) ^ 2 / 8)) =
      -(Real.sqrt s - 1) ^ 3 * (3 * s + 9 * Real.sqrt s + 8) := by
  let t := Real.sqrt s
  have ht : 0 < t := Real.sqrt_pos.mpr hs
  have hsq : t ^ 2 = s := Real.sq_sqrt hs.le
  change 8 * t * (t⁻¹ - (1 - (s - 1) / 2 + 3 * (s - 1) ^ 2 / 8)) =
    -(t - 1) ^ 3 * (3 * s + 9 * t + 8)
  rw [← hsq]
  field_simp
  ring

/-- An explicit cubic remainder on the actual domain separated
from the norm singularity. Source: arXiv:2604.01978v1, §7.3,
the remainder in the Taylor step of `lem:delta_method`. -/
theorem reciprocal_sqrt_taylor_bound (s : ℝ) (hs : 1 / 4 ≤ s) :
    |(Real.sqrt s)⁻¹ - (1 - (s - 1) / 2 + 3 * (s - 1) ^ 2 / 8)| ≤ 4 * |s - 1| ^ 3 := by
  let t := Real.sqrt s
  have hs0 : 0 < s := by linarith
  have ht : 0 < t := Real.sqrt_pos.mpr hs0
  have hsq : t ^ 2 = s := Real.sq_sqrt hs0.le
  have hhalf : 1 / 2 ≤ t := by nlinarith [Real.sqrt_nonneg s]
  have hp : 0 ≤ 3 * t ^ 2 + 9 * t + 8 := by positivity
  have hc : 3 * t ^ 2 + 9 * t + 8 ≤ 32 * t * (t + 1) ^ 3 := by
    have ha : (t + 1) ^ 2 ≤ (t + 1) ^ 3 := by
      nlinarith [mul_nonneg ht.le (sq_nonneg (t + 1))]
    have hb := mul_le_mul_of_nonneg_right hhalf (pow_nonneg (by positivity : 0 ≤ t + 1) 3)
    nlinarith [sq_nonneg t]
  have he := congrArg abs (reciprocal_sqrt_taylor_identity s hs0)
  change |8 * t * (t⁻¹ - (1 - (s - 1) / 2 + 3 * (s - 1) ^ 2 / 8))| =
    |-(t - 1) ^ 3 * (3 * s + 9 * t + 8)| at he
  rw [← hsq] at he
  simp only [abs_mul, abs_neg, abs_pow, abs_of_pos ht,
    abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 8), abs_of_nonneg hp] at he
  have hu : |s - 1| ^ 3 = |t - 1| ^ 3 * (t + 1) ^ 3 := by
    have hf : s - 1 = (t - 1) * (t + 1) := by nlinarith [hsq]
    rw [hf, abs_mul, abs_of_pos (by positivity : 0 < t + 1), mul_pow]
  apply (mul_le_mul_iff_right₀ (by positivity : (0 : ℝ) < 8 * t)).mp
  calc
    _ = |t - 1| ^ 3 * (3 * t ^ 2 + 9 * t + 8) := by simpa only [hsq] using he
    _ ≤ |t - 1| ^ 3 * (32 * t * (t + 1) ^ 3) :=
      mul_le_mul_of_nonneg_left hc (pow_nonneg (abs_nonneg _) 3)
    _ = 8 * t * (4 * |s - 1| ^ 3) := by rw [hu]; ring

/-- On the actual good norm event, the reciprocal square root is
bounded by two. Source: arXiv:2604.01978v1, §7.3,
localization of the singular function in `lem:delta_method`. -/
theorem reciprocal_sqrt_one_add_abs_le_two (u : ℝ) (hu : |u| ≤ 1 / 2) :
    |(Real.sqrt (1 + u))⁻¹| ≤ 2 := by
  obtain ⟨hlo, hhi⟩ := abs_le.mp hu
  have hs : 0 < 1 + u := by linarith
  have ht : 0 < Real.sqrt (1 + u) := Real.sqrt_pos.mpr hs
  have hhalf : 1 / 2 ≤ Real.sqrt (1 + u) := by
    nlinarith [Real.sq_sqrt hs.le, Real.sqrt_nonneg (1 + u)]
  rw [abs_of_pos (inv_pos.mpr ht)]
  have he := mul_inv_cancel₀ ht.ne'
  nlinarith [mul_nonneg (sub_nonneg.mpr hhalf) (inv_nonneg.mpr ht.le)]

/-- The actual quadratic polynomial is bounded on the same good
norm event. Source: arXiv:2604.01978v1, §7.3,
the second-order Taylor polynomial in `lem:delta_method`. -/
theorem reciprocal_sqrt_polynomial_abs_le_two (u : ℝ) (hu : |u| ≤ 1 / 2) :
    |1 - u / 2 + 3 * u ^ 2 / 8| ≤ 2 := by
  obtain ⟨hlo, hhi⟩ := abs_le.mp hu
  have hsq : u ^ 2 ≤ 1 / 4 := by
    nlinarith [mul_nonneg (by linarith : 0 ≤ 1 / 2 - u) (by linarith : 0 ≤ u + 1 / 2)]
  rw [abs_le]
  constructor <;> nlinarith [sq_nonneg u]

/-- Multiplying the actual two reciprocal square roots introduces
only a cubic error on the good norm event. Source:
arXiv:2604.01978v1, §7.3, `lem:delta_method`, denominator of `g`. -/
theorem reciprocal_sqrt_product_taylor_bound (u v : ℝ)
    (hu : |u| ≤ 1 / 2) (hv : |v| ≤ 1 / 2) :
    |(Real.sqrt (1 + u))⁻¹ * (Real.sqrt (1 + v))⁻¹ -
      (1 - u / 2 + 3 * u ^ 2 / 8) * (1 - v / 2 + 3 * v ^ 2 / 8)| ≤
        8 * (|u| ^ 3 + |v| ^ 3) := by
  have hbu : |(Real.sqrt (1 + u))⁻¹ - (1 - u / 2 + 3 * u ^ 2 / 8)| ≤ 4 * |u| ^ 3 := by
    simpa using reciprocal_sqrt_taylor_bound (1 + u) (by linarith [(abs_le.mp hu).1])
  have hbv : |(Real.sqrt (1 + v))⁻¹ - (1 - v / 2 + 3 * v ^ 2 / 8)| ≤ 4 * |v| ^ 3 := by
    simpa using reciprocal_sqrt_taylor_bound (1 + v) (by linarith [(abs_le.mp hv).1])
  have he : (Real.sqrt (1 + u))⁻¹ * (Real.sqrt (1 + v))⁻¹ -
      (1 - u / 2 + 3 * u ^ 2 / 8) * (1 - v / 2 + 3 * v ^ 2 / 8) =
        ((Real.sqrt (1 + u))⁻¹ - (1 - u / 2 + 3 * u ^ 2 / 8)) * (Real.sqrt (1 + v))⁻¹ +
          (1 - u / 2 + 3 * u ^ 2 / 8) *
            ((Real.sqrt (1 + v))⁻¹ - (1 - v / 2 + 3 * v ^ 2 / 8)) := by ring
  rw [he]
  calc
    _ ≤ |((Real.sqrt (1 + u))⁻¹ - (1 - u / 2 + 3 * u ^ 2 / 8)) * (Real.sqrt (1 + v))⁻¹| +
        |(1 - u / 2 + 3 * u ^ 2 / 8) *
          ((Real.sqrt (1 + v))⁻¹ - (1 - v / 2 + 3 * v ^ 2 / 8))| := abs_add_le _ _
    _ ≤ (4 * |u| ^ 3) * 2 + 2 * (4 * |v| ^ 3) := by
      simp only [abs_mul]
      exact add_le_add
        (mul_le_mul hbu (reciprocal_sqrt_one_add_abs_le_two v hv) (abs_nonneg _) (by positivity))
        (mul_le_mul (reciprocal_sqrt_polynomial_abs_le_two u hu) hbv (abs_nonneg _) (by norm_num))
    _ = _ := by ring

/-- A bounded original numerator preserves the explicit cubic
denominator error. Source: arXiv:2604.01978v1, §7.3,
the local Taylor expansion of the normalized correlation in `lem:delta_method`. -/
theorem reciprocal_sqrt_product_weighted_taylor_bound (a u v : ℝ)
    (ha : |a| ≤ 2) (hu : |u| ≤ 1 / 2) (hv : |v| ≤ 1 / 2) :
    |a * ((Real.sqrt (1 + u))⁻¹ * (Real.sqrt (1 + v))⁻¹) -
      a * ((1 - u / 2 + 3 * u ^ 2 / 8) * (1 - v / 2 + 3 * v ^ 2 / 8))| ≤
        16 * (|u| ^ 3 + |v| ^ 3) := by
  rw [← mul_sub, abs_mul]
  calc
    _ ≤ 2 * (8 * (|u| ^ 3 + |v| ^ 3)) :=
      mul_le_mul ha (reciprocal_sqrt_product_taylor_bound u v hu hv)
        (abs_nonneg _) (by norm_num)
    _ = _ := by ring

/-- Positive squared norms, the scalar localization and two
nonzero deviations satisfy all of the preceding hypotheses jointly. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 4 : ℝ) ≤ 1 / 2 ∧
    |(1 / 4 : ℝ)| ≤ 1 / 2 ∧ |(-1 / 4 : ℝ)| ≤ 1 / 2 ∧ |(3 / 2 : ℝ)| ≤ 2 := by norm_num

end Transformer.Homogenized
