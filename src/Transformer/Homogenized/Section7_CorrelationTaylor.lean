/-
# The actual quadratic correlation expansion and its cubic remainder

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the Taylor-Lagrange
step for `g(a,b,c) = a / sqrt(bc)` at `(r,1,1)`.
Here `w,u,v` are the actual deviations of its three arguments.
The quadratic polynomial below has exactly the linear terms and
Hessian coefficients displayed in the source.

The pointwise remainder is bounded by
`40 (|w| + |u| + |v|)³` when each deviation has absolute value at
most `1/2` and `|r| ≤ 1`. The denominator is then positive. The
proof multiplies the proved scalar reciprocal-square-root estimates
and explicitly bounds the omitted polynomial monomials.

The polynomial truncation bound needs only the second norm deviation
bounded above; the first enters its cubic majorant without restriction.
The final theorem imposes both local norm bounds to apply both actual
reciprocal-square-root estimates.

These deterministic bounds neither assume the covariance calculation
in the source nor supply a probabilistic remainder. The latter still
requires actual Gaussian moments and control outside this local event.
In particular, this module does not assert the source's final `1/d`
coefficient, whose covariance calculation needs correction.
-/

import Transformer.Homogenized.Section7_ReciprocalSqrtTaylor

open Real

namespace Transformer.Homogenized

/-- The genuine quadratic Taylor polynomial at `(r,1,1)` for the
source's normalized correlation. Source: arXiv:2604.01978v1,
§7.3, the displayed Hessian in `lem:delta_method`. -/
noncomputable def deltaCorrelationTaylor (r w u v : ℝ) : ℝ :=
  r + w - r * (u + v) / 2 - w * (u + v) / 2 +
    r * (3 * (u ^ 2 + v ^ 2) / 8 + u * v / 4)

/-- A cubic monomial is bounded by a common cubic majorant.
Source: arXiv:2604.01978v1, §7.3, the third-order remainder
in the Taylor step of `lem:delta_method`. -/
theorem delta_taylor_abs_triple_le_cube (a b c S : ℝ)
    (ha : |a| ≤ S) (hb : |b| ≤ S) (hc : |c| ≤ S) :
    |a * b * c| ≤ S ^ 3 := by
  have hS : 0 ≤ S := (abs_nonneg a).trans ha
  simp only [abs_mul]
  calc
    _ ≤ S * S * S := by gcongr
    _ = _ := by ring

/-- Multiplying the two scalar quadratic polynomials leaves only
third- and fourth-degree monomials after the genuine quadratic
correlation polynomial is removed. Source: arXiv:2604.01978v1,
§7.3, `lem:delta_method`, the displayed Hessian. -/
theorem delta_taylor_polynomial_product_remainder (r w u v : ℝ) :
    (r + w) * ((1 - u / 2 + 3 * u ^ 2 / 8) * (1 - v / 2 + 3 * v ^ 2 / 8)) -
      deltaCorrelationTaylor r w u v =
        w * (3 * (u ^ 2 + v ^ 2) / 8 + u * v / 4) +
          (r + w) * (-(3 / 16) * (u * v * v) - (3 / 16) * (u * u * v) +
            (9 / 64) * (u * u * v * v)) := by
  unfold deltaCorrelationTaylor
  ring

/-- The actual polynomial truncation has a cubic bound on the
source's local norm event. Source: arXiv:2604.01978v1, §7.3,
the Taylor-Lagrange step in `lem:delta_method`. -/
theorem delta_taylor_polynomial_product_bound (r w u v : ℝ)
    (hr : |r| ≤ 1) (hw : |w| ≤ 1 / 2) (hv : |v| ≤ 1 / 2) :
    |(r + w) * ((1 - u / 2 + 3 * u ^ 2 / 8) * (1 - v / 2 + 3 * v ^ 2 / 8)) -
      deltaCorrelationTaylor r w u v| ≤ 3 * (|w| + |u| + |v|) ^ 3 := by
  let S := |w| + |u| + |v|
  have hS : 0 ≤ S := by dsimp [S]; positivity
  have hwS : |w| ≤ S := by dsimp [S]; linarith [abs_nonneg u, abs_nonneg v]
  have huS : |u| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg v]
  have hvS : |v| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg u]
  have h1 := delta_taylor_abs_triple_le_cube w u u S hwS huS huS
  have h2 := delta_taylor_abs_triple_le_cube w v v S hwS hvS hvS
  have h3 := delta_taylor_abs_triple_le_cube w u v S hwS huS hvS
  have h4 := delta_taylor_abs_triple_le_cube u v v S huS hvS hvS
  have h5 := delta_taylor_abs_triple_le_cube u u v S huS huS hvS
  have h6 : |u * u * v * v| ≤ S ^ 3 := by
    rw [abs_mul]
    calc
      _ ≤ S ^ 3 * 1 := mul_le_mul h5 (hv.trans (by norm_num)) (abs_nonneg _) (by positivity)
      _ = _ := mul_one _
  have hA : |w * (3 * (u ^ 2 + v ^ 2) / 8 + u * v / 4)| ≤ S ^ 3 := by
    have he : w * (3 * (u ^ 2 + v ^ 2) / 8 + u * v / 4) =
        (3 / 8) * (w * u * u) + (3 / 8) * (w * v * v) + (1 / 4) * (w * u * v) := by ring
    rw [he]
    have h := abs_add_three ((3 / 8) * (w * u * u)) ((3 / 8) * (w * v * v))
      ((1 / 4) * (w * u * v))
    norm_num [abs_mul] at h
    simp only [abs_mul] at h1 h2 h3
    nlinarith
  have hB : |-(3 / 16) * (u * v * v) - (3 / 16) * (u * u * v) +
      (9 / 64) * (u * u * v * v)| ≤ S ^ 3 := by
    have h := abs_add_three (-(3 / 16) * (u * v * v)) (-(3 / 16) * (u * u * v))
      ((9 / 64) * (u * u * v * v))
    rw [sub_eq_add_neg]
    simp only [neg_mul]
    simp only [neg_mul] at h
    norm_num [abs_mul] at h
    norm_num [abs_mul] at h4 h5 h6
    nlinarith [pow_nonneg hS 3]
  have hrw : |r + w| ≤ 2 := (abs_add_le r w).trans (by linarith)
  rw [delta_taylor_polynomial_product_remainder]
  calc
    _ ≤ |w * (3 * (u ^ 2 + v ^ 2) / 8 + u * v / 4)| + |r + w| *
        |-(3 / 16) * (u * v * v) - (3 / 16) * (u * u * v) +
          (9 / 64) * (u * u * v * v)| := by
      have h := abs_add_le (w * (3 * (u ^ 2 + v ^ 2) / 8 + u * v / 4))
        ((r + w) * (-(3 / 16) * (u * v * v) - (3 / 16) * (u * u * v) +
          (9 / 64) * (u * u * v * v)))
      rw [abs_mul (r + w)] at h
      exact h
    _ ≤ S ^ 3 + 2 * S ^ 3 := add_le_add hA (mul_le_mul hrw hB (abs_nonneg _) (by norm_num))
    _ = _ := by ring

/-- **Explicit local cubic correlation remainder.** The actual
normalized correlation differs from the genuine quadratic Taylor
polynomial by at most `40 (|w|+|u|+|v|)³` on the stated local event.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`, with the
pointwise remainder made explicit before any expectation is taken. -/
theorem delta_correlation_taylor_bound (r w u v : ℝ)
    (hr : |r| ≤ 1) (hw : |w| ≤ 1 / 2) (hu : |u| ≤ 1 / 2) (hv : |v| ≤ 1 / 2) :
    |(r + w) / Real.sqrt ((1 + u) * (1 + v)) - deltaCorrelationTaylor r w u v| ≤
      40 * (|w| + |u| + |v|) ^ 3 := by
  let S := |w| + |u| + |v|
  have hS : 0 ≤ S := by dsimp [S]; positivity
  have huS : |u| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg v]
  have hvS : |v| ≤ S := by dsimp [S]; linarith [abs_nonneg w, abs_nonneg u]
  have hucube := pow_le_pow_left₀ (abs_nonneg u) huS 3
  have hvcube := pow_le_pow_left₀ (abs_nonneg v) hvS 3
  have hrw : |r + w| ≤ 2 := (abs_add_le r w).trans (by linarith)
  have hden : (r + w) / Real.sqrt ((1 + u) * (1 + v)) =
      (r + w) * ((Real.sqrt (1 + u))⁻¹ * (Real.sqrt (1 + v))⁻¹) := by
    rw [div_eq_mul_inv, Real.sqrt_mul (by linarith [(abs_le.mp hu).1]), mul_inv]
  rw [hden]
  calc
    _ ≤ |(r + w) * ((Real.sqrt (1 + u))⁻¹ * (Real.sqrt (1 + v))⁻¹) -
          (r + w) * ((1 - u / 2 + 3 * u ^ 2 / 8) * (1 - v / 2 + 3 * v ^ 2 / 8))| +
        |(r + w) * ((1 - u / 2 + 3 * u ^ 2 / 8) * (1 - v / 2 + 3 * v ^ 2 / 8)) -
          deltaCorrelationTaylor r w u v| := abs_sub_le _ _ _
    _ ≤ 16 * (|u| ^ 3 + |v| ^ 3) + 3 * S ^ 3 := add_le_add
      (reciprocal_sqrt_product_weighted_taylor_bound (r + w) u v hrw hu hv)
      (delta_taylor_polynomial_product_bound r w u v hr hw hv)
    _ ≤ _ := by nlinarith [pow_nonneg hS 3]

/-- The local hypotheses hold jointly at nonzero numerator, norm
and mixed deviations, with a nontrivial original correlation. -/
example : |(1 / 2 : ℝ)| ≤ 1 ∧ |(-1 / 4 : ℝ)| ≤ 1 / 2 ∧
    |(1 / 4 : ℝ)| ≤ 1 / 2 ∧ |(-1 / 4 : ℝ)| ≤ 1 / 2 ∧
    |(1 / 4 : ℝ)| ≤ 1 ∧ |(-1 / 4 : ℝ)| ≤ 1 := by norm_num

end Transformer.Homogenized
