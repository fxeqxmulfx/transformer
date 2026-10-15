/-
# Actual intermediate Gaussian correlation and cubic-bias estimates

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, composition
of the two original independent matrix factors. The first factor
produces an actual random normalized correlation `R` from the
original deterministic input correlation `r`.

The exact original second moments and the global normalization
bound give `E|R-r| ≤ 6√(2/d)` uniformly over every unit input pair.
The actual cubic correction then satisfies
`|E(R³-R) - (r³-r)| ≤ 12√(2/d)`.

After dividing that change by `2d`, its contribution is at most
`6√2 d^(-3/2)`. This is the rate needed to compose the two
single-factor expansions without assuming that the intermediate
correlation is deterministic. The random original normalized
queries are used throughout, including any zero image.

These estimates concern one actual independent Gaussian factor.
They do not assign a Gaussian law to the original product matrix.
-/

import Transformer.Homogenized.Section7_GaussianFactorDelta
import Transformer.Homogenized.Section7_CorrelationDeviation

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- First absolute norm deviation under the actual unit-query law.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the intermediate single-factor norm fluctuations. -/
theorem integral_gaussianMatrix_unit_query_abs_deviation_le {d : ℕ} (hd : 0 < d)
    (x : EucSpace d) (hx : ‖x‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| ∂gaussianMatrixLaw d) ≤ Real.sqrt (2 / (d : ℝ)) := by
  have hX : MemLp (fun M : Matrix (Fin d) (Fin d) ℝ =>
      ‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1) 2 (gaussianMatrixLaw d) := by
    simpa only [hx, one_pow] using memLp_gaussianMatrix_query_norm_deviation x
  have h := first_abs_moment_le_sqrt_second (gaussianMatrixLaw d) _ hX
  rw [integral_gaussianMatrix_unit_query_deviation_sq hd x hx] at h
  exact h

/-- First absolute score deviation under the original matrix law.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the actual intermediate correlation fluctuations. -/
theorem integral_gaussianMatrix_unit_score_abs_deviation_le {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y|
        ∂gaussianMatrixLaw d) ≤ Real.sqrt (2 / (d : ℝ)) := by
  exact (first_abs_moment_le_sqrt_second (gaussianMatrixLaw d) _
    (memLp_gaussianMatrix_query_score_deviation x y)).trans
      (Real.sqrt_le_sqrt (integral_gaussianMatrix_unit_score_deviation_sq_le hd x y hx hy))

/-- The actual normalized correlation deviation is integrable for
arbitrary original inputs. Source: arXiv:2604.01978v1,
§7.3, the normalized correlation in `lem:delta_method`. -/
theorem integrable_gaussianMatrix_correlation_deviation {d : ℕ} (x y : EucSpace d) :
    Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x))
        (normalizeLayer (Matrix.toEuclideanLin M y)) - inner ℝ x y|) (gaussianMatrixLaw d) := by
  have h := (integrable_normalized_inner (gaussianMatrixLaw d)
    (fun M => Matrix.toEuclideanLin M x) (fun M => Matrix.toEuclideanLin M y)
    (measurable_gaussianMatrix_query x) (measurable_gaussianMatrix_query y)).sub
      (integrable_const (inner ℝ x y))
  simpa only [Real.norm_eq_abs, Pi.sub_apply] using h.norm

/-- **Uniform first absolute deviation of the actual intermediate correlation.**
The bound holds for every original unit input pair in positive dimension.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the actual normalized correlation after one independent factor. -/
theorem integral_gaussianMatrix_unit_correlation_deviation_le {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      |inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x))
        (normalizeLayer (Matrix.toEuclideanLin M y)) - inner ℝ x y| ∂gaussianMatrixLaw d) ≤
          6 * Real.sqrt (2 / (d : ℝ)) := by
  have hr : |inner ℝ x y| ≤ 1 := by
    simpa only [hx, hy, mul_one] using abs_real_inner_le_norm x y
  have iw : Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y|)
        (gaussianMatrixLaw d) := by
    simpa only [Real.norm_eq_abs] using (memLp_gaussianMatrix_query_score_deviation x y).integrable (by norm_num) |>.norm
  have iu : Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1|) (gaussianMatrixLaw d) := by
    simpa only [Real.norm_eq_abs, hx, one_pow] using
      (memLp_gaussianMatrix_query_norm_deviation x).integrable (by norm_num) |>.norm
  have iv : Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1|) (gaussianMatrixLaw d) := by
    simpa only [Real.norm_eq_abs, hy, one_pow] using
      (memLp_gaussianMatrix_query_norm_deviation y).integrable (by norm_num) |>.norm
  have isum : Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| +
        |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| + |‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1|)
          (gaussianMatrixLaw d) := by convert (iw.add iu).add iv using 1
  have hpoint (M : Matrix (Fin d) (Fin d) ℝ) := delta_normalized_inner_deviation_le (inner ℝ x y) hr
    ((Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M x) ((Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M y)
  simp only [delta_dimension_scaled_normalized_inner hd, delta_dimension_scaled_inner,
    delta_dimension_scaled_norm_sq] at hpoint
  have hb := integral_mono (integrable_gaussianMatrix_correlation_deviation x y) (isum.const_mul 2) hpoint
  rw [integral_const_mul, integral_add (by convert iw.add iu using 1) iv, integral_add iw iu] at hb
  linarith [integral_gaussianMatrix_unit_score_abs_deviation_le hd x y hx hy,
    integral_gaussianMatrix_unit_query_abs_deviation_le hd x hx,
    integral_gaussianMatrix_unit_query_abs_deviation_le hd y hy]

/-- The cubic bias at the actual intermediate correlation has
the original deterministic bias up to the proved fluctuation bound.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the change in the second factor's cubic correction. -/
theorem integral_gaussianMatrix_unit_cubic_bias_deviation_le {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    |(∫ M : Matrix (Fin d) (Fin d) ℝ,
        ((inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x))
          (normalizeLayer (Matrix.toEuclideanLin M y))) ^ 3 -
            inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x))
              (normalizeLayer (Matrix.toEuclideanLin M y))) ∂gaussianMatrixLaw d) -
                (inner ℝ x y ^ 3 - inner ℝ x y)| ≤ 12 * Real.sqrt (2 / (d : ℝ)) := by
  have hR := measurable_normalized_inner _ _ (measurable_gaussianMatrix_query x) (measurable_gaussianMatrix_query y)
  have hr : |inner ℝ x y| ≤ 1 := by
    simpa only [hx, hy, mul_one] using abs_real_inner_le_norm x y
  have hbound (M : Matrix (Fin d) (Fin d) ℝ) :
      |inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x)) (normalizeLayer (Matrix.toEuclideanLin M y))| ≤ 1 := by
    exact (abs_real_inner_le_norm _ _).trans
      ((mul_le_mul (norm_normalizeLayer_le_one _) (norm_normalizeLayer_le_one _)
        (norm_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)).trans_eq (one_mul _))
  have hb := integral_delta_cubic_bias_deviation_le (gaussianMatrixLaw d) _ hR hbound (inner ℝ x y) hr
  linarith [integral_gaussianMatrix_unit_correlation_deviation_le hd x y hx hy]

/-- Dividing the actual intermediate cubic-bias change by `2d`
gives the full `d^(-3/2)` composition rate.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the uniform error when combining the independent factor corrections. -/
theorem integral_gaussianMatrix_unit_cubic_bias_scaled_deviation_le {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    |((∫ M : Matrix (Fin d) (Fin d) ℝ,
        ((inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x))
          (normalizeLayer (Matrix.toEuclideanLin M y))) ^ 3 -
            inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x))
              (normalizeLayer (Matrix.toEuclideanLin M y))) ∂gaussianMatrixLaw d) -
                (inner ℝ x y ^ 3 - inner ℝ x y)) / (2 * (d : ℝ))| ≤
                  6 * Real.sqrt 2 * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  have hrate : Real.sqrt (2 / (d : ℝ)) / (d : ℝ) =
      Real.sqrt 2 * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
    have h := sqrt_second_fourth_dimension_rate 1 d (by norm_num) hdR
    rw [Real.sqrt_div (by norm_num : (0 : ℝ) ≤ 1), Real.sqrt_one, Real.sqrt_sq hdR.le] at h
    simpa only [mul_one, div_eq_mul_inv, one_mul] using h
  rw [abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 * d)]
  calc
    _ ≤ (12 * Real.sqrt (2 / (d : ℝ))) / (2 * (d : ℝ)) :=
      div_le_div_of_nonneg_right (integral_gaussianMatrix_unit_cubic_bias_deviation_le hd x y hx hy) (by positivity)
    _ = 6 * (Real.sqrt (2 / (d : ℝ)) / (d : ℝ)) := by ring
    _ = _ := by rw [hrate]; ring

/-- Positive dimension and genuine coordinate unit queries
jointly satisfy every original hypothesis of the fluctuation estimates. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 := by
  exact ⟨by decide, by simp, by simp⟩

end Transformer.Homogenized
