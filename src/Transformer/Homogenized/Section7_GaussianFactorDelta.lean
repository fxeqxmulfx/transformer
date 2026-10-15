/-
# Uniform normalized-correlation expansion for one actual Gaussian factor

ArXiv:2604.01978v1, §7.3, `lem:delta_method`. For the images
of every two original unit inputs under one Gaussian matrix factor,
the expected normalized correlation is

  `r + (r³-r)/(2d) + O(d^(-3/2))`.

The constant is independent of positive dimension and of both inputs.
The proof rescales the original images by `1/√d`, which preserves
their actual normalized correlation. Their centered score and norm
deviations are exactly those whose genuine Gaussian moments have
already been derived from the original independent entries.

The integrated global cubic Taylor error is bounded by the proved
uniform third absolute moments. The genuine expected quadratic
polynomial is computed from the exact actual covariance formulas.
No covariance or remainder estimate is assumed as an input here.

The displayed coefficient is a single-factor correction. The source
asserts the same coefficient for a product of two independent factors.
This theorem does not identify those two distinct original laws;
the product expansion requires a further composition of factor laws.
-/

import Transformer.Homogenized.Section7_GaussianTaylorMean
import Transformer.Homogenized.Section7_GaussianScoreThird
import Transformer.Homogenized.Barycenter

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- Rescaling both original images by `1/√d` gives exactly the
dimension-normalized score, including zero dimension.
Source: arXiv:2604.01978v1, §7.3, the centered score
used in `lem:delta_method`, for one Gaussian factor. -/
theorem delta_dimension_scaled_inner {d : ℕ} (p q : EucSpace d) :
    inner ℝ ((Real.sqrt (d : ℝ))⁻¹ • p) ((Real.sqrt (d : ℝ))⁻¹ • q) =
      inner ℝ p q / (d : ℝ) := by
  rw [real_inner_smul_left, real_inner_smul_right, div_eq_mul_inv]
  calc
    _ = (Real.sqrt (d : ℝ))⁻¹ ^ 2 * inner ℝ p q := by ring
    _ = _ := by rw [inv_pow, Real.sq_sqrt (Nat.cast_nonneg d)]; ring

/-- The same original rescaling gives exactly the dimension-normalized
squared norm. Source: arXiv:2604.01978v1, §7.3,
the norm variables of `lem:delta_method`, for one factor. -/
theorem delta_dimension_scaled_norm_sq {d : ℕ} (p : EucSpace d) :
    ‖(Real.sqrt (d : ℝ))⁻¹ • p‖ ^ 2 = ‖p‖ ^ 2 / (d : ℝ) := by
  rw [norm_smul_of_nonneg (inv_nonneg.mpr (Real.sqrt_nonneg _)), mul_pow,
    inv_pow, Real.sq_sqrt (Nat.cast_nonneg d), div_eq_mul_inv, mul_comm]

/-- Positive-dimensional rescaling leaves the actual original
normalized query correlation unchanged, even at zero vectors.
Source: arXiv:2604.01978v1, §7.3, the normalized queries
in `lem:delta_method`. -/
theorem delta_dimension_scaled_normalized_inner {d : ℕ} (hd : 0 < d) (p q : EucSpace d) :
    inner ℝ (normalizeLayer ((Real.sqrt (d : ℝ))⁻¹ • p))
      (normalizeLayer ((Real.sqrt (d : ℝ))⁻¹ • q)) =
        inner ℝ (normalizeLayer p) (normalizeLayer q) := by
  have hscale : 0 < (Real.sqrt (d : ℝ))⁻¹ :=
    inv_pos.mpr (Real.sqrt_pos.mpr (by exact_mod_cast hd))
  rw [normalizeLayer_smul hscale, normalizeLayer_smul hscale]

/-- The actual unit-query quadratic polynomial is integrable from
the original second moments alone. Source: arXiv:2604.01978v1,
§7.3, the expected Hessian polynomial in `lem:delta_method`. -/
theorem integrable_gaussianMatrix_unit_delta_polynomial {d : ℕ}
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      deltaCorrelationTaylor (inner ℝ x y)
        (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y)
        (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1)
        (‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1)) (gaussianMatrixLaw d) := by
  simpa only [hx, hy, one_pow] using
    integrable_deltaCorrelationTaylor (gaussianMatrixLaw d) (inner ℝ x y) _ _ _
      (memLp_gaussianMatrix_query_score_deviation x y)
      (memLp_gaussianMatrix_query_norm_deviation x) (memLp_gaussianMatrix_query_norm_deviation y)

/-- The genuine polynomial at the actual rescaled vector queries
has the same exact Gaussian expectation. The rescaling changes
their raw moments and preserves their normalized correlation.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the Taylor polynomial for the dimension-normalized single factor. -/
theorem integral_gaussianMatrix_unit_scaled_delta_taylor_mean {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      deltaCorrelationTaylor (inner ℝ x y)
        (inner ℝ ((Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M x)
          ((Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M y) - inner ℝ x y)
        (‖(Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M x‖ ^ 2 - 1)
        (‖(Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M y‖ ^ 2 - 1) ∂gaussianMatrixLaw d) =
          inner ℝ x y + (inner ℝ x y ^ 3 - inner ℝ x y) / (2 * (d : ℝ)) := by
  simp only [delta_dimension_scaled_inner, delta_dimension_scaled_norm_sq]
  exact integral_gaussianMatrix_unit_delta_taylor_mean hd x y hx hy

/-- **Uniform expected correlation expansion for one original Gaussian factor.**
All original unit input pairs and positive dimensions use the same
positive constant. The actual matrix images are normalized directly.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the single-factor expansion needed before composing the product law. -/
theorem gaussianMatrix_factor_delta_method :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
      |(∫ M : Matrix (Fin d) (Fin d) ℝ,
          inner ℝ (normalizeLayer (Matrix.toEuclideanLin M x))
            (normalizeLayer (Matrix.toEuclideanLin M y)) ∂gaussianMatrixLaw d) -
          (inner ℝ x y + (inner ℝ x y ^ 3 - inner ℝ x y) / (2 * (d : ℝ)))| ≤
        C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, hmom⟩ := gaussianMatrix_unit_delta_deviation_third_uniform
  refine ⟨40 * C, by positivity, ?_⟩
  intro d hd x y hx hy
  let p : Matrix (Fin d) (Fin d) ℝ → EucSpace d :=
    fun M => (Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M x
  let q : Matrix (Fin d) (Fin d) ℝ → EucSpace d :=
    fun M => (Real.sqrt (d : ℝ))⁻¹ • Matrix.toEuclideanLin M y
  have hp : Measurable p := by
    have hmx := measurable_gaussianMatrix_query x
    dsimp [p]
    fun_prop
  have hq : Measurable q := by
    have hmy := measurable_gaussianMatrix_query y
    dsimp [q]
    fun_prop
  have hr : |inner ℝ x y| ≤ 1 := by
    simpa only [hx, hy, mul_one] using abs_real_inner_le_norm x y
  obtain ⟨hi, hb⟩ := hmom d hd x y hx hy
  have hscaled : Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      (|inner ℝ (p M) (q M) - inner ℝ x y| + |‖p M‖ ^ 2 - 1| + |‖q M‖ ^ 2 - 1|) ^ 3)
        (gaussianMatrixLaw d) := by
    simpa only [p, q, delta_dimension_scaled_inner, delta_dimension_scaled_norm_sq] using hi
  have he := integral_normalized_inner_delta_taylor_error
    (gaussianMatrixLaw d) (inner ℝ x y) hr p q hp hq hscaled
  simp only [p, q, delta_dimension_scaled_normalized_inner hd] at he
  rw [integral_gaussianMatrix_unit_scaled_delta_taylor_mean hd x y hx hy] at he
  simp only [delta_dimension_scaled_inner, delta_dimension_scaled_norm_sq] at he
  calc
    _ ≤ 40 * ∫ M : Matrix (Fin d) (Fin d) ℝ,
        (|inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| +
          |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| +
            |‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1|) ^ 3 ∂gaussianMatrixLaw d := he
    _ ≤ 40 * (C * (d : ℝ) ^ (-(3 : ℝ) / 2)) := mul_le_mul_of_nonneg_left hb (by norm_num)
    _ = _ := by ring

/-- Genuine coordinate unit inputs in positive dimension satisfy
all original hypotheses jointly. Their images are the actual matrix
columns and the theorem uses their actual normalized correlation. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 := by
  exact ⟨by decide, by simp, by simp⟩

end Transformer.Homogenized
