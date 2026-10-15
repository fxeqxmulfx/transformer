/-
# Covariance of the actual Euclidean queries of one Gaussian factor

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the actual query
score and squared norms. The row-sum formulas are identified here
with the original Euclidean inner products and matrix images.

For arbitrary original inputs, the normalized score has its original
input inner product as mean. Its centered second moment and its mixed
moment with each squared-norm deviation are computed exactly. The
two squared-norm deviations have covariance `2 ⟨x,y⟩²/d`.
The normalized score and its deviation are proved square integrable.

These are exact formulas for one original independent Gaussian factor,
with no unit-norm restriction needed for the identities. The unit-input
second-moment bound follows from the genuine Cauchy–Schwarz inequality.
The source's two-factor query is still treated by composing the factors;
no Gaussian law is assigned to the product matrix itself.
-/

import Transformer.Homogenized.Section7_GaussianGramCovariance
import Transformer.Homogenized.Section7_GaussianUnitQuery

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual coefficient dot product is the input inner product.
Source: arXiv:2604.01978v1, §7.3, the center in `lem:delta_method`. -/
theorem gaussianQuery_coeff_inner {d : ℕ} (x y : EucSpace d) :
    dotProduct (WithLp.ofLp x) (WithLp.ofLp y) = inner ℝ x y := by
  simp only [dotProduct, PiLp.inner_apply, RCLike.inner_apply, conj_trivial]
  exact Finset.sum_congr rfl fun i hi => mul_comm _ _

/-- The actual query score equals its original row sum. Source:
arXiv:2604.01978v1, §7.3, `lem:delta_method`, original matrix action. -/
theorem gaussianMatrix_query_inner {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ) (x y : EucSpace d) :
    inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) =
      ∑ i : Fin d, dotProduct (WithLp.ofLp x) (M i) * dotProduct (WithLp.ofLp y) (M i) := by
  simp only [PiLp.inner_apply, RCLike.inner_apply, conj_trivial, gaussianMatrix_query_coordinate]
  exact Finset.sum_congr rfl fun i hi => mul_comm _ _

/-- The actual query squared norm equals its original row sum.
Source: arXiv:2604.01978v1, §7.3, squared norms in `lem:delta_method`. -/
theorem gaussianMatrix_query_norm_sq {d : ℕ} (M : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    ‖Matrix.toEuclideanLin M x‖ ^ 2 = ∑ i : Fin d, dotProduct (WithLp.ofLp x) (M i) ^ 2 := by
  simp only [EuclideanSpace.real_norm_sq_eq, gaussianMatrix_query_coordinate]

/-- The actual normalized Euclidean score is square integrable.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem memLp_gaussianMatrix_query_score {d : ℕ} (x y : EucSpace d) :
    MemLp (fun M : Matrix (Fin d) (Fin d) ℝ =>
      inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ)) 2 (gaussianMatrixLaw d) := by
  have h := memLp_gaussianMatrix_row_average
    (fun z => dotProduct (WithLp.ofLp x) z * dotProduct (WithLp.ofLp y) z)
    (memLp_gaussianRow_linear_product (WithLp.ofLp x) (WithLp.ofLp y))
  simpa only [← gaussianMatrix_query_inner] using h

/-- The actual score deviation is square integrable. Source:
arXiv:2604.01978v1, §7.3, centered scores in `lem:delta_method`. -/
theorem memLp_gaussianMatrix_query_score_deviation {d : ℕ} (x y : EucSpace d) :
    MemLp (fun M : Matrix (Fin d) (Fin d) ℝ =>
      inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y)
        2 (gaussianMatrixLaw d) := by
  simpa only [← gaussianMatrix_query_inner, gaussianQuery_coeff_inner] using
    memLp_gaussianMatrix_score_deviation (WithLp.ofLp x) (WithLp.ofLp y)

/-- The actual squared-norm deviation is square integrable at its
original input-norm center. Source: arXiv:2604.01978v1,
§7.3, the norm deviations in `lem:delta_method`. -/
theorem memLp_gaussianMatrix_query_norm_deviation {d : ℕ} (x : EucSpace d) :
    MemLp (fun M : Matrix (Fin d) (Fin d) ℝ =>
      ‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2) 2 (gaussianMatrixLaw d) := by
  simpa only [real_inner_self_eq_norm_sq] using memLp_gaussianMatrix_query_score_deviation x x

/-- The actual normalized score is centered at the input inner
product. Source: arXiv:2604.01978v1, §7.3, the mean in `lem:delta_method`. -/
theorem integral_gaussianMatrix_query_score {d : ℕ} (hd : 0 < d) (x y : EucSpace d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ)
        ∂gaussianMatrixLaw d) = inner ℝ x y := by
  have h := integral_gaussianMatrix_row_average hd
    (fun z => dotProduct (WithLp.ofLp x) z * dotProduct (WithLp.ofLp y) z)
    (memLp_gaussianRow_linear_product (WithLp.ofLp x) (WithLp.ofLp y))
  rw [integral_gaussianRow_linear_product, gaussianQuery_coeff_inner] at h
  simpa only [← gaussianMatrix_query_inner] using h

/-- The actual score deviation has mean zero. Source:
arXiv:2604.01978v1, §7.3, the centering in `lem:delta_method`. -/
theorem integral_gaussianMatrix_query_score_deviation {d : ℕ} (hd : 0 < d) (x y : EucSpace d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y)
        ∂gaussianMatrixLaw d) = 0 := by
  rw [integral_sub ((memLp_gaussianMatrix_query_score x y).integrable (by norm_num))
    (integrable_const _), integral_gaussianMatrix_query_score hd]
  simp

/-- The actual squared-norm deviation has mean zero at its original
center. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianMatrix_query_norm_deviation {d : ℕ} (hd : 0 < d) (x : EucSpace d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2) ∂gaussianMatrixLaw d) = 0 := by
  simpa only [real_inner_self_eq_norm_sq] using integral_gaussianMatrix_query_score_deviation hd x x

/-- Actual squared query norms have their original exact mixed
deviation moment. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianMatrix_query_square_deviation_product {d : ℕ} (hd : 0 < d) (x y : EucSpace d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2) *
        (‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - ‖y‖ ^ 2) ∂gaussianMatrixLaw d) =
          2 * inner ℝ x y ^ 2 / (d : ℝ) := by
  simpa only [← gaussianMatrix_query_norm_sq, ← EuclideanSpace.real_norm_sq_eq, gaussianQuery_coeff_inner] using
    integral_gaussianMatrix_square_deviation_product hd (WithLp.ofLp x) (WithLp.ofLp y)

/-- The actual score and first squared query norm have their exact
mixed deviation moment. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianMatrix_query_score_square_deviation {d : ℕ} (hd : 0 < d) (x y : EucSpace d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y) *
        (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2) ∂gaussianMatrixLaw d) =
          2 * ‖x‖ ^ 2 * inner ℝ x y / (d : ℝ) := by
  simpa only [← gaussianMatrix_query_inner, ← gaussianMatrix_query_norm_sq,
    ← EuclideanSpace.real_norm_sq_eq, gaussianQuery_coeff_inner] using
    integral_gaussianMatrix_score_square_deviation hd (WithLp.ofLp x) (WithLp.ofLp y)

/-- **Exact actual score-deviation second moment.** The formula keeps
the original input norms and correlation.
Source: arXiv:2604.01978v1, §7.3, the covariance in `lem:delta_method`. -/
theorem integral_gaussianMatrix_query_score_deviation_sq {d : ℕ} (hd : 0 < d) (x y : EucSpace d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y) ^ 2
        ∂gaussianMatrixLaw d) = (‖x‖ ^ 2 * ‖y‖ ^ 2 + inner ℝ x y ^ 2) / (d : ℝ) := by
  simpa only [← gaussianMatrix_query_inner, ← EuclideanSpace.real_norm_sq_eq, gaussianQuery_coeff_inner] using
    integral_gaussianMatrix_score_deviation_sq hd (WithLp.ofLp x) (WithLp.ofLp y)

/-- For actual unit inputs the score deviation second moment is
uniformly bounded by `2/d`. Source: arXiv:2604.01978v1,
§7.3, the single-factor covariance bound in `lem:delta_method`. -/
theorem integral_gaussianMatrix_unit_score_deviation_sq_le {d : ℕ} (hd : 0 < d)
    (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y) ^ 2
        ∂gaussianMatrixLaw d) ≤ 2 / (d : ℝ) := by
  have hxy : |inner ℝ x y| ≤ 1 := by simpa only [hx, hy, mul_one] using abs_real_inner_le_norm x y
  have hs : inner ℝ x y ^ 2 ≤ 1 := by simpa only [sq_abs, one_pow] using pow_le_pow_left₀ (abs_nonneg _) hxy 2
  rw [integral_gaussianMatrix_query_score_deviation_sq hd, hx, hy]
  exact div_le_div_of_nonneg_right (by nlinarith) (Nat.cast_nonneg d)

/-- Positive dimension and two genuine unit queries satisfy all
input hypotheses jointly. The formulas also allow arbitrary unnormalized inputs. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 := by
  exact ⟨by decide, by simp, by simp⟩

end Transformer.Homogenized
