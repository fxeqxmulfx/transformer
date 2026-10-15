/-
# Conditional moments of the original Gaussian product query

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, and §2.3.3,
initialization (G). For a fixed row `g` of the second original
factor, the query coordinates are the linear forms `g · Wᵢ` in
the independent original rows of the first factor.

Their conditional variances are exactly `S = ∑ gᵢ²`. Their fourth
moments are `3 S²`; products of squares in distinct output rows
have conditional expectation `S²`. The squared query norm therefore
has conditional second moment `d (d + 2) S²`.

These are conditional calculations for the actual two-factor query,
not moments of a replacement Gaussian product matrix. In particular,
the random conditional variance must itself be integrated under the
original key-row law. All terms are integrable before sums of their
integrals are distributed. Zero rows and empty dimensions are included.
-/

import Transformer.Homogenized.Section7_GaussianMatrixLaw
import Transformer.Homogenized.Section7_GaussianRowFourth

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Every conditional original query-coordinate power is integrable.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_gaussianMatrix_row_linear_pow {d : ℕ} (g : Fin d → ℝ)
    (i : Fin d) (n : ℕ) :
    Integrable (fun M : Matrix (Fin d) (Fin d) ℝ => dotProduct g (M i) ^ n)
      (gaussianMatrixLaw d) :=
  (gaussianMatrix_row_linear_hasLaw g i).integrable_comp
    (integrable_gaussianReal_pow _ n)

/-- The exact conditional coordinate variance under the original
matrix law. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianMatrix_row_linear_sq {d : ℕ} (g : Fin d → ℝ) (i : Fin d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ, dotProduct g (M i) ^ 2 ∂gaussianMatrixLaw d) =
      ∑ k : Fin d, g k ^ 2 := by
  have h := integral_sq_of_map_eq_gaussianReal
    (by fun_prop : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ => dotProduct g (M i)))
    (gaussianMatrix_row_linear_hasLaw g i).map_eq
  rw [Real.coe_toNNReal _ (Finset.sum_nonneg fun k hk => sq_nonneg _)] at h
  exact h

/-- The exact conditional coordinate fourth moment under the
original matrix law. Source: arXiv:2604.01978v1, §7.3. -/
theorem integral_gaussianMatrix_row_linear_fourth {d : ℕ} (g : Fin d → ℝ) (i : Fin d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ, dotProduct g (M i) ^ 4 ∂gaussianMatrixLaw d) =
      3 * (∑ k : Fin d, g k ^ 2) ^ 2 := by
  have h := (gaussianMatrix_row_linear_hasLaw g i).integral_comp
    (f := fun x : ℝ => x ^ 4) (by fun_prop)
  rw [integral_gaussianReal_fourth,
    Real.coe_toNNReal _ (Finset.sum_nonneg fun k hk => sq_nonneg _)] at h
  exact h

/-- Conditional products of original query-coordinate squares are
integrable. Source: arXiv:2604.01978v1, §7.3, the covariance in
`lem:delta_method`. -/
theorem integrable_gaussianMatrix_linear_square_product {d : ℕ}
    (g : Fin d → ℝ) (i j : Fin d) :
    Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      dotProduct g (M i) ^ 2 * dotProduct g (M j) ^ 2) (gaussianMatrixLaw d) := by
  apply ((integrable_gaussianMatrix_row_linear_pow g i 4).add
    (integrable_gaussianMatrix_row_linear_pow g j 4)).mono' (by fun_prop)
  filter_upwards with M
  rw [Real.norm_of_nonneg (by positivity)]
  change dotProduct g (M i) ^ 2 * dotProduct g (M j) ^ 2 ≤
    dotProduct g (M i) ^ 4 + dotProduct g (M j) ^ 4
  nlinarith [sq_nonneg (dotProduct g (M i) ^ 2 - dotProduct g (M j) ^ 2)]

/-- Conditional square products have the exact diagonal and
off-diagonal values. Source: arXiv:2604.01978v1, §7.3,
the Gaussian-factor calculation needed for `lem:delta_method`. -/
theorem integral_gaussianMatrix_linear_square_product {d : ℕ}
    (g : Fin d → ℝ) (i j : Fin d) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      dotProduct g (M i) ^ 2 * dotProduct g (M j) ^ 2 ∂gaussianMatrixLaw d) =
        (if i = j then 3 else 1) * (∑ k : Fin d, g k ^ 2) ^ 2 := by
  by_cases hij : i = j
  · subst j
    simpa only [← pow_add, ite_true] using integral_gaussianMatrix_row_linear_fourth g i
  · have h := (indepFun_gaussianMatrix_row_linear g i j hij).comp
      (by fun_prop : Measurable (fun x : ℝ => x ^ 2))
      (by fun_prop : Measurable (fun x : ℝ => x ^ 2))
    simp only [Function.comp_def] at h
    have he := h.integral_mul_eq_mul_integral
      (integrable_gaussianMatrix_row_linear_pow g i 2).aestronglyMeasurable
      (integrable_gaussianMatrix_row_linear_pow g j 2).aestronglyMeasurable
    change (∫ M : Matrix (Fin d) (Fin d) ℝ,
      dotProduct g (M i) ^ 2 * dotProduct g (M j) ^ 2 ∂gaussianMatrixLaw d) =
        (∫ M, dotProduct g (M i) ^ 2 ∂gaussianMatrixLaw d) *
          (∫ M, dotProduct g (M j) ^ 2 ∂gaussianMatrixLaw d) at he
    rw [integral_gaussianMatrix_row_linear_sq, integral_gaussianMatrix_row_linear_sq] at he
    simpa only [ite_eq_right hij, one_mul, pow_two] using he

/-- The original conditional query has an integrable fourth norm
moment. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_gaussianMatrix_query_norm_fourth {d : ℕ} (g : Fin d → ℝ) :
    Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      (∑ i : Fin d, dotProduct g (M i) ^ 2) ^ 2) (gaussianMatrixLaw d) := by
  have he (M : Matrix (Fin d) (Fin d) ℝ) :
      (∑ i : Fin d, dotProduct g (M i) ^ 2) ^ 2 =
        ∑ i : Fin d, ∑ j : Fin d, dotProduct g (M i) ^ 2 * dotProduct g (M j) ^ 2 := by
    rw [pow_two, Finset.sum_mul_sum]
  simp_rw [he]
  exact integrable_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj =>
    integrable_gaussianMatrix_linear_square_product g i j

/-- The exact conditional fourth query-norm moment is
`d (d + 2) (∑ gᵢ²)²`. Source: arXiv:2604.01978v1, §7.3,
`lem:delta_method`, retaining the random key-row variance. -/
theorem integral_gaussianMatrix_query_norm_fourth {d : ℕ} (g : Fin d → ℝ) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      (∑ i : Fin d, dotProduct g (M i) ^ 2) ^ 2 ∂gaussianMatrixLaw d) =
        (d : ℝ) * (d + 2) * (∑ k : Fin d, g k ^ 2) ^ 2 := by
  have he (M : Matrix (Fin d) (Fin d) ℝ) :
      (∑ i : Fin d, dotProduct g (M i) ^ 2) ^ 2 =
        ∑ i : Fin d, ∑ j : Fin d, dotProduct g (M i) ^ 2 * dotProduct g (M j) ^ 2 := by
    rw [pow_two, Finset.sum_mul_sum]
  simp_rw [he]
  rw [integral_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj =>
    integrable_gaussianMatrix_linear_square_product g i j]
  simp_rw [integral_finsetSum _ fun j hj => integrable_gaussianMatrix_linear_square_product g _ j,
    integral_gaussianMatrix_linear_square_product]
  have hif (i j : Fin d) : (if i = j then (3 : ℝ) else 1) = 1 + if i = j then 2 else 0 := by
    split_ifs <;> norm_num
  simp_rw [hif]
  simp [Finset.sum_add_distrib, ← Finset.sum_mul, mul_add]
  ring

/-- A nonzero fixed key row in dimension two has conditional fourth
query-norm moment 200; its two output coordinate squares have mixed
moment 25. Both calculations use the same original matrix law. -/
example : (∫ M : Matrix (Fin 2) (Fin 2) ℝ,
      (∑ i : Fin 2, dotProduct ![1, 2] (M i) ^ 2) ^ 2 ∂gaussianMatrixLaw 2) = 200 ∧
    (∫ M : Matrix (Fin 2) (Fin 2) ℝ,
      dotProduct ![1, 2] (M 0) ^ 2 * dotProduct ![1, 2] (M 1) ^ 2 ∂gaussianMatrixLaw 2) = 25 := by
  constructor
  · rw [integral_gaussianMatrix_query_norm_fourth]; norm_num [Fin.sum_univ_two]
  · rw [integral_gaussianMatrix_linear_square_product]; norm_num [Fin.sum_univ_two]

end Transformer.Homogenized
