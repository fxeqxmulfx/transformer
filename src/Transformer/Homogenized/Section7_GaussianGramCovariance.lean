module

public import Transformer.Homogenized.Section7_GaussianRowAverages

/-!
# Actual single-factor Gaussian Gram covariances

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the covariance
of query scores and squared norms. This module computes the actual
row-polynomial covariances from their proved second and fourth moments,
then transfers them to the averages of the original independent rows.

For arbitrary coefficient vectors `g,h`, the row square–square
covariance is `2 (g·h)²`, the score–square covariance is
`2 (∑ gᵢ²) (g·h)`, and the score variance is
`(∑ gᵢ²) (∑ hᵢ²) + (g·h)²`. The actual matrix averages have these
covariances divided by `d` and are centered by their actual means.

All row statistics and actual average deviations lie in `L²` before
their moments are distributed. The formulas concern one original
Gaussian factor. The source's product matrix must still compose the
two factors; these single-factor formulas are not assigned directly
to that product, whose norm covariance was already shown different.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual row score is square integrable. Source:
arXiv:2604.01978v1, §7.3, the covariance inputs in `lem:delta_method`. -/
theorem memLp_gaussianRow_linear_product {d : ℕ} (g h : Fin d → ℝ) :
    MemLp (fun z : Fin d → ℝ => dotProduct g z * dotProduct h z) 2 (gaussianRowLaw d) := by
  have hm : Measurable (fun z : Fin d → ℝ => dotProduct g z * dotProduct h z) := by fun_prop
  apply (memLp_two_iff_integrable_sq hm.aestronglyMeasurable).mpr
  simpa only [mul_pow] using integrable_gaussianRow_linear_square_product g h

/-- Actual squared row linear forms lie in `L²`. Source:
arXiv:2604.01978v1, §7.3, squared query norms in `lem:delta_method`. -/
theorem memLp_gaussianRow_linear_square {d : ℕ} (g : Fin d → ℝ) :
    MemLp (fun z : Fin d → ℝ => dotProduct g z ^ 2) 2 (gaussianRowLaw d) := by
  have hm : Measurable (fun z : Fin d → ℝ => dotProduct g z ^ 2) := by fun_prop
  apply (memLp_two_iff_integrable_sq hm.aestronglyMeasurable).mpr
  simpa only [← pow_mul] using integrable_gaussianRow_linear_pow g 4

/-- The squared row linear form has its actual coefficient variance.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianRow_linear_sq {d : ℕ} (g : Fin d → ℝ) :
    (∫ z : Fin d → ℝ, dotProduct g z ^ 2 ∂gaussianRowLaw d) = ∑ i : Fin d, g i ^ 2 := by
  simpa only [dotProduct, ← pow_two] using integral_gaussianRow_linear_product g g

/-- The actual row square–square covariance is twice the squared
coefficient inner product. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem covariance_gaussianRow_linear_squares {d : ℕ} (g h : Fin d → ℝ) :
    covariance (fun z : Fin d → ℝ => dotProduct g z ^ 2) (fun z => dotProduct h z ^ 2)
      (gaussianRowLaw d) = 2 * dotProduct g h ^ 2 := by
  rw [covariance_eq_sub (memLp_gaussianRow_linear_square g) (memLp_gaussianRow_linear_square h)]
  change (∫ z : Fin d → ℝ, dotProduct g z ^ 2 * dotProduct h z ^ 2 ∂gaussianRowLaw d) -
    (∫ z, dotProduct g z ^ 2 ∂gaussianRowLaw d) * (∫ z, dotProduct h z ^ 2 ∂gaussianRowLaw d) = _
  rw [integral_gaussianRow_linear_square_product, integral_gaussianRow_linear_sq, integral_gaussianRow_linear_sq]
  ring

/-- The actual row score–square covariance keeps its true coefficient
variance. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem covariance_gaussianRow_linear_product_square {d : ℕ} (g h : Fin d → ℝ) :
    covariance (fun z : Fin d → ℝ => dotProduct g z * dotProduct h z) (fun z => dotProduct g z ^ 2)
      (gaussianRowLaw d) = 2 * (∑ i : Fin d, g i ^ 2) * dotProduct g h := by
  rw [covariance_eq_sub (memLp_gaussianRow_linear_product g h) (memLp_gaussianRow_linear_square g)]
  change (∫ z : Fin d → ℝ, (dotProduct g z * dotProduct h z) * dotProduct g z ^ 2 ∂gaussianRowLaw d) -
    (∫ z, dotProduct g z * dotProduct h z ∂gaussianRowLaw d) *
      (∫ z, dotProduct g z ^ 2 ∂gaussianRowLaw d) = _
  have he : (fun z : Fin d → ℝ => (dotProduct g z * dotProduct h z) * dotProduct g z ^ 2) =
      fun z => dotProduct g z ^ 3 * dotProduct h z := by funext z; ring
  rw [he, integral_gaussianRow_linear_cube_product, integral_gaussianRow_linear_product, integral_gaussianRow_linear_sq]
  ring

/-- The actual row score variance includes both variance product
and squared correlation. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem covariance_gaussianRow_linear_product_self {d : ℕ} (g h : Fin d → ℝ) :
    covariance (fun z : Fin d → ℝ => dotProduct g z * dotProduct h z)
      (fun z => dotProduct g z * dotProduct h z) (gaussianRowLaw d) =
        (∑ i : Fin d, g i ^ 2) * (∑ i : Fin d, h i ^ 2) + dotProduct g h ^ 2 := by
  rw [covariance_eq_sub (memLp_gaussianRow_linear_product g h) (memLp_gaussianRow_linear_product g h)]
  change (∫ z : Fin d → ℝ, (dotProduct g z * dotProduct h z) * (dotProduct g z * dotProduct h z)
    ∂gaussianRowLaw d) - (∫ z, dotProduct g z * dotProduct h z ∂gaussianRowLaw d) *
      (∫ z, dotProduct g z * dotProduct h z ∂gaussianRowLaw d) = _
  have he : (fun z : Fin d → ℝ => (dotProduct g z * dotProduct h z) * (dotProduct g z * dotProduct h z)) =
      fun z => dotProduct g z ^ 2 * dotProduct h z ^ 2 := by funext z; ring
  rw [he, integral_gaussianRow_linear_square_product, integral_gaussianRow_linear_product]
  ring

/-- Actual normalized squared-norm averages have their exact centered
mixed moment. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`, one factor. -/
theorem integral_gaussianMatrix_square_deviation_product {d : ℕ} (hd : 0 < d) (g h : Fin d → ℝ) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      ((∑ i : Fin d, dotProduct g (M i) ^ 2) / (d : ℝ) - ∑ j : Fin d, g j ^ 2) *
        ((∑ i : Fin d, dotProduct h (M i) ^ 2) / (d : ℝ) - ∑ j : Fin d, h j ^ 2)
          ∂gaussianMatrixLaw d) = 2 * dotProduct g h ^ 2 / (d : ℝ) := by
  have h := integral_gaussianMatrix_centered_row_averages hd
    (fun z => dotProduct g z ^ 2) (fun z => dotProduct h z ^ 2)
    (memLp_gaussianRow_linear_square g) (memLp_gaussianRow_linear_square h)
  rw [integral_gaussianRow_linear_sq, integral_gaussianRow_linear_sq, covariance_gaussianRow_linear_squares] at h
  exact h

/-- The actual score–square average mixed moment has the original
coefficient norms and correlation. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianMatrix_score_square_deviation {d : ℕ} (hd : 0 < d) (g h : Fin d → ℝ) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      ((∑ i : Fin d, dotProduct g (M i) * dotProduct h (M i)) / (d : ℝ) - dotProduct g h) *
        ((∑ i : Fin d, dotProduct g (M i) ^ 2) / (d : ℝ) - ∑ j : Fin d, g j ^ 2)
          ∂gaussianMatrixLaw d) = 2 * (∑ j : Fin d, g j ^ 2) * dotProduct g h / (d : ℝ) := by
  have h := integral_gaussianMatrix_centered_row_averages hd
    (fun z => dotProduct g z * dotProduct h z) (fun z => dotProduct g z ^ 2)
    (memLp_gaussianRow_linear_product g h) (memLp_gaussianRow_linear_square g)
  rw [integral_gaussianRow_linear_product, integral_gaussianRow_linear_sq,
    covariance_gaussianRow_linear_product_square] at h
  exact h

/-- The actual average score deviation has its exact second moment.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`, one original factor. -/
theorem integral_gaussianMatrix_score_deviation_sq {d : ℕ} (hd : 0 < d) (g h : Fin d → ℝ) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      ((∑ i : Fin d, dotProduct g (M i) * dotProduct h (M i)) / (d : ℝ) - dotProduct g h) ^ 2
        ∂gaussianMatrixLaw d) =
          ((∑ j : Fin d, g j ^ 2) * (∑ j : Fin d, h j ^ 2) + dotProduct g h ^ 2) / (d : ℝ) := by
  have h := integral_gaussianMatrix_centered_row_averages hd
    (fun z => dotProduct g z * dotProduct h z) (fun z => dotProduct g z * dotProduct h z)
    (memLp_gaussianRow_linear_product g h) (memLp_gaussianRow_linear_product g h)
  rw [integral_gaussianRow_linear_product, covariance_gaussianRow_linear_product_self] at h
  simpa only [pow_two] using h

/-- The actual average score deviation lies in `L²`, so its original
second moments and first absolute moments are well defined.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem memLp_gaussianMatrix_score_deviation {d : ℕ} (g h : Fin d → ℝ) :
    MemLp (fun M : Matrix (Fin d) (Fin d) ℝ =>
      (∑ i : Fin d, dotProduct g (M i) * dotProduct h (M i)) / (d : ℝ) - dotProduct g h)
        2 (gaussianMatrixLaw d) := by
  convert (memLp_gaussianMatrix_row_average (fun z => dotProduct g z * dotProduct h z)
    (memLp_gaussianRow_linear_product g h)).sub (memLp_const (dotProduct g h)) using 1

/-- Positive dimension and nonzero actual row statistics witness
the moment hypotheses. Their mixed covariances are computed, not supplied. -/
example : (0 : ℕ) < 2 ∧
    covariance (fun z : Fin 2 → ℝ => dotProduct ![1, 2] z ^ 2)
      (fun z => dotProduct ![3, 4] z ^ 2) (gaussianRowLaw 2) = 242 ∧
    covariance (fun z : Fin 2 → ℝ => dotProduct ![1, 2] z * dotProduct ![3, 4] z)
      (fun z => dotProduct ![1, 2] z ^ 2) (gaussianRowLaw 2) = 110 := by
  refine ⟨by decide, ?_, ?_⟩
  · rw [covariance_gaussianRow_linear_squares]; norm_num [dotProduct, Fin.sum_univ_two]
  · rw [covariance_gaussianRow_linear_product_square]; norm_num [dotProduct, Fin.sum_univ_two]

end Transformer.Homogenized
