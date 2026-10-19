module

public import Transformer.Homogenized.Section7_GaussianRowFourth

/-!
# Mixed fourth moments of the actual original Gaussian row

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the mixed entries
of the covariance used to integrate the quadratic correlation polynomial.
The actual scalar linear forms in one original standard Gaussian row
have their already proved Gaussian laws and fourth moments.

Polynomial polarization gives the mixed square–square and cube–linear
moments directly from fourth moments of actual linear combinations.
Their exact values retain the actual coefficient norms and inner product;
neither a covariance identity nor a mixed moment is assumed.

The products are integrable by domination by the two fourth powers.
Every fourth power and every sum used in the polarization argument is
proved integrable before its integral is distributed. The identities
include zero coefficient vectors and empty dimensions, and will be
summed over the independent original matrix rows in the query calculation.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Every power of an actual original Gaussian row linear form is
integrable. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_gaussianRow_linear_pow {d : ℕ} (g : Fin d → ℝ) (n : ℕ) :
    Integrable (fun z : Fin d → ℝ => dotProduct g z ^ n) (gaussianRowLaw d) :=
  (gaussianRow_linear_hasLaw g).integrable_comp (integrable_gaussianReal_pow _ n)

/-- The actual product of two row linear forms is integrable.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`, one original factor. -/
theorem integrable_gaussianRow_linear_product {d : ℕ} (g h : Fin d → ℝ) :
    Integrable (fun z : Fin d → ℝ => dotProduct g z * dotProduct h z) (gaussianRowLaw d) := by
  simpa only [dotProduct, id_eq] using integrable_gaussian_coordinate_linear_product (P := gaussianRowLaw d)
    (fun (i : Fin d) (z : Fin d → ℝ) => z i) (fun _ => (1 : ℝ≥0)) measurable_pi_apply
    (fun i => by simpa only [one_pow] using gaussianRowLaw_map_coordinate i) g h id id

/-- The true second mixed moment is the original coefficient inner
product. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianRow_linear_product {d : ℕ} (g h : Fin d → ℝ) :
    (∫ z : Fin d → ℝ, dotProduct g z * dotProduct h z ∂gaussianRowLaw d) = dotProduct g h := by
  have hprod := integral_gaussian_coordinate_linear_product (P := gaussianRowLaw d)
    (fun (i : Fin d) (z : Fin d → ℝ) => z i) (fun _ => (1 : ℝ≥0)) measurable_pi_apply
    (iIndepFun_pi (X := fun _ : Fin d => id) (fun _ => aemeasurable_id))
    (fun i => by simpa only [one_pow] using gaussianRowLaw_map_coordinate i) g h id id
  simpa [dotProduct] using hprod

/-- Actual linear combinations have all the required finite moments.
Source: arXiv:2604.01978v1, §7.3, polarization for `lem:delta_method`. -/
theorem integrable_gaussianRow_linear_combination_pow {d : ℕ}
    (g h : Fin d → ℝ) (a b : ℝ) (n : ℕ) :
    Integrable (fun z : Fin d → ℝ => (a * dotProduct g z + b * dotProduct h z) ^ n)
      (gaussianRowLaw d) := by
  simpa only [add_dotProduct, smul_dotProduct, smul_eq_mul] using
    integrable_gaussianRow_linear_pow (a • g + b • h) n

/-- The exact fourth moment of every actual row linear combination.
Source: arXiv:2604.01978v1, §7.3, the covariance calculation in
`lem:delta_method`, derived here before mixing the two original factors. -/
theorem integral_gaussianRow_linear_combination_fourth {d : ℕ}
    (g h : Fin d → ℝ) (a b : ℝ) :
    (∫ z : Fin d → ℝ, (a * dotProduct g z + b * dotProduct h z) ^ 4 ∂gaussianRowLaw d) =
      3 * (a ^ 2 * (∑ i : Fin d, g i ^ 2) + 2 * a * b * dotProduct g h +
        b ^ 2 * (∑ i : Fin d, h i ^ 2)) ^ 2 := by
  have hk : (∑ i : Fin d, (a • g + b • h) i ^ 2) =
      a ^ 2 * (∑ i : Fin d, g i ^ 2) + 2 * a * b * dotProduct g h + b ^ 2 * (∑ i : Fin d, h i ^ 2) := by
    calc
      _ = dotProduct (a • g + b • h) (a • g + b • h) := by simp only [dotProduct, pow_two]
      _ = _ := by
        simp only [add_dotProduct, dotProduct_add, smul_dotProduct, dotProduct_smul, smul_eq_mul]
        rw [dotProduct_comm h g]
        simp only [dotProduct, pow_two]
        ring
  have hfourth := integral_gaussianRow_linear_fourth (a • g + b • h)
  rw [hk] at hfourth
  simpa only [add_dotProduct, smul_dotProduct, smul_eq_mul] using hfourth

/-- Actual products of two squared row linear forms are integrable.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_gaussianRow_linear_square_product {d : ℕ} (g h : Fin d → ℝ) :
    Integrable (fun z : Fin d → ℝ => dotProduct g z ^ 2 * dotProduct h z ^ 2) (gaussianRowLaw d) := by
  apply ((integrable_gaussianRow_linear_pow g 4).add
    (integrable_gaussianRow_linear_pow h 4)).mono' (by fun_prop)
  filter_upwards with z
  rw [Real.norm_of_nonneg (by positivity)]
  change dotProduct g z ^ 2 * dotProduct h z ^ 2 ≤ dotProduct g z ^ 4 + dotProduct h z ^ 4
  nlinarith [sq_nonneg (dotProduct g z ^ 2 - dotProduct h z ^ 2)]

/-- Actual cube–linear row products are integrable by their original
fourth powers. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_gaussianRow_linear_cube_product {d : ℕ} (g h : Fin d → ℝ) :
    Integrable (fun z : Fin d → ℝ => dotProduct g z ^ 3 * dotProduct h z) (gaussianRowLaw d) := by
  apply ((integrable_gaussianRow_linear_pow g 4).add
    (integrable_gaussianRow_linear_pow h 4)).mono' (by fun_prop)
  filter_upwards with z
  let a := dotProduct g z
  let b := dotProduct h z
  have hab : 2 * |a| * |b| ≤ a ^ 2 + b ^ 2 := by nlinarith [sq_nonneg (|a| - |b|), sq_abs a, sq_abs b]
  have hm := mul_le_mul_of_nonneg_left hab (sq_nonneg a)
  have hc : 2 * a ^ 2 * b ^ 2 ≤ a ^ 4 + b ^ 4 := by nlinarith [sq_nonneg (a ^ 2 - b ^ 2)]
  change ‖a ^ 3 * b‖ ≤ a ^ 4 + b ^ 4
  rw [Real.norm_eq_abs, abs_mul, abs_pow, pow_succ, sq_abs]
  nlinarith [sq_nonneg (a ^ 2), sq_nonneg (b ^ 2)]

/-- **Actual square–square Gaussian mixed moment.** Its two terms
are the product of the true variances and twice the true squared
coefficient inner product. Source: arXiv:2604.01978v1, §7.3,
the covariance input in `lem:delta_method`. -/
theorem integral_gaussianRow_linear_square_product {d : ℕ} (g h : Fin d → ℝ) :
    (∫ z : Fin d → ℝ, dotProduct g z ^ 2 * dotProduct h z ^ 2 ∂gaussianRowLaw d) =
      (∑ i : Fin d, g i ^ 2) * (∑ i : Fin d, h i ^ 2) + 2 * dotProduct g h ^ 2 := by
  have ip := integrable_gaussianRow_linear_combination_pow g h 1 1 4
  have im := integrable_gaussianRow_linear_combination_pow g h 1 (-1) 4
  have ig := integrable_gaussianRow_linear_pow g 4
  have ih := integrable_gaussianRow_linear_pow h 4
  have ia : Integrable (fun z : Fin d → ℝ =>
      (1 * dotProduct g z + 1 * dotProduct h z) ^ 4 + (1 * dotProduct g z + (-1) * dotProduct h z) ^ 4)
      (gaussianRowLaw d) := by convert ip.add im using 1
  have isub : Integrable (fun z : Fin d → ℝ =>
      (1 * dotProduct g z + 1 * dotProduct h z) ^ 4 + (1 * dotProduct g z + (-1) * dotProduct h z) ^ 4 -
        2 * dotProduct g z ^ 4) (gaussianRowLaw d) := by convert ia.sub (ig.const_mul 2) using 1
  have hp (z : Fin d → ℝ) : 12 * (dotProduct g z ^ 2 * dotProduct h z ^ 2) =
      (1 * dotProduct g z + 1 * dotProduct h z) ^ 4 + (1 * dotProduct g z + (-1) * dotProduct h z) ^ 4 -
        2 * dotProduct g z ^ 4 - 2 * dotProduct h z ^ 4 := by ring
  have hi : 12 * (∫ z : Fin d → ℝ, dotProduct g z ^ 2 * dotProduct h z ^ 2 ∂gaussianRowLaw d) =
      (∫ z : Fin d → ℝ, (1 * dotProduct g z + 1 * dotProduct h z) ^ 4 ∂gaussianRowLaw d) +
        (∫ z : Fin d → ℝ, (1 * dotProduct g z + (-1) * dotProduct h z) ^ 4 ∂gaussianRowLaw d) -
          2 * (∫ z : Fin d → ℝ, dotProduct g z ^ 4 ∂gaussianRowLaw d) -
            2 * (∫ z : Fin d → ℝ, dotProduct h z ^ 4 ∂gaussianRowLaw d) := by
    rw [← integral_const_mul]
    simp_rw [hp]
    rw [integral_sub isub (ih.const_mul 2), integral_sub ia (ig.const_mul 2),
      integral_add ip im, integral_const_mul, integral_const_mul]
  rw [integral_gaussianRow_linear_combination_fourth, integral_gaussianRow_linear_combination_fourth,
    integral_gaussianRow_linear_fourth, integral_gaussianRow_linear_fourth] at hi
  nlinarith [hi]

/-- **Actual cube–linear Gaussian mixed moment.** The variance and
coefficient inner product are computed from the original row law.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianRow_linear_cube_product {d : ℕ} (g h : Fin d → ℝ) :
    (∫ z : Fin d → ℝ, dotProduct g z ^ 3 * dotProduct h z ∂gaussianRowLaw d) =
      3 * (∑ i : Fin d, g i ^ 2) * dotProduct g h := by
  have ip := integrable_gaussianRow_linear_combination_pow g h 2 1 4
  have im := integrable_gaussianRow_linear_combination_pow g h 2 (-1) 4
  have jp := integrable_gaussianRow_linear_combination_pow g h 1 1 4
  have jm := integrable_gaussianRow_linear_combination_pow g h 1 (-1) 4
  have isub : Integrable (fun z : Fin d → ℝ =>
      (2 * dotProduct g z + 1 * dotProduct h z) ^ 4 - (2 * dotProduct g z + (-1) * dotProduct h z) ^ 4)
      (gaussianRowLaw d) := by convert ip.sub im using 1
  have js : Integrable (fun z : Fin d → ℝ =>
      (2 * dotProduct g z + 1 * dotProduct h z) ^ 4 - (2 * dotProduct g z + (-1) * dotProduct h z) ^ 4 -
        2 * (1 * dotProduct g z + 1 * dotProduct h z) ^ 4) (gaussianRowLaw d) := by
    convert isub.sub (jp.const_mul 2) using 1
  have hp (z : Fin d → ℝ) : 48 * (dotProduct g z ^ 3 * dotProduct h z) =
      (2 * dotProduct g z + 1 * dotProduct h z) ^ 4 - (2 * dotProduct g z + (-1) * dotProduct h z) ^ 4 -
        2 * (1 * dotProduct g z + 1 * dotProduct h z) ^ 4 +
          2 * (1 * dotProduct g z + (-1) * dotProduct h z) ^ 4 := by ring
  have hi : 48 * (∫ z : Fin d → ℝ, dotProduct g z ^ 3 * dotProduct h z ∂gaussianRowLaw d) =
      (∫ z : Fin d → ℝ, (2 * dotProduct g z + 1 * dotProduct h z) ^ 4 ∂gaussianRowLaw d) -
        (∫ z : Fin d → ℝ, (2 * dotProduct g z + (-1) * dotProduct h z) ^ 4 ∂gaussianRowLaw d) -
          2 * (∫ z : Fin d → ℝ, (1 * dotProduct g z + 1 * dotProduct h z) ^ 4 ∂gaussianRowLaw d) +
            2 * (∫ z : Fin d → ℝ, (1 * dotProduct g z + (-1) * dotProduct h z) ^ 4 ∂gaussianRowLaw d) := by
    rw [← integral_const_mul]
    simp_rw [hp]
    rw [integral_add js (jm.const_mul 2), integral_sub isub (jp.const_mul 2),
      integral_sub ip im, integral_const_mul, integral_const_mul]
  rw [integral_gaussianRow_linear_combination_fourth, integral_gaussianRow_linear_combination_fourth,
    integral_gaussianRow_linear_combination_fourth, integral_gaussianRow_linear_combination_fourth] at hi
  nlinarith [hi]

/-- Distinct nonzero coefficient rows in dimension two have genuine
mixed square–square moment 367 and cube–linear moment 165. -/
example : (∫ z : Fin 2 → ℝ, dotProduct ![1, 2] z ^ 2 * dotProduct ![3, 4] z ^ 2 ∂gaussianRowLaw 2) = 367 ∧
    (∫ z : Fin 2 → ℝ, dotProduct ![1, 2] z ^ 3 * dotProduct ![3, 4] z ∂gaussianRowLaw 2) = 165 := by
  constructor
  · rw [integral_gaussianRow_linear_square_product]; norm_num [dotProduct, Fin.sum_univ_two]
  · rw [integral_gaussianRow_linear_cube_product]; norm_num [dotProduct, Fin.sum_univ_two]

end Transformer.Homogenized
