/-
# Exact fourth moments of the original Gaussian rows

ArXiv:2604.01978v1, §2.3.3, initialization (G), and §7.3,
`lem:delta_method`. The source computes the covariance of squared
query norms for a product of two independent Gaussian matrices.
These row moments retain both original factors and provide the
conditional fourth moments needed to check that covariance.

The scalar fourth moment follows from Gaussian integration by parts
with the cubic test and its actual derivative. Distinct coordinates
are independent under the original finite product law. Expanding the
square of the squared row norm therefore gives `d * (d + 2)`.
The conditional linear-form moment uses its proved Gaussian law;
the bilinear product of two random rows is not declared Gaussian.

All polynomial terms are proved integrable before taking sums of
their integrals. Empty dimensions and zero coefficient rows are
included in the identities without any nondegeneracy assumption.
-/

import Transformer.Homogenized.Section7_GaussianRowMoments
import Transformer.Homogenized.Section7_GaussianPolynomialStein

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Every scalar Gaussian power is integrable, including variance zero.
Source: arXiv:2604.01978v1, §7.3, moments in `lem:delta_method`. -/
theorem integrable_gaussianReal_pow (v : ℝ≥0) (n : ℕ) :
    Integrable (fun x : ℝ => x ^ n) (gaussianReal 0 v) := by
  exact integrable_pow_of_mem_interior_integrableExpSet
    (X := id) (by simp) n

/-- The exact centered scalar fourth moment, obtained from the cubic
Stein test. Source: arXiv:2604.01978v1, §7.3, Gaussian query moments. -/
theorem integral_gaussianReal_fourth (v : ℝ≥0) :
    (∫ x : ℝ, x ^ 4 ∂gaussianReal 0 v) = 3 * (v : ℝ) ^ 2 := by
  have hb (x : ℝ) : |x| ^ 2 ≤ 1 + |x| ^ 3 := by
    by_cases h : |x| ≤ 1
    · nlinarith [abs_nonneg x, pow_nonneg (abs_nonneg x) 3]
    · have h' : 1 ≤ |x| := le_of_not_ge h
      nlinarith [sq_nonneg (|x| - 1), abs_nonneg x]
  have h := centered_gaussian_stein_polynomial v (fun x : ℝ => x ^ 3)
    (fun x : ℝ => 3 * x ^ 2) (by fun_prop) (by fun_prop)
    (fun x => by simpa using hasDerivAt_pow 3 x) 1 3 3
    (fun x => by simp only [norm_pow, Real.norm_eq_abs, one_mul]; linarith)
    (fun x => by simpa only [norm_mul, Real.norm_eq_abs, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 3), abs_pow] using
      mul_le_mul_of_nonneg_left (hb x) (by norm_num : (0 : ℝ) ≤ 3))
  have hs : (∫ x : ℝ, x ^ 2 ∂gaussianReal 0 v) = v :=
    integral_sq_of_map_eq_gaussianReal measurable_id Measure.map_id
  have he : (fun x : ℝ => x * x ^ 3) = fun x => x ^ 4 := by
    funext x; ring
  rw [he, integral_const_mul, hs] at h
  nlinarith [h]

/-- The actual original row coordinates have all finite moments.
Source: arXiv:2604.01978v1, initialization (G) and §7.3. -/
theorem integrable_gaussianRow_coordinate_pow {d : ℕ} (i : Fin d) (n : ℕ) :
    Integrable (fun z : Fin d → ℝ => z i ^ n) (gaussianRowLaw d) := by
  have hl : HasLaw (fun z : Fin d → ℝ => z i) (gaussianReal 0 1) (gaussianRowLaw d) :=
    ⟨(measurable_pi_apply i).aemeasurable, gaussianRowLaw_map_coordinate i⟩
  exact hl.integrable_comp (integrable_gaussianReal_pow 1 n)

/-- The actual row-coordinate fourth moment is three.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianRow_coordinate_fourth {d : ℕ} (i : Fin d) :
    (∫ z : Fin d → ℝ, z i ^ 4 ∂gaussianRowLaw d) = 3 := by
  have hl : HasLaw (fun z : Fin d → ℝ => z i) (gaussianReal 0 1) (gaussianRowLaw d) :=
    ⟨(measurable_pi_apply i).aemeasurable, gaussianRowLaw_map_coordinate i⟩
  have h := hl.integral_comp (f := fun x : ℝ => x ^ 4) (by fun_prop)
  change (∫ z : Fin d → ℝ, z i ^ 4 ∂gaussianRowLaw d) = _ at h
  rw [h, integral_gaussianReal_fourth]
  norm_num

/-- Products of actual coordinate squares are integrable before
expanding the squared norm. Source: arXiv:2604.01978v1, §7.3. -/
theorem integrable_gaussianRow_square_product {d : ℕ} (i j : Fin d) :
    Integrable (fun z : Fin d → ℝ => z i ^ 2 * z j ^ 2) (gaussianRowLaw d) := by
  apply ((integrable_gaussianRow_coordinate_pow i 4).add
    (integrable_gaussianRow_coordinate_pow j 4)).mono' (by fun_prop)
  filter_upwards with z
  rw [Real.norm_of_nonneg (by positivity)]
  change z i ^ 2 * z j ^ 2 ≤ z i ^ 4 + z j ^ 4
  nlinarith [sq_nonneg (z i ^ 2 - z j ^ 2)]

/-- Distinct original coordinate squares have expectation one; the
diagonal has expectation three. Source: arXiv:2604.01978v1, §7.3. -/
theorem integral_gaussianRow_square_product {d : ℕ} (i j : Fin d) :
    (∫ z : Fin d → ℝ, z i ^ 2 * z j ^ 2 ∂gaussianRowLaw d) =
      if i = j then 3 else 1 := by
  by_cases hij : i = j
  · subst j
    simpa only [← pow_add, ite_true] using integral_gaussianRow_coordinate_fourth i
  · have hind : iIndepFun (fun i : Fin d => fun z : Fin d → ℝ => z i) (gaussianRowLaw d) :=
      iIndepFun_pi (X := fun _ : Fin d => id) (fun _ => aemeasurable_id)
    have h := (hind.indepFun hij).comp (by fun_prop : Measurable (fun x : ℝ => x ^ 2))
      (by fun_prop : Measurable (fun x : ℝ => x ^ 2))
    simp only [Function.comp_def] at h
    have he := h.integral_mul_eq_mul_integral
      (integrable_gaussianRow_coordinate_pow i 2).aestronglyMeasurable
      (integrable_gaussianRow_coordinate_pow j 2).aestronglyMeasurable
    change (∫ z : Fin d → ℝ, z i ^ 2 * z j ^ 2 ∂gaussianRowLaw d) =
      (∫ z : Fin d → ℝ, z i ^ 2 ∂gaussianRowLaw d) *
        (∫ z : Fin d → ℝ, z j ^ 2 ∂gaussianRowLaw d) at he
    rw [integral_sq_of_map_eq_gaussianReal (measurable_pi_apply i) (gaussianRowLaw_map_coordinate i),
      integral_sq_of_map_eq_gaussianReal (measurable_pi_apply j) (gaussianRowLaw_map_coordinate j)] at he
    simpa only [ite_eq_right hij, NNReal.coe_one, mul_one] using he

/-- The squared actual row norm has an integrable square.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integrable_gaussianRow_norm_fourth (d : ℕ) :
    Integrable (fun z : Fin d → ℝ => (∑ i : Fin d, z i ^ 2) ^ 2) (gaussianRowLaw d) := by
  simp_rw [pow_two, Finset.sum_mul_sum]
  exact integrable_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj =>
    by simpa only [pow_two] using integrable_gaussianRow_square_product i j

/-- The actual standard-row norm has fourth moment `d * (d + 2)`.
Source: arXiv:2604.01978v1, §7.3, the covariance calculation in
`lem:delta_method`, before multiplying the two Gaussian factors. -/
theorem integral_gaussianRow_norm_fourth (d : ℕ) :
    (∫ z : Fin d → ℝ, (∑ i : Fin d, z i ^ 2) ^ 2 ∂gaussianRowLaw d) =
      (d : ℝ) * (d + 2) := by
  have he (z : Fin d → ℝ) : (∑ i : Fin d, z i ^ 2) ^ 2 =
      ∑ i : Fin d, ∑ j : Fin d, z i ^ 2 * z j ^ 2 := by
    rw [pow_two, Finset.sum_mul_sum]
  simp_rw [he]
  rw [integral_finsetSum _ fun i hi => integrable_finsetSum _ fun j hj =>
    integrable_gaussianRow_square_product i j]
  simp_rw [integral_finsetSum _ fun j hj => integrable_gaussianRow_square_product _ j,
    integral_gaussianRow_square_product]
  have hif (i j : Fin d) : (if i = j then (3 : ℝ) else 1) = 1 + if i = j then 2 else 0 := by
    split_ifs <;> norm_num
  simp_rw [hif]
  simp [Finset.sum_add_distrib, mul_add]

/-- The exact conditional fourth moment of one Gaussian linear row.
Source: arXiv:2604.01978v1, §7.3, with the first original factor
held fixed instead of declaring the product matrix Gaussian. -/
theorem integral_gaussianRow_linear_fourth {d : ℕ} (g : Fin d → ℝ) :
    (∫ z : Fin d → ℝ, dotProduct g z ^ 4 ∂gaussianRowLaw d) =
      3 * (∑ i : Fin d, g i ^ 2) ^ 2 := by
  have h := (gaussianRow_linear_hasLaw g).integral_comp
    (f := fun x : ℝ => x ^ 4) (by fun_prop)
  rw [integral_gaussianReal_fourth, Real.coe_toNNReal _ (Finset.sum_nonneg fun i hi => sq_nonneg _)] at h
  exact h

/-- In dimension two the original row norm has fourth moment eight,
and the actual linear form with coefficients `(1,2)` has moment 75. -/
example : (∫ z : Fin 2 → ℝ, (∑ i : Fin 2, z i ^ 2) ^ 2 ∂gaussianRowLaw 2) = 8 ∧
    (∫ z : Fin 2 → ℝ, dotProduct ![1, 2] z ^ 4 ∂gaussianRowLaw 2) = 75 := by
  constructor
  · rw [integral_gaussianRow_norm_fourth]; norm_num
  · rw [integral_gaussianRow_linear_fourth]
    norm_num [Fin.sum_univ_two]

end Transformer.Homogenized
