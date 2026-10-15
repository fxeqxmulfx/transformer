/-
# Cubic norm-deviation moments for arbitrary actual Gaussian queries

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the third-order
Taylor remainder. The previously derived unit-query norm moment is
extended to every deterministic original input by its exact scaling.
The center remains the actual squared input norm.

For a query `x`, its squared-norm deviation equals `‖x‖²` times
the deviation for its normalized input. Consequently its third
absolute moment is bounded by a common constant times
`‖x‖⁶ d^(-3/2)`, including the zero query. The constant is the same
one already obtained from the original Gaussian row law.

Elementary cubic inequalities also control two and three nonnegative
deviations, and the absolute difference of two scalar deviations.
These apply directly to the genuine score polarization and the sum
of absolute deviations in the global correlation Taylor remainder.
No Gaussian law is attributed to the product of two matrix factors.
-/

import Transformer.Homogenized.Section7_GaussianQueryCovariance

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The original squared query deviation has its exact homogeneous
scaling, even for a zero input. Source: arXiv:2604.01978v1,
§7.3, the squared-norm variables of `lem:delta_method`. -/
theorem gaussianMatrix_query_deviation_normalize {d : ℕ}
    (M : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    ‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2 =
      ‖x‖ ^ 2 * (‖Matrix.toEuclideanLin M (normalizeLayer x)‖ ^ 2 / (d : ℝ) - 1) := by
  by_cases hx : x = 0
  · subst x
    simp [normalizeLayer]
  have hnx : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hvec : x = ‖x‖ • normalizeLayer x := by
    rw [normalizeLayer, smul_smul, mul_inv_cancel₀ hnx, one_smul]
  have hquery : Matrix.toEuclideanLin M x =
      ‖x‖ • Matrix.toEuclideanLin M (normalizeLayer x) := by
    calc
      _ = Matrix.toEuclideanLin M (‖x‖ • normalizeLayer x) := congrArg _ hvec
      _ = _ := (Matrix.toEuclideanLin M).map_smul _ _
  rw [hquery, norm_smul_of_nonneg (norm_nonneg x)]
  ring

/-- The third absolute norm-deviation scales by the sixth power
of the original input norm. Source: arXiv:2604.01978v1,
§7.3, the third moment used in `lem:delta_method`. -/
theorem gaussianMatrix_query_deviation_abs_cube_normalize {d : ℕ}
    (M : Matrix (Fin d) (Fin d) ℝ) (x : EucSpace d) :
    |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2| ^ 3 =
      ‖x‖ ^ 6 * |‖Matrix.toEuclideanLin M (normalizeLayer x)‖ ^ 2 / (d : ℝ) - 1| ^ 3 := by
  rw [gaussianMatrix_query_deviation_normalize, abs_mul, abs_of_nonneg (sq_nonneg _)]
  ring

/-- **Uniform actual cubic moment for every original query.**
The exact input-norm factor is retained, and the zero query is included.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the norm component of the third-order remainder for one original factor. -/
theorem gaussianMatrix_query_deviation_third_uniform :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x : EucSpace d,
      Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
        |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2| ^ 3) (gaussianMatrixLaw d) ∧
      (∫ M : Matrix (Fin d) (Fin d) ℝ,
        |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - ‖x‖ ^ 2| ^ 3 ∂gaussianMatrixLaw d) ≤
          C * ‖x‖ ^ 6 * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, h⟩ := gaussianMatrix_unit_query_deviation_third_uniform
  refine ⟨C, hC, ?_⟩
  intro d hd x
  by_cases hx : x = 0
  · subst x
    simp
  obtain ⟨hi, hb⟩ := h d hd (normalizeLayer x) (norm_normalizeLayer hx)
  constructor
  · simpa only [gaussianMatrix_query_deviation_abs_cube_normalize] using hi.const_mul (‖x‖ ^ 6)
  · calc
      _ = ‖x‖ ^ 6 * ∫ M : Matrix (Fin d) (Fin d) ℝ,
          |‖Matrix.toEuclideanLin M (normalizeLayer x)‖ ^ 2 / (d : ℝ) - 1| ^ 3 ∂gaussianMatrixLaw d := by
        simp_rw [gaussianMatrix_query_deviation_abs_cube_normalize]
        exact integral_const_mul _ _
      _ ≤ ‖x‖ ^ 6 * (C * (d : ℝ) ^ (-(3 : ℝ) / 2)) :=
        mul_le_mul_of_nonneg_left hb (by positivity)
      _ = _ := by ring

/-- Two nonnegative cubic deviations control their sum with an
explicit dimension-independent constant. Source: arXiv:2604.01978v1,
§7.3, the cubic remainder in `lem:delta_method`. -/
theorem delta_cube_add_le (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (a + b) ^ 3 ≤ 4 * (a ^ 3 + b ^ 3) := by
  nlinarith [mul_nonneg (sq_nonneg (a - b)) (add_nonneg ha hb)]

/-- Three actual nonnegative deviations have a common cubic
majorant. Source: arXiv:2604.01978v1, §7.3,
the sum of score and norm errors in `lem:delta_method`. -/
theorem delta_cube_add_three_le (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
    (a + b + c) ^ 3 ≤ 16 * (a ^ 3 + b ^ 3 + c ^ 3) := by
  have hab := delta_cube_add_le a b ha hb
  have habc := delta_cube_add_le (a + b) c (add_nonneg ha hb) hc
  have hc3 : 0 ≤ c ^ 3 := by positivity
  nlinarith

/-- The difference used by actual score polarization has its
third absolute power controlled by the two norm deviations.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`, cubic moments. -/
theorem delta_abs_sub_cube_le (a b : ℝ) :
    |a - b| ^ 3 ≤ 4 * (|a| ^ 3 + |b| ^ 3) := by
  have hab : |a - b| ≤ |a| + |b| := by simpa using abs_sub_le a 0 b
  exact (pow_le_pow_left₀ (abs_nonneg _) hab 3).trans
    (delta_cube_add_le |a| |b| (abs_nonneg _) (abs_nonneg _))

/-- The actual sum of the three absolute Taylor deviations is
controlled directly by their separate third absolute powers.
All scalar inputs, including zero and negative deviations, are allowed.
Together with integrability of the three separate moments, this gives
an integrable majorant for the full global remainder polynomial.
The constant is independent of the original queries and dimension.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the third absolute moment in the Taylor remainder. -/
theorem delta_abs_cube_sum_le (w u v : ℝ) :
    (|w| + |u| + |v|) ^ 3 ≤ 16 * (|w| ^ 3 + |u| ^ 3 + |v| ^ 3) := by
  exact delta_cube_add_three_le |w| |u| |v|
    (abs_nonneg w) (abs_nonneg u) (abs_nonneg v)

/-- The two polarized original inputs have norms at most two
for every original unit pair. Source: arXiv:2604.01978v1,
§7.3, the original unit inputs of `lem:delta_method`. -/
theorem norm_gaussian_polarized_unit_queries_le {d : ℕ} (x y : EucSpace d)
    (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) : ‖x + y‖ ≤ 2 ∧ ‖x - y‖ ≤ 2 := by
  constructor
  · have h := norm_add_le x y
    rw [hx, hy] at h
    linarith
  · have h := norm_sub_le x y
    rw [hx, hy] at h
    linarith

/-- Nonzero original queries and genuine unit pairs jointly
satisfy the scaling and polarized-query hypotheses. The three
nonnegative numbers also satisfy both cubic-sum hypotheses. -/
example : (0 : ℕ) < 2 ∧ (EuclideanSpace.single (0 : Fin 2) (1 : ℝ)) ≠ 0 ∧
    ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 3 := by
  refine ⟨by decide, ?_, by simp, by simp, by norm_num, by norm_num, by norm_num⟩
  intro h
  have hn : ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 := by simp
  rw [h] at hn
  norm_num at hn

end Transformer.Homogenized
