/-
# Actual Gaussian score and full Taylor-deviation cubic moments

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the third-order
remainder for normalized correlation. Polarization expresses the
original centered score as one quarter of the difference of the
centered squared-norm queries at the original inputs `x+y` and `x-y`.

The genuine cubic norm estimates at these two inputs give an actual
third absolute score moment, uniformly over every unit input pair
and positive dimension. No independence of the two polarized queries
is needed. Their input norms are both bounded by two.

Finally, the cubic moment of the sum of the three absolute deviations
in the genuine global Taylor error is bounded at the same uniform
rate `d^(-3/2)`. Integrability follows from the explicit moment majorant.
All statements concern one original Gaussian matrix factor. They
provide the actual remainder input needed before composing two factors.
-/

import Transformer.Homogenized.Section7_GaussianQueryThird

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- Actual score polarization with the original squared-norm centers.
Source: arXiv:2604.01978v1, §7.3, the centered variables
in `lem:delta_method`, applied to one original factor. -/
theorem gaussianMatrix_query_score_deviation_polarization {d : ℕ}
    (M : Matrix (Fin d) (Fin d) ℝ) (x y : EucSpace d) :
    inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y =
      ((‖Matrix.toEuclideanLin M (x + y)‖ ^ 2 / (d : ℝ) - ‖x + y‖ ^ 2) -
        (‖Matrix.toEuclideanLin M (x - y)‖ ^ 2 / (d : ℝ) - ‖x - y‖ ^ 2)) / 4 := by
  rw [(Matrix.toEuclideanLin M).map_add, (Matrix.toEuclideanLin M).map_sub]
  simp only [norm_add_sq_real, norm_sub_sq_real]
  ring

/-- The actual score's cubic absolute deviation is bounded by the
two polarized norm deviations. Source: arXiv:2604.01978v1,
§7.3, the third-order remainder in `lem:delta_method`. -/
theorem gaussianMatrix_query_score_abs_cube_le {d : ℕ}
    (M : Matrix (Fin d) (Fin d) ℝ) (x y : EucSpace d) :
    |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| ^ 3 ≤
      (|‖Matrix.toEuclideanLin M (x + y)‖ ^ 2 / (d : ℝ) - ‖x + y‖ ^ 2| ^ 3 +
        |‖Matrix.toEuclideanLin M (x - y)‖ ^ 2 / (d : ℝ) - ‖x - y‖ ^ 2| ^ 3) / 16 := by
  rw [gaussianMatrix_query_score_deviation_polarization, abs_div, div_pow]
  norm_num only
  have h := delta_abs_sub_cube_le
    (‖Matrix.toEuclideanLin M (x + y)‖ ^ 2 / (d : ℝ) - ‖x + y‖ ^ 2)
    (‖Matrix.toEuclideanLin M (x - y)‖ ^ 2 / (d : ℝ) - ‖x - y‖ ^ 2)
  linarith

/-- **Uniform third absolute moment of the actual original score.**
Every unit input pair uses the same positive constant.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the score component of the single-factor Taylor remainder. -/
theorem gaussianMatrix_unit_score_deviation_third_uniform :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
      Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
        |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| ^ 3)
          (gaussianMatrixLaw d) ∧
      (∫ M : Matrix (Fin d) (Fin d) ℝ,
        |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| ^ 3
          ∂gaussianMatrixLaw d) ≤ C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, h⟩ := gaussianMatrix_query_deviation_third_uniform
  refine ⟨8 * C, by positivity, ?_⟩
  intro d hd x y hx hy
  obtain ⟨ip, hp⟩ := h d hd (x + y)
  obtain ⟨im, hm⟩ := h d hd (x - y)
  have isum : Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |‖Matrix.toEuclideanLin M (x + y)‖ ^ 2 / (d : ℝ) - ‖x + y‖ ^ 2| ^ 3 +
        |‖Matrix.toEuclideanLin M (x - y)‖ ^ 2 / (d : ℝ) - ‖x - y‖ ^ 2| ^ 3) (gaussianMatrixLaw d) := by
    convert ip.add im using 1
  have hmeas : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| ^ 3) := by
    have hmx := measurable_gaussianMatrix_query x
    have hmy := measurable_gaussianMatrix_query y
    fun_prop
  have iscore := (isum.div_const 16).mono' hmeas.aestronglyMeasurable
    (ae_of_all _ fun M => by
      rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
      exact gaussianMatrix_query_score_abs_cube_le M x y)
  refine ⟨iscore, ?_⟩
  have hnorms := norm_gaussian_polarized_unit_queries_le x y hx hy
  have hp6 : ‖x + y‖ ^ 6 ≤ (64 : ℝ) := by
    convert pow_le_pow_left₀ (norm_nonneg (x + y)) hnorms.1 6 using 1
    norm_num
  have hm6 : ‖x - y‖ ^ 6 ≤ (64 : ℝ) := by
    convert pow_le_pow_left₀ (norm_nonneg (x - y)) hnorms.2 6 using 1
    norm_num
  have hD : 0 ≤ (d : ℝ) ^ (-(3 : ℝ) / 2) := Real.rpow_nonneg (Nat.cast_nonneg d) _
  have hpbound := hp.trans (mul_le_mul_of_nonneg_right
    (mul_le_mul_of_nonneg_left hp6 hC.le) hD)
  have hmbound := hm.trans (mul_le_mul_of_nonneg_right
    (mul_le_mul_of_nonneg_left hm6 hC.le) hD)
  calc
    _ ≤ ∫ M : Matrix (Fin d) (Fin d) ℝ,
        (|‖Matrix.toEuclideanLin M (x + y)‖ ^ 2 / (d : ℝ) - ‖x + y‖ ^ 2| ^ 3 +
          |‖Matrix.toEuclideanLin M (x - y)‖ ^ 2 / (d : ℝ) - ‖x - y‖ ^ 2| ^ 3) / 16 ∂gaussianMatrixLaw d :=
      integral_mono iscore (isum.div_const 16) fun M => gaussianMatrix_query_score_abs_cube_le M x y
    _ = ((∫ M : Matrix (Fin d) (Fin d) ℝ,
        |‖Matrix.toEuclideanLin M (x + y)‖ ^ 2 / (d : ℝ) - ‖x + y‖ ^ 2| ^ 3 ∂gaussianMatrixLaw d) +
          ∫ M : Matrix (Fin d) (Fin d) ℝ,
            |‖Matrix.toEuclideanLin M (x - y)‖ ^ 2 / (d : ℝ) - ‖x - y‖ ^ 2| ^ 3 ∂gaussianMatrixLaw d) / 16 := by
      rw [integral_div, integral_add ip im]
    _ ≤ _ := by linarith

/-- **The full actual Taylor-deviation cubic moment is uniformly small.**
The genuine score and both original squared-norm errors are included.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
the actual third-order remainder for one original Gaussian factor. -/
theorem gaussianMatrix_unit_delta_deviation_third_uniform :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
      Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
        (|inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| +
          |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| +
            |‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1|) ^ 3) (gaussianMatrixLaw d) ∧
      (∫ M : Matrix (Fin d) (Fin d) ℝ,
        (|inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| +
          |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| +
            |‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1|) ^ 3 ∂gaussianMatrixLaw d) ≤
              C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨Cs, hCs, hs⟩ := gaussianMatrix_unit_score_deviation_third_uniform
  obtain ⟨Cn, hCn, hn⟩ := gaussianMatrix_unit_query_deviation_third_uniform
  refine ⟨16 * (Cs + 2 * Cn), by positivity, ?_⟩
  intro d hd x y hx hy
  obtain ⟨iw, hw⟩ := hs d hd x y hx hy
  obtain ⟨iu, hu⟩ := hn d hd x hx
  obtain ⟨iv, hv⟩ := hn d hd y hy
  have isum : Integrable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      |inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| ^ 3 +
        |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| ^ 3 +
          |‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1| ^ 3) (gaussianMatrixLaw d) := by
    convert (iw.add iu).add iv using 1
  have hmeas : Measurable (fun M : Matrix (Fin d) (Fin d) ℝ =>
      (|inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y| +
        |‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1| +
          |‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1|) ^ 3) := by
    have hmx := measurable_gaussianMatrix_query x
    have hmy := measurable_gaussianMatrix_query y
    fun_prop
  have ibound (M : Matrix (Fin d) (Fin d) ℝ) := delta_abs_cube_sum_le
    (inner ℝ (Matrix.toEuclideanLin M x) (Matrix.toEuclideanLin M y) / (d : ℝ) - inner ℝ x y)
    (‖Matrix.toEuclideanLin M x‖ ^ 2 / (d : ℝ) - 1) (‖Matrix.toEuclideanLin M y‖ ^ 2 / (d : ℝ) - 1)
  have itotal := (isum.const_mul 16).mono' hmeas.aestronglyMeasurable
    (ae_of_all _ fun M => by
      rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
      exact ibound M)
  refine ⟨itotal, ?_⟩
  have hb := integral_mono itotal (isum.const_mul 16) ibound
  rw [integral_const_mul, integral_add (by convert iw.add iu using 1) iv,
    integral_add iw iu] at hb
  linarith

/-- Positive dimension and two actual coordinate unit queries
jointly satisfy every original input hypothesis of the cubic bounds. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 := by
  exact ⟨by decide, by simp, by simp⟩

end Transformer.Homogenized
