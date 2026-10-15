/-
# Covariance of averages over the original independent Gaussian rows

ArXiv:2604.01978v1, §7.3, `lem:delta_method`, the covariance
of normalized query scores and squared norms. Each such statistic
is an average of actual polynomial functions of the original matrix rows.

Every original row has its prescribed Gaussian product law, and the
rows are jointly independent. Square-integrable row statistics therefore
have zero covariance across distinct rows and their original row-law
covariance on the diagonal. Summing gives the covariance of the actual
row sums, and dimension normalization gives the exact `1/d` factor.

The original centered-average product integral is identified with this
covariance using the proved original row means. All statistics remain
functions of the actual Gaussian matrix; no mixed moment or covariance
value is assumed. The next step substitutes the genuine row polynomials
whose Gaussian mixed moments have already been proved.
-/

import Transformer.Homogenized.Section7_GaussianMatrixLaw
import Transformer.Homogenized.Section7_GaussianMixedFourth
import Mathlib.Probability.Moments.Covariance

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- An actual original matrix row has its prescribed full row law.
Source: arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem gaussianMatrix_row_hasLaw {d : ℕ} (i : Fin d) :
    HasLaw (fun M : Matrix (Fin d) (Fin d) ℝ => M i) (gaussianRowLaw d) (gaussianMatrixLaw d) :=
  ⟨(Measurable.of_eval fun j => Matrix.measurable_apply (i := i) (j := j)).aemeasurable,
    gaussianMatrixLaw_map_row i⟩

/-- Square integrability transfers to each actual original row.
Source: arXiv:2604.01978v1, §7.3, the row statistics in `lem:delta_method`. -/
theorem memLp_gaussianMatrix_row_statistic {d : ℕ} (F : (Fin d → ℝ) → ℝ)
    (hF : MemLp F 2 (gaussianRowLaw d)) (i : Fin d) :
    MemLp (fun M : Matrix (Fin d) (Fin d) ℝ => F (M i)) 2 (gaussianMatrixLaw d) := by
  simpa only [Function.comp_def] using (gaussianMatrix_row_hasLaw i).memLp_comp hF

/-- The actual row average is square integrable, before taking any
centered products. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem memLp_gaussianMatrix_row_average {d : ℕ} (F : (Fin d → ℝ) → ℝ)
    (hF : MemLp F 2 (gaussianRowLaw d)) :
    MemLp (fun M : Matrix (Fin d) (Fin d) ℝ => (∑ i : Fin d, F (M i)) / (d : ℝ))
      2 (gaussianMatrixLaw d) := by
  have hs := memLp_finsetSum Finset.univ (fun (i : Fin d) hi => memLp_gaussianMatrix_row_statistic F hF i)
  have h := hs.const_mul (d : ℝ)⁻¹
  simpa only [div_eq_mul_inv, mul_comm] using h

/-- The actual average mean equals the original row mean in every
positive dimension. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_gaussianMatrix_row_average {d : ℕ} (hd : 0 < d)
    (F : (Fin d → ℝ) → ℝ) (hF : MemLp F 2 (gaussianRowLaw d)) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ, (∑ i : Fin d, F (M i)) / (d : ℝ) ∂gaussianMatrixLaw d) =
      ∫ z, F z ∂gaussianRowLaw d := by
  have hdR : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have he (i : Fin d) : (∫ M : Matrix (Fin d) (Fin d) ℝ, F (M i) ∂gaussianMatrixLaw d) =
      ∫ z, F z ∂gaussianRowLaw d := (gaussianMatrix_row_hasLaw i).integral_comp hF.aestronglyMeasurable
  rw [integral_div, integral_finsetSum _ fun i hi =>
    (memLp_gaussianMatrix_row_statistic F hF i).integrable (by norm_num)]
  simp only [he, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- Original row statistics have the true row covariance on the
diagonal and zero covariance on distinct independent rows.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem covariance_gaussianMatrix_row_statistics {d : ℕ} (F G : (Fin d → ℝ) → ℝ)
    (hF : MemLp F 2 (gaussianRowLaw d)) (hG : MemLp G 2 (gaussianRowLaw d)) (i j : Fin d) :
    covariance (fun M : Matrix (Fin d) (Fin d) ℝ => F (M i)) (fun M => G (M j))
      (gaussianMatrixLaw d) = if i = j then covariance F G (gaussianRowLaw d) else 0 := by
  by_cases hij : i = j
  · subst j
    simpa only [Function.comp_def, ite_true] using
      (gaussianMatrix_row_hasLaw i).covariance_comp hF.aemeasurable hG.aemeasurable
  · have hFm : AEMeasurable F ((gaussianMatrixLaw d).map (fun M => M i)) := by
      rw [gaussianMatrixLaw_map_row]; exact hF.aemeasurable
    have hGm : AEMeasurable G ((gaussianMatrixLaw d).map (fun M => M j)) := by
      rw [gaussianMatrixLaw_map_row]; exact hG.aemeasurable
    have h := ((iIndepFun_gaussianMatrix_rows d).indepFun hij).comp₀
      (gaussianMatrix_row_hasLaw i).aemeasurable (gaussianMatrix_row_hasLaw j).aemeasurable hFm hGm
    simp only [Function.comp_def] at h
    rw [ite_eq_right hij]
    exact h.covariance_eq_zero (memLp_gaussianMatrix_row_statistic F hF i)
      (memLp_gaussianMatrix_row_statistic G hG j)

/-- The actual row-sum covariance has exactly one contribution per
original row. Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem covariance_gaussianMatrix_row_sums {d : ℕ} (F G : (Fin d → ℝ) → ℝ)
    (hF : MemLp F 2 (gaussianRowLaw d)) (hG : MemLp G 2 (gaussianRowLaw d)) :
    covariance (fun M : Matrix (Fin d) (Fin d) ℝ => ∑ i : Fin d, F (M i))
      (fun M => ∑ i : Fin d, G (M i)) (gaussianMatrixLaw d) =
        (d : ℝ) * covariance F G (gaussianRowLaw d) := by
  have hFi (i : Fin d) := memLp_gaussianMatrix_row_statistic F hF i
  have hGi (i : Fin d) := memLp_gaussianMatrix_row_statistic G hG i
  have hGs := memLp_finsetSum Finset.univ (fun (i : Fin d) hi => hGi i)
  rw [covariance_fun_sum_left hFi hGs]
  calc
    _ = ∑ i : Fin d, ∑ j : Fin d,
        covariance (fun M : Matrix (Fin d) (Fin d) ℝ => F (M i)) (fun M => G (M j))
          (gaussianMatrixLaw d) := Finset.sum_congr rfl fun i hi => covariance_fun_sum_right hGi (hFi i)
    _ = _ := by
      simp_rw [covariance_gaussianMatrix_row_statistics F G hF hG]
      simp [nsmul_eq_mul]

/-- **Exact covariance of the actual original row averages.**
The original row covariance is divided by the actual dimension.
Source: arXiv:2604.01978v1, §7.3, the covariance in `lem:delta_method`. -/
theorem covariance_gaussianMatrix_row_averages {d : ℕ} (hd : 0 < d)
    (F G : (Fin d → ℝ) → ℝ) (hF : MemLp F 2 (gaussianRowLaw d)) (hG : MemLp G 2 (gaussianRowLaw d)) :
    covariance (fun M : Matrix (Fin d) (Fin d) ℝ => (∑ i : Fin d, F (M i)) / (d : ℝ))
      (fun M => (∑ i : Fin d, G (M i)) / (d : ℝ)) (gaussianMatrixLaw d) =
        covariance F G (gaussianRowLaw d) / (d : ℝ) := by
  have hdR : (d : ℝ) ≠ 0 := by exact_mod_cast hd.ne'
  have he (H : (Fin d → ℝ) → ℝ) :
      (fun M : Matrix (Fin d) (Fin d) ℝ => (∑ i : Fin d, H (M i)) / (d : ℝ)) =
        fun M => (d : ℝ)⁻¹ * (∑ i : Fin d, H (M i)) := by funext M; ring
  rw [he F, he G, covariance_const_mul_left, covariance_const_mul_right,
    covariance_gaussianMatrix_row_sums F G hF hG]
  field_simp

/-- The actual mean-centered product integral equals the original
row covariance over dimension. Source: arXiv:2604.01978v1,
§7.3, the centered observable covariance in `lem:delta_method`. -/
theorem integral_gaussianMatrix_centered_row_averages {d : ℕ} (hd : 0 < d)
    (F G : (Fin d → ℝ) → ℝ) (hF : MemLp F 2 (gaussianRowLaw d)) (hG : MemLp G 2 (gaussianRowLaw d)) :
    (∫ M : Matrix (Fin d) (Fin d) ℝ,
      ((∑ i : Fin d, F (M i)) / (d : ℝ) - ∫ z, F z ∂gaussianRowLaw d) *
        ((∑ i : Fin d, G (M i)) / (d : ℝ) - ∫ z, G z ∂gaussianRowLaw d)
          ∂gaussianMatrixLaw d) = covariance F G (gaussianRowLaw d) / (d : ℝ) := by
  have h := covariance_gaussianMatrix_row_averages hd F G hF hG
  rw [covariance, integral_gaussianMatrix_row_average hd F hF,
    integral_gaussianMatrix_row_average hd G hG] at h
  exact h

/-- Positive dimension and two nonconstant actual row polynomials
satisfy the square-integrability hypotheses jointly under the original law. -/
example : (0 : ℕ) < 2 ∧
    MemLp (fun z : Fin 2 → ℝ => z 0 ^ 2) 2 (gaussianRowLaw 2) ∧
    MemLp (fun z : Fin 2 → ℝ => z 0 * z 1) 2 (gaussianRowLaw 2) := by
  have hF : Measurable (fun z : Fin 2 → ℝ => z 0 ^ 2) := by fun_prop
  have hG : Measurable (fun z : Fin 2 → ℝ => z 0 * z 1) := by fun_prop
  refine ⟨by decide, (memLp_two_iff_integrable_sq hF.aestronglyMeasurable).mpr ?_,
    (memLp_two_iff_integrable_sq hG.aestronglyMeasurable).mpr ?_⟩
  · simpa only [← pow_mul] using integrable_gaussianRow_coordinate_pow (0 : Fin 2) 4
  · simpa only [mul_pow] using integrable_gaussianRow_square_product (0 : Fin 2) 1

end Transformer.Homogenized
