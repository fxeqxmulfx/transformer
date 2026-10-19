module

public import Transformer.Homogenized.Section7_GaussianProductQueries

/-!
# The corrected correlation expansion for two actual Gaussian factors

ArXiv:2604.01978v1, §7.3, `lem:delta_method`. The source
asserts correction `(r³-r)/(2d)` for the product of two independent
Gaussian matrix factors. Each actual factor contributes that correction;
their product instead has correction `(r³-r)/d`.

Conditioning on the first original matrix leaves the second matrix
with its original independent Gaussian law. The intermediate normalized
queries are jointly unit almost surely, so the proved uniform
single-factor theorem applies at their actual random correlation `R`.
All conditional errors and cubic corrections are genuinely integrable.

The first factor's expected correlation has its own single-factor
correction. The change from `E(R³-R)` to `r³-r`, divided by `2d`,
has already been bounded at the uniform `d^(-3/2)` rate.
Adding the two factor errors and this proved composition error gives
the corrected product expansion for every original unit input pair.

This module uses the actual product of independent standard Gaussian
matrix laws. Transplanting it to every original Gaussian head law,
with arbitrary value variance and positive query variance, is separate.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- **Uniform conditional expansion for the original product correlation.**
The correction is evaluated at the actual intermediate random correlation.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
conditioning on the first original independent Gaussian factor. -/
theorem gaussianMatrix_product_conditional_delta_method :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
      |(∫ P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ,
          inner ℝ (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)))
            (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 y)))
              ∂(gaussianMatrixLaw d).prod (gaussianMatrixLaw d)) -
        ((∫ N : Matrix (Fin d) (Fin d) ℝ,
          inner ℝ (normalizeLayer (Matrix.toEuclideanLin N x))
            (normalizeLayer (Matrix.toEuclideanLin N y)) ∂gaussianMatrixLaw d) +
          (∫ N : Matrix (Fin d) (Fin d) ℝ,
            ((inner ℝ (normalizeLayer (Matrix.toEuclideanLin N x))
              (normalizeLayer (Matrix.toEuclideanLin N y))) ^ 3 -
                inner ℝ (normalizeLayer (Matrix.toEuclideanLin N x))
                  (normalizeLayer (Matrix.toEuclideanLin N y))) ∂gaussianMatrixLaw d) / (2 * (d : ℝ)))| ≤
        C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨C, hC, hfactor⟩ := gaussianMatrix_factor_delta_method
  refine ⟨C, hC, ?_⟩
  intro d hd x y hx hy
  let R : Matrix (Fin d) (Fin d) ℝ → ℝ := fun N =>
    inner ℝ (normalizeLayer (Matrix.toEuclideanLin N x)) (normalizeLayer (Matrix.toEuclideanLin N y))
  let F : Matrix (Fin d) (Fin d) ℝ → ℝ := fun N =>
    ∫ M : Matrix (Fin d) (Fin d) ℝ,
      inner ℝ (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N x))))
        (normalizeLayer (Matrix.toEuclideanLin M (normalizeLayer (Matrix.toEuclideanLin N y)))) ∂gaussianMatrixLaw d
  have hR : Measurable R := measurable_normalized_inner _ _
    (measurable_gaussianMatrix_query x) (measurable_gaussianMatrix_query y)
  have iR : Integrable R (gaussianMatrixLaw d) := integrable_normalized_inner _ _ _
    (measurable_gaussianMatrix_query x) (measurable_gaussianMatrix_query y)
  have iF : Integrable F (gaussianMatrixLaw d) :=
    integrable_gaussian_product_conditional_correlation x y
  have hbound (N : Matrix (Fin d) (Fin d) ℝ) : |R N| ≤ 1 := by
    exact (abs_real_inner_le_norm _ _).trans
      ((mul_le_mul (norm_normalizeLayer_le_one _) (norm_normalizeLayer_le_one _)
        (norm_nonneg _) (by norm_num : (0 : ℝ) ≤ 1)).trans_eq (one_mul _))
  have iR3 : Integrable (fun N => R N ^ 3) (gaussianMatrixLaw d) := by
    apply Integrable.of_bound (by fun_prop) 1
    filter_upwards with N
    rw [Real.norm_eq_abs, abs_pow]
    exact (pow_le_pow_left₀ (abs_nonneg _) (hbound N) 3).trans_eq (one_pow _)
  have ibias : Integrable (fun N => R N ^ 3 - R N) (gaussianMatrixLaw d) := by
    convert iR3.sub iR using 1
  have iB : Integrable (fun N => R N + (R N ^ 3 - R N) / (2 * (d : ℝ)))
      (gaussianMatrixLaw d) := by
    convert iR.add (ibias.div_const (2 * (d : ℝ))) using 1
  have ierr : Integrable (fun N => F N - (R N + (R N ^ 3 - R N) / (2 * (d : ℝ))))
      (gaussianMatrixLaw d) := by convert iF.sub iB using 1
  have hae : ∀ᵐ N ∂gaussianMatrixLaw d,
      |F N - (R N + (R N ^ 3 - R N) / (2 * (d : ℝ)))| ≤ C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
    filter_upwards [ae_gaussianMatrix_normalized_queries_unit hd x y hx hy] with N hN
    exact hfactor d hd (normalizeLayer (Matrix.toEuclideanLin N x))
      (normalizeLayer (Matrix.toEuclideanLin N y)) hN.1 hN.2
  have hb : |(∫ N, F N ∂gaussianMatrixLaw d) -
      ∫ N, (R N + (R N ^ 3 - R N) / (2 * (d : ℝ))) ∂gaussianMatrixLaw d| ≤
        C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
    rw [← integral_sub iF iB]
    refine abs_integral_le_integral_abs.trans ?_
    calc
      _ ≤ ∫ _N : Matrix (Fin d) (Fin d) ℝ, C * (d : ℝ) ^ (-(3 : ℝ) / 2) ∂gaussianMatrixLaw d :=
        integral_mono_ae (by simpa only [Real.norm_eq_abs] using ierr.norm) (integrable_const _) hae
      _ = _ := by simp
  rw [integral_add iR (ibias.div_const (2 * (d : ℝ))), integral_div] at hb
  rw [integral_gaussian_product_correlation]
  exact hb

/-- **Corrected uniform delta method for the actual two-factor product.**
The source gives `(r³-r)/(2d)`; the proved correction is `(r³-r)/d`.
All original unit input pairs and positive dimensions share one constant.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`,
with its two independent Gaussian factors retained. -/
theorem gaussianMatrix_product_delta_method :
    ∃ C : ℝ, 0 < C ∧ ∀ d : ℕ, 0 < d → ∀ x y : EucSpace d, ‖x‖ = 1 → ‖y‖ = 1 →
      |(∫ P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ,
          inner ℝ (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)))
            (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 y)))
              ∂(gaussianMatrixLaw d).prod (gaussianMatrixLaw d)) -
          (inner ℝ x y + (inner ℝ x y ^ 3 - inner ℝ x y) / (d : ℝ))| ≤
        C * (d : ℝ) ^ (-(3 : ℝ) / 2) := by
  obtain ⟨Cp, hCp, hp⟩ := gaussianMatrix_product_conditional_delta_method
  obtain ⟨Cf, hCf, hf⟩ := gaussianMatrix_factor_delta_method
  refine ⟨Cp + Cf + 6 * Real.sqrt 2, by positivity, ?_⟩
  intro d hd x y hx hy
  let r := inner ℝ x y
  let m := ∫ N : Matrix (Fin d) (Fin d) ℝ,
    inner ℝ (normalizeLayer (Matrix.toEuclideanLin N x))
      (normalizeLayer (Matrix.toEuclideanLin N y)) ∂gaussianMatrixLaw d
  let b := ∫ N : Matrix (Fin d) (Fin d) ℝ,
    ((inner ℝ (normalizeLayer (Matrix.toEuclideanLin N x))
      (normalizeLayer (Matrix.toEuclideanLin N y))) ^ 3 -
        inner ℝ (normalizeLayer (Matrix.toEuclideanLin N x))
          (normalizeLayer (Matrix.toEuclideanLin N y))) ∂gaussianMatrixLaw d
  let e := ∫ P : Matrix (Fin d) (Fin d) ℝ × Matrix (Fin d) (Fin d) ℝ,
    inner ℝ (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 x)))
      (normalizeLayer (Matrix.toEuclideanLin P.2 (Matrix.toEuclideanLin P.1 y)))
        ∂(gaussianMatrixLaw d).prod (gaussianMatrixLaw d)
  have h1 : |e - (m + b / (2 * (d : ℝ)))| ≤ Cp * (d : ℝ) ^ (-(3 : ℝ) / 2) :=
    hp d hd x y hx hy
  have h2 : |m - (r + (r ^ 3 - r) / (2 * (d : ℝ)))| ≤ Cf * (d : ℝ) ^ (-(3 : ℝ) / 2) :=
    hf d hd x y hx hy
  have h3 : |(b - (r ^ 3 - r)) / (2 * (d : ℝ))| ≤
      6 * Real.sqrt 2 * (d : ℝ) ^ (-(3 : ℝ) / 2) :=
    integral_gaussianMatrix_unit_cubic_bias_scaled_deviation_le hd x y hx hy
  change |e - (r + (r ^ 3 - r) / (d : ℝ))| ≤ _
  calc
    _ = |(e - (m + b / (2 * (d : ℝ)))) + (m - (r + (r ^ 3 - r) / (2 * (d : ℝ)))) +
        (b - (r ^ 3 - r)) / (2 * (d : ℝ))| := by congr 1; ring
    _ ≤ |e - (m + b / (2 * (d : ℝ)))| + |m - (r + (r ^ 3 - r) / (2 * (d : ℝ)))| +
        |(b - (r ^ 3 - r)) / (2 * (d : ℝ))| := (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
    _ ≤ _ := by linarith

/-- Two genuine coordinate unit inputs in positive dimension
jointly satisfy every original input hypothesis of both product expansions. -/
example : (0 : ℕ) < 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 := by
  exact ⟨by decide, by simp, by simp⟩

end Transformer.Homogenized
