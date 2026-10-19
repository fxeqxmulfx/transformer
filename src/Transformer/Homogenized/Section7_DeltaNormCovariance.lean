module

public import Transformer.Homogenized.Section7_ProductQueryFourth
public import Transformer.Homogenized.Section7_GaussianScaling

/-!
# Corrected normalized query-norm covariance in the delta method

ArXiv:2604.01978v1, §7.3, proof of `lem:delta_method`. Its displayed
covariance gives the diagonal squared-norm variance as
`2/d + O(d⁻²)` for `A = (d σ_A²)⁻¹ WᵀW′`.

Under the actual two-factor initialization, the exact variance is
`4/d + 4/d²`. Here the original Gaussian head has entry variance
`1/d`, so its product matrix already has the source's normalization.
The query is an original coordinate unit vector and the squared norm
has mean exactly one. Both original factors and all their entry
independence conditions are retained.

The corrected exact diagonal formula is proved by scaling the actual
product-query moments. The proposed `2/d` leading term is then
refuted beyond every dimension threshold and for every error constant.
This establishes an error in the source covariance calculation. The
full normalized-correlation expansion needs a separate remainder
argument and is not inferred solely from this diagonal discrepancy.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Original Gaussian entry scaling changes the squared query-norm
mean by `c⁴`, including the zero-variance law. Source:
arXiv:2604.01978v1, initialization (G), §7.3, `lem:delta_method`. -/
theorem integral_gaussianHead_scaled_query_norm_sq {d : ℕ} (c : ℝ≥0) (j : Fin d) :
    (∫ θ : HeadParam d, ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2
      ∂gaussHeadLaw d c c) = (c : ℝ) ^ 4 * (d : ℝ) ^ 2 := by
  have hm : Measurable (fun θ : HeadParam d =>
      ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2) := by
    simp_rw [EuclideanSpace.real_norm_sq_eq, qkMap_single]
    fun_prop
  rw [gaussHeadLaw_eq_map_unit_scale, integral_map (by fun_prop) hm.aestronglyMeasurable]
  simp_rw [qkMap_scaled_gaussianHead, norm_smul, Real.norm_of_nonneg (sq_nonneg _), mul_pow]
  rw [integral_const_mul, integral_gaussianHead_query_norm_sq]
  ring

/-- The actual fourth query-norm moment scales by `c⁸`, because
both original Gaussian factors scale. Source: arXiv:2604.01978v1,
§7.3, the normalized matrix used in `lem:delta_method`. -/
theorem integral_gaussianHead_scaled_query_norm_fourth {d : ℕ} (c : ℝ≥0) (j : Fin d) :
    (∫ θ : HeadParam d, ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 4
      ∂gaussHeadLaw d c c) = (c : ℝ) ^ 8 * (d : ℝ) ^ 2 * (d + 2) ^ 2 := by
  have hm : Measurable (fun θ : HeadParam d =>
      ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 4) := by
    simp_rw [show (4 : ℕ) = 2 * 2 by decide, pow_mul,
      EuclideanSpace.real_norm_sq_eq, qkMap_single]
    fun_prop
  rw [gaussHeadLaw_eq_map_unit_scale, integral_map (by fun_prop) hm.aestronglyMeasurable]
  simp_rw [qkMap_scaled_gaussianHead, norm_smul, Real.norm_of_nonneg (sq_nonneg _), mul_pow]
  rw [integral_const_mul, integral_gaussianHead_query_norm_fourth]
  ring

/-- Both original factors scale, giving the exact `c⁸` factor in
squared-norm variance. Source: arXiv:2604.01978v1, §7.3,
the normalization preceding the covariance in `lem:delta_method`. -/
theorem variance_gaussianHead_scaled_query_norm_sq {d : ℕ} (c : ℝ≥0) (j : Fin d) :
    variance (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)
      (gaussHeadLaw d c c) = 4 * (c : ℝ) ^ 8 * (d : ℝ) ^ 2 * (d + 1) := by
  have hm : Measurable (fun θ : HeadParam d =>
      ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2) := by
    simp_rw [EuclideanSpace.real_norm_sq_eq, qkMap_single]
    fun_prop
  rw [gaussHeadLaw_eq_map_unit_scale, variance_map hm.aemeasurable (by fun_prop)]
  simp only [Function.comp_def, qkMap_scaled_gaussianHead, norm_smul,
    Real.norm_of_nonneg (sq_nonneg _), mul_pow]
  rw [variance_const_mul, variance_gaussianHead_query_norm_sq]
  ring

/-- The original normalized product query has squared-norm mean
one, as required by the delta-method center. Source:
arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
theorem integral_std_gaussianHead_query_norm_sq {d : ℕ} (j : Fin d) :
    (∫ θ : HeadParam d, ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2
      ∂gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)) = 1 := by
  have hd : (d : ℝ) ≠ 0 := by have hj := j.isLt; exact_mod_cast (by omega : d ≠ 0)
  rw [integral_gaussianHead_scaled_query_norm_sq,
    show (stdSigmaV d : ℝ) ^ 4 = ((stdSigmaV d : ℝ) ^ 2) ^ 2 by ring, stdSigmaV_sq]
  field_simp

/-- **Corrected exact covariance diagonal for the actual product.**
The source states `2/d + O(d⁻²)`; the original two-factor Gaussian
query instead has squared-norm variance `4/d + 4/d²`.
Source: arXiv:2604.01978v1, §7.3, proof of `lem:delta_method`,
the second and third diagonal entries of its displayed covariance. -/
theorem variance_std_gaussianHead_query_norm_sq {d : ℕ} (j : Fin d) :
    variance (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)
      (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)) = 4 / (d : ℝ) + 4 / (d : ℝ) ^ 2 := by
  have hd : (d : ℝ) ≠ 0 := by have hj := j.isLt; exact_mod_cast (by omega : d ≠ 0)
  rw [variance_gaussianHead_scaled_query_norm_sq,
    show (stdSigmaV d : ℝ) ^ 8 = ((stdSigmaV d : ℝ) ^ 2) ^ 4 by ring, stdSigmaV_sq]
  field_simp

/-- **Counterexamples beyond arbitrary dimension thresholds.**
The `2/d` covariance leading term in the source cannot have an
`O(d⁻²)` error for the actual Gaussian product initialization.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`, displayed covariance. -/
theorem gaussian_delta_norm_covariance_counterexample (C : ℝ) (D : ℕ) :
    ∃ d : ℕ, 0 < d ∧ D ≤ d ∧ ∃ j : Fin d,
      C / (d : ℝ) ^ 2 <
        |variance (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)
          (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)) - 2 / (d : ℝ)| := by
  obtain ⟨d, hd⟩ := exists_nat_gt (max (D : ℝ) (|C| + 1))
  have hlarge : |C| + 1 < (d : ℝ) := lt_of_le_of_lt (le_max_right _ _) hd
  have hdR : (0 : ℝ) < d := by linarith [abs_nonneg C]
  have hdpos : 0 < d := by exact_mod_cast hdR
  have hDd : D ≤ d := by exact_mod_cast ((le_max_left (D : ℝ) (|C| + 1)).trans hd.le)
  let j : Fin d := ⟨0, hdpos⟩
  refine ⟨d, hdpos, hDd, j, ?_⟩
  rw [variance_std_gaussianHead_query_norm_sq]
  have he : 4 / (d : ℝ) + 4 / (d : ℝ) ^ 2 - 2 / (d : ℝ) =
      (2 * (d : ℝ) + 4) / (d : ℝ) ^ 2 := by field_simp; ring
  rw [he, abs_of_pos (div_pos (by positivity) (sq_pos_of_pos hdR))]
  exact div_lt_div_of_pos_right (by linarith [le_abs_self C]) (sq_pos_of_pos hdR)

/-- **The covariance estimate used in the source delta-method proof
is false.** Allowing a prefactor and an arbitrary dimension threshold
does not recover its `2/d + O(d⁻²)` diagonal.
Source: arXiv:2604.01978v1, §7.3, proof of `lem:delta_method`. -/
theorem not_gaussian_delta_norm_covariance :
    ¬ ∃ (C : ℝ) (D : ℕ), 0 < C ∧ ∀ d : ℕ, 0 < d → D ≤ d → ∀ j : Fin d,
      |variance (fun θ : HeadParam d => ‖qkMap θ (EuclideanSpace.single j (1 : ℝ))‖ ^ 2)
        (gaussHeadLaw d (stdSigmaV d) (stdSigmaV d)) - 2 / (d : ℝ)| ≤ C / (d : ℝ) ^ 2 := by
  rintro ⟨C, D, -, h⟩
  obtain ⟨d, hd, hDd, j, hbad⟩ := gaussian_delta_norm_covariance_counterexample C D
  exact not_lt_of_ge (h d hd hDd j) hbad

/-- A positive original Gaussian head and a unit query jointly have
the required mean one and the corrected variance three in dimension
two. Thus the discrepancy uses the actual normalized source model.
Source: arXiv:2604.01978v1, §7.3, `lem:delta_method`. -/
example : 0 < stdSigmaV 2 ∧ ‖EuclideanSpace.single (0 : Fin 2) (1 : ℝ)‖ = 1 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) (stdSigmaV 2) (gaussHeadLaw 2 (stdSigmaV 2) (stdSigmaV 2)) ∧
    (∫ θ : HeadParam 2, ‖qkMap θ (EuclideanSpace.single 0 (1 : ℝ))‖ ^ 2
      ∂gaussHeadLaw 2 (stdSigmaV 2) (stdSigmaV 2)) = 1 ∧
    variance (fun θ : HeadParam 2 => ‖qkMap θ (EuclideanSpace.single 0 (1 : ℝ))‖ ^ 2)
      (gaussHeadLaw 2 (stdSigmaV 2) (stdSigmaV 2)) = 3 := by
  refine ⟨stdSigmaV_pos (by decide), by simp, isGaussianHeadLaw_gaussHeadLaw _ _ _,
    integral_std_gaussianHead_query_norm_sq 0, ?_⟩
  rw [variance_std_gaussianHead_query_norm_sq]; norm_num

end Transformer.Homogenized
