import Transformer.Modes.Section3_CubicCumulant
import Transformer.Modes.Section3_FrequencyMoments
/-!
# The cubic characteristic correction and its Gaussian integrability

The positive-sign characteristic expansion in arXiv:2412.09080v3,
§5.4 `eq:br-9.10`, has cubic coefficient `-i E⟨ξ,X⟩³ / 6`. The
function below is exactly that literal third-moment expression. The
proved cumulant identity identifies it with the coefficients of `ψ`
in §3.1 `eq:psi`; no regularity or error estimate is built into it.

Its homogeneity gives `n B(ξ/√n) = B(ξ)/√n`, the normalization of the
first Edgeworth correction. The actual fourth-moment Taylor estimate
is restated with this coefficient, and its Gaussian-damped version is
proved integrable with a quantitative bound by the third moment.

This supplies the spectral correction used in the `s = 3` comparison.
Identifying its inverse integral with the density correction `ψ`, and
bounding the remaining normalized-sum error, are further proof steps.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes
/-- The literal cubic phase coefficient in the positive-sign characteristic function.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
using the actual third linear-form moment and the sign `i³ = -i`. -/
noncomputable def cubicCharacteristicCorrection (μ : Measure (ℝ × ℝ)) (ξ : ℝ × ℝ) : ℂ :=
  -Complex.I * ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6

/-- The literal phase coefficient equals the source's third-cumulant polynomial.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`. -/
theorem cubicCharacteristicCorrection_eq_cumulant_polynomial
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (ξ : ℝ × ℝ) :
    cubicCharacteristicCorrection μ ξ = -Complex.I *
      ((∑ k ∈ Finset.range 4, cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * ξ.1 ^ k * ξ.2 ^ (3 - k) : ℝ) : ℂ) := by
  have h := integral_dot_cube_div_six_eq_cumulant_polynomial μ hμ hexp ξ
  have hC : ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6 =
      ((∑ k ∈ Finset.range 4, cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * ξ.1 ^ k * ξ.2 ^ (3 - k) : ℝ) : ℂ) := by
    exact_mod_cast h
  unfold cubicCharacteristicCorrection
  calc
    _ = -Complex.I * (((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6) := by ring
    _ = _ := by rw [hC]

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2⟩

/-- The actual cubic phase coefficient is smooth under exponential moments.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, its differentiated correction. -/
theorem contDiff_cubicCharacteristicCorrection (μ : Measure (ℝ × ℝ)) (hexp : HasExpMoments μ) :
    ContDiff ℝ ⊤ (cubicCharacteristicCorrection μ) := by
  have h := Complex.ofRealCLM.contDiff.comp (contDiff_integral_dot_cube μ hexp)
  change ContDiff ℝ ⊤ (fun ξ : ℝ × ℝ => -Complex.I *
    ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6)
  exact (contDiff_const.mul h).div_const _

example : HasExpMoments (lawY 3 0) := hasExpMoments_lawY (by norm_num : (0 : ℝ) < 3) 0

/-- The cubic phase coefficient is bounded by the actual third norm moment.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`. -/
theorem norm_cubicCharacteristicCorrection_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    ‖cubicCharacteristicCorrection μ ξ‖ ≤ (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 3 := by
  have he : cubicCharacteristicCorrection μ ξ =
      -(Complex.I * ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6) := by
    unfold cubicCharacteristicCorrection
    ring
  rw [he, norm_neg]
  exact norm_cubic_characteristic_phase_term_le μ hmom ξ

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Real frequency scaling gives the exact cubic phase scale.
Source: arXiv:2412.09080v3, §3 `thm:br`, the normalized third-order coefficient. -/
theorem cubicCharacteristicCorrection_smul_frequency (μ : Measure (ℝ × ℝ)) (c : ℝ) (ξ : ℝ × ℝ) :
    cubicCharacteristicCorrection μ (c • ξ) = (c : ℂ) ^ 3 * cubicCharacteristicCorrection μ ξ := by
  unfold cubicCharacteristicCorrection
  rw [integral_dot_cube_smul_frequency]
  push_cast
  ring

/-- The normalized-sum phase coefficient has exactly the factor `1 / sqrt n`.
Source: arXiv:2412.09080v3, §3 `thm:br` and §3.1 `eq:psi`. -/
theorem normalized_cubicCharacteristicCorrection (μ : Measure (ℝ × ℝ)) {n : ℕ} (hn : 1 ≤ n)
    (ξ : ℝ × ℝ) :
    (n : ℂ) * cubicCharacteristicCorrection μ ((Real.sqrt n)⁻¹ • ξ) =
      (((Real.sqrt n)⁻¹ : ℝ) : ℂ) * cubicCharacteristicCorrection μ ξ := by
  have h := normalized_cubic_moment μ hn ξ
  have hC : (n : ℂ) * ((∫ z,
      (((Real.sqrt n)⁻¹ • ξ).1 * z.1 + ((Real.sqrt n)⁻¹ • ξ).2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) =
      (((Real.sqrt n)⁻¹ : ℝ) : ℂ) * ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) := by
    exact_mod_cast h
  unfold cubicCharacteristicCorrection
  calc
    _ = (-Complex.I / 6) * ((n : ℂ) * ((∫ z,
      (((Real.sqrt n)⁻¹ • ξ).1 * z.1 + ((Real.sqrt n)⁻¹ • ξ).2 * z.2) ^ 3 ∂μ : ℝ) : ℂ)) := by ring
    _ = (-Complex.I / 6) * ((((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
      ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ)) := by rw [hC]
    _ = _ := by ring

example : 1 ≤ (1 : ℕ) := le_rfl

/-- The actual cubic Taylor error is bounded by the fourth norm moment.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, before normalized-sum powers. -/
theorem norm_characteristic2_sub_cubicCorrection
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) :
    ‖characteristic2 μ ξ - (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2 + cubicCharacteristicCorrection μ ξ)‖ ≤
      (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 4 := by
  convert norm_characteristic2_sub_cubic μ hμ hmom ξ using 1
  congr 1
  unfold cubicCharacteristicCorrection
  ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Gaussian damping makes the actual cubic phase coefficient integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`, the cubic correction
term. Its norm integral is bounded by a finite Gaussian polynomial moment. -/
theorem integrable_cubicCharacteristicCorrection_gaussian_and_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hexp : HasExpMoments μ)
    (hmom : MemLp id 3 μ) {b : ℝ} (hb : 0 < b) :
    Integrable (fun ξ : ℝ × ℝ => cubicCharacteristicCorrection μ ξ *
      ((Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) : ℝ) : ℂ)) ∧
    (∫ ξ : ℝ × ℝ, ‖cubicCharacteristicCorrection μ ξ * ((Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) : ℝ) : ℂ)‖) ≤
      (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) *
        ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ 3 * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) := by
  let K : ℝ := (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)
  have hM : Integrable (fun ξ : ℝ × ℝ => K * (‖ξ‖ ^ 3 * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)))) :=
    (integrable_norm_pow_gaussian_frequency 3 hb).const_mul K
  have hB := (contDiff_cubicCharacteristicCorrection μ hexp).continuous
  have hC : Continuous (fun ξ : ℝ × ℝ => cubicCharacteristicCorrection μ ξ *
      ((Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) : ℝ) : ℂ)) := by fun_prop
  have hp (ξ : ℝ × ℝ) : ‖cubicCharacteristicCorrection μ ξ *
      ((Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) : ℝ) : ℂ)‖ ≤
      K * (‖ξ‖ ^ 3 * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact (mul_le_mul_of_nonneg_right (norm_cubicCharacteristicCorrection_le μ hmom ξ)
      (Real.exp_pos _).le).trans_eq (by dsimp [K]; ring)
  have hI := hM.mono' hC.aestronglyMeasurable (ae_of_all _ hp)
  refine ⟨hI, ?_⟩
  calc
    _ ≤ ∫ ξ : ℝ × ℝ, K * (‖ξ‖ ^ 3 * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) :=
      integral_mono_ae hI.norm hM (ae_of_all _ hp)
    _ = _ := integral_const_mul _ _

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, hasExpMoments_stdGauss2, IsGaussian.memLp_id _ _ (by simp), one_pos⟩

end Transformer.Modes
