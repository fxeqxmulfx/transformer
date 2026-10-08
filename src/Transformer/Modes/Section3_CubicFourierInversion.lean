import Transformer.Modes.Section3_CubicFourier

/-!
# Actual inverse Fourier integral of the cubic density correction

The positive-sign transform proved for the literal `ψ` in
arXiv:2412.09080v3, §3.1 `eq:psi`, is transported to complex coordinates.
These coordinates have the Euclidean norm required by Mathlib's Fourier
inversion theorem, and the measurable equivalence preserves volume.

The reflected transform `𝓕 ψ(-ξ)` is precisely the Gaussian cubic
coefficient at `2π ξ`. Its integrability is proved from the actual
third norm moment and Gaussian damping. Continuity and integrability
of `ψ` then justify Fourier inversion pointwise, so the inverse spectral
correction equals exactly the source's spatial function.

All signs and the `2π` frequency scale are explicit. This uses the
corrected monomial multiplier for the §5.4 display before
`eq:higher-error-goal`, retaining the original spatial `eq:psi`.
The correction is a signed integrable function, not an assumed law.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- The actual spatial correction is smooth in complex coordinates.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem contDiff_psiOf_complex_coordinates (μ : Measure (ℝ × ℝ)) :
    ContDiff ℝ ⊤ (fun x : ℂ => ((psiOf μ (x.re, x.im) : ℝ) : ℂ)) := by
  exact Complex.ofRealCLM.contDiff.comp ((contDiff_psiOf μ).comp Complex.equivRealProdCLM.contDiff)

/-- The volume-preserving coordinates retain the actual correction integrability.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_psiOf_complex_coordinates (μ : Measure (ℝ × ℝ)) :
    Integrable (fun x : ℂ => ((psiOf μ (x.re, x.im) : ℝ) : ℂ)) := by
  exact Complex.volume_preserving_equiv_real_prod.integrable_comp_of_integrable (integrable_psiOf_complex μ)

/-- The reflected actual transform is the normalized positive-sign Gaussian cubic coefficient.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem fourier_psiOf_complex_neg
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (ξ : ℂ) :
    (𝓕 (fun x : ℂ => ((psiOf μ (x.re, x.im) : ℝ) : ℂ))) (-ξ) =
      characteristic2 stdGauss2 ((2 * Real.pi) • (ξ.re, ξ.im)) *
        cubicCharacteristicCorrection μ ((2 * Real.pi) • (ξ.re, ξ.im)) := by
  let f : ℝ × ℝ → ℂ := fun x => Complex.exp
    (Complex.I * ((((2 * Real.pi) • (ξ.re, ξ.im)).1 * x.1 +
      ((2 * Real.pi) • (ξ.re, ξ.im)).2 * x.2 : ℝ) : ℂ)) * ((psiOf μ x : ℝ) : ℂ)
  rw [Real.fourier_eq']
  have he (x : ℂ) : Complex.exp (((-2 * Real.pi * inner ℝ x (-ξ) : ℝ) : ℂ) * Complex.I) •
      ((psiOf μ (x.re, x.im) : ℝ) : ℂ) = f (Complex.measurableEquivRealProd x) := by
    dsimp only [f]
    simp only [Complex.measurableEquivRealProd_apply, Prod.smul_fst, Prod.smul_snd,
      smul_eq_mul, Complex.inner, Complex.mul_re, Complex.neg_re, Complex.neg_im,
      Complex.conj_re, Complex.conj_im]
    congr 1
    congr 1
    push_cast
    ring
  simp_rw [he]
  rw [Complex.volume_preserving_equiv_real_prod.integral_comp
    Complex.measurableEquivRealProd.measurableEmbedding f]
  exact integral_psiOf_characteristic_eq_gaussian_cubic μ hμ hexp _

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2⟩

/-- The actual third moment makes the normalized spectral correction integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_gaussian_cubic_complex_coordinates
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (hexp : HasExpMoments μ) (hmom : MemLp id 3 μ) :
    Integrable (fun ξ : ℂ => characteristic2 stdGauss2 ((2 * Real.pi) • (ξ.re, ξ.im)) *
      cubicCharacteristicCorrection μ ((2 * Real.pi) • (ξ.re, ξ.im))) := by
  have hG : Integrable (fun ξ : ℝ × ℝ => characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ) := by
    convert (integrable_cubicCharacteristicCorrection_gaussian_and_bound μ hexp hmom
      (by norm_num : (0 : ℝ) < 1 / 2)).1 using 1
    funext ξ
    have he : Real.exp (-(1 / 2 : ℝ) * (ξ.1 ^ 2 + ξ.2 ^ 2)) =
        Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) := by congr 1; ring
    rw [characteristic2_stdGauss2, he]
    ring
  have hC := Complex.volume_preserving_equiv_real_prod.integrable_comp_of_integrable hG
  convert hC.comp_smul (by positivity : (2 * Real.pi : ℝ) ≠ 0) using 1
  funext ξ
  simp only [Function.comp_def, Complex.measurableEquivRealProd_apply,
    Complex.smul_re, Complex.smul_im]
  rfl

example : IsProbabilityMeasure stdGauss2 ∧ HasExpMoments stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, hasExpMoments_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual transform of the spatial correction is integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_fourier_psiOf_complex_coordinates
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (hmom : MemLp id 3 μ) :
    Integrable (𝓕 (fun x : ℂ => ((psiOf μ (x.re, x.im) : ℝ) : ℂ))) := by
  convert (integrable_gaussian_cubic_complex_coordinates μ hexp hmom).comp_neg using 1
  funext ξ
  have h := fourier_psiOf_complex_neg μ hμ hexp (-ξ)
  simpa only [neg_neg] using h

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2,
    IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual inverse spectral integral recovers the literal spatial correction.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem fourier_gaussian_cubic_complex_coordinates
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (hmom : MemLp id 3 μ) (x : ℂ) :
    (𝓕 (fun ξ : ℂ => characteristic2 stdGauss2 ((2 * Real.pi) • (ξ.re, ξ.im)) *
      cubicCharacteristicCorrection μ ((2 * Real.pi) • (ξ.re, ξ.im)))) x =
        ((psiOf μ (x.re, x.im) : ℝ) : ℂ) := by
  have h := (contDiff_psiOf_complex_coordinates μ).continuous.fourierInv_fourier_eq
    (integrable_psiOf_complex_coordinates μ) (integrable_fourier_psiOf_complex_coordinates μ hμ hexp hmom)
  rw [Real.fourierInv_eq_fourier_comp_neg] at h
  have he : (fun ξ : ℂ => characteristic2 stdGauss2 ((2 * Real.pi) • (ξ.re, ξ.im)) *
      cubicCharacteristicCorrection μ ((2 * Real.pi) • (ξ.re, ξ.im))) =
        fun ξ => (𝓕 (fun x : ℂ => ((psiOf μ (x.re, x.im) : ℝ) : ℂ))) (-ξ) := by
    funext ξ
    exact (fourier_psiOf_complex_neg μ hμ hexp ξ).symm
  rw [he]
  exact congrFun h x

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2,
    IsGaussian.memLp_id _ _ (by simp)⟩

/-- In the source's real-pair coordinates the actual inverse correction is exactly `ψ`.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem fourier_gaussian_cubic_eq_psiOf
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (hmom : MemLp id 3 μ) (x : ℝ × ℝ) :
    (𝓕 (fun ξ : ℂ => characteristic2 stdGauss2 ((2 * Real.pi) • (ξ.re, ξ.im)) *
      cubicCharacteristicCorrection μ ((2 * Real.pi) • (ξ.re, ξ.im))))
        (Complex.measurableEquivRealProd.symm x) = ((psiOf μ x : ℝ) : ℂ) := by
  rw [fourier_gaussian_cubic_complex_coordinates μ hμ hexp hmom]
  have hx : ((Complex.measurableEquivRealProd.symm x).re,
      (Complex.measurableEquivRealProd.symm x).im) = x :=
    Complex.measurableEquivRealProd.apply_symm_apply x
  rw [hx]

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2,
    IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
