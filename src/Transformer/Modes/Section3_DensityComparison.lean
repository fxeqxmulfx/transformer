import Transformer.Modes.Section3_DensityInversion
import Mathlib.MeasureTheory.Function.AEEqOfLIntegral

/-!
# Density comparison from the actual characteristic functions

The Fourier inversion step of arXiv:2412.09080v3, §5.4
`eq:higher-error-goal`, compares actual continuous densities through
an integral of their Fourier difference. Continuous nonnegative
Lebesgue densities are unique everywhere, not just almost everywhere.
Thus each density agrees pointwise with the proved inverse integral.

The `2π` frequency scaling used by Mathlib contributes the precise
factor `(2π)⁻²` in dimension two. The complex coordinate equivalence
preserves volume, so it contributes no further Jacobian. Integrability
of both actual characteristic functions justifies subtracting their
inverse integrals.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- Two continuous nonnegative densities of the same Lebesgue measure
agree everywhere. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4,
the continuous densities used in Fourier inversion. -/
theorem continuous_nonnegative_density_unique {q r : ℝ × ℝ → ℝ}
    (hq : Continuous q) (hr : Continuous r) (hq0 : ∀ z, 0 ≤ q z) (hr0 : ∀ z, 0 ≤ r z)
    (heq : volume.withDensity (fun z => ENNReal.ofReal (q z)) =
      volume.withDensity (fun z => ENNReal.ofReal (r z))) : q = r := by
  have hae := (withDensity_eq_iff_of_sigmaFinite
    (ENNReal.continuous_ofReal.comp hq).measurable.aemeasurable
    (ENNReal.continuous_ofReal.comp hr).measurable.aemeasurable).mp heq
  apply Measure.eq_of_ae_eq (μ := (volume : Measure (ℝ × ℝ))) _ hq hr
  filter_upwards [hae] with z hz
  have h := congrArg ENNReal.toReal hz
  simpa only [Function.comp_def, ENNReal.toReal_ofReal (hq0 z), ENNReal.toReal_ofReal (hr0 z)] using h

example : Continuous phi2 ∧ (∀ z, 0 ≤ phi2 z) ∧
    volume.withDensity (fun z => ENNReal.ofReal (phi2 z)) =
      volume.withDensity (fun z => ENNReal.ofReal (phi2 z)) :=
  ⟨by unfold phi2; fun_prop, isDensityOf_stdGauss2.nonneg, rfl⟩

/-- Every continuous density of a law with an integrable actual
characteristic function equals its literal inverse integral pointwise.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`. -/
theorem continuous_density_eq_inverseCharacteristic
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hI : Integrable (characteristic2 μ))
    {q : ℝ × ℝ → ℝ} (hq : Continuous q) (hq0 : ∀ z, 0 ≤ q z)
    (heq : μ = volume.withDensity (fun z => ENNReal.ofReal (q z))) :
    q = fun z => inverseCharacteristic (complexLaw μ) (Complex.measurableEquivRealProd.symm z) := by
  have hC := integrable_charFun_complexLaw μ hI
  have hQi := continuous_inverseCharacteristic (complexLaw μ) hC
  have hQ : Continuous (fun z : ℝ × ℝ =>
      inverseCharacteristic (complexLaw μ) (Complex.measurableEquivRealProd.symm z)) :=
    hQi.comp Complex.equivRealProdCLM.symm.continuous
  apply continuous_nonnegative_density_unique hq hQ hq0
    (fun z => inverseCharacteristic_nonneg (complexLaw μ) hC _)
  calc
    _ = μ := heq.symm
    _ = (complexLaw μ).map Complex.measurableEquivRealProd := (complexLaw_map_realProd μ).symm
    _ = (volume.withDensity (fun w => ENNReal.ofReal (inverseCharacteristic (complexLaw μ) w))).map
        Complex.measurableEquivRealProd := congrArg
          (fun law : Measure ℂ => law.map Complex.measurableEquivRealProd)
          (measure_eq_withDensity_inverseCharacteristic (complexLaw μ) hC)
    _ = _ := map_realProd_withDensity _ hQi.measurable

example : ∃ q : ℝ × ℝ → ℝ, Continuous q ∧ (∀ z, 0 ≤ q z) ∧
    ((Measure.pi fun _ : Fin 17 => lawY 3 0).map (scaledSum 17)) =
      volume.withDensity (fun z => ENNReal.ofReal (q z)) := by
  obtain ⟨q, hq, hq0, hqμ, _⟩ :=
    exists_bounded_continuous_density_of_integrable_characteristic2 _
      (integrable_characteristic_scaledSum_lawY (n := 17)
        (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))
  exact ⟨q, hq, hq0, hqμ⟩

/-- Subtracting two integrable actual characteristic functions bounds
the difference of their inverse integrals uniformly in the spatial
variable. Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`. -/
theorem inverseCharacteristic_sub_abs_le (μ ν : Measure ℂ)
    (hI : Integrable (charFun μ)) (hJ : Integrable (charFun ν)) (w : ℂ) :
    |inverseCharacteristic μ w - inverseCharacteristic ν w| ≤
      ∫ ξ, ‖normalizedCharacteristic μ ξ - normalizedCharacteristic ν ξ‖ := by
  let e : ℂ → ℂ := fun ξ => Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I)
  have he : AEStronglyMeasurable e := (by dsimp [e]; fun_prop : Continuous e).aestronglyMeasurable
  have heb : ∀ᵐ ξ ∂volume, ‖e ξ‖ ≤ (1 : ℝ) := ae_of_all _ fun ξ => by
    dsimp [e]
    rw [Complex.norm_exp_ofReal_mul_I]
  have hU := (integrable_normalizedCharacteristic μ hI).bdd_mul he heb
  have hV := (integrable_normalizedCharacteristic ν hJ).bdd_mul he heb
  have hd : 𝓕 (normalizedCharacteristic μ) w - 𝓕 (normalizedCharacteristic ν) w =
      ∫ ξ, e ξ * (normalizedCharacteristic μ ξ - normalizedCharacteristic ν ξ) := by
    rw [Real.fourier_eq', Real.fourier_eq']
    simp only [smul_eq_mul]
    rw [← integral_sub hU hV]
    congr 1
    funext ξ
    ring
  change |(𝓕 (normalizedCharacteristic μ) w).re - (𝓕 (normalizedCharacteristic ν) w).re| ≤ _
  rw [← Complex.sub_re, hd]
  apply (Complex.abs_re_le_norm _).trans
  apply (norm_integral_le_integral_norm _).trans_eq
  apply integral_congr_ae
  exact ae_of_all _ fun ξ => by
    dsimp [e]
    rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]

example : Integrable (charFun (complexLaw
    ((Measure.pi fun _ : Fin 17 => lawY 3 0).map (scaledSum 17)))) ∧
    Integrable (charFun (complexLaw
      ((Measure.pi fun _ : Fin 18 => lawY 3 0).map (scaledSum 18)))) :=
  ⟨integrable_charFun_complexLaw_scaledSum_lawY (by norm_num : (0 : ℝ) < 3) 0 (by norm_num),
    integrable_charFun_complexLaw_scaledSum_lawY (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)⟩

/-- The two-dimensional frequency normalization contributes exactly
`(2π)⁻²` to the integral of the characteristic difference; the coordinate
change preserves volume. Source: arXiv:2412.09080v3, §5.4
`eq:higher-error-goal`, with its `exp(-i dot)` convention. -/
theorem integral_norm_normalizedCharacteristic_sub_complexLaw (μ ν : Measure (ℝ × ℝ)) :
    (∫ ξ : ℂ, ‖normalizedCharacteristic (complexLaw μ) ξ - normalizedCharacteristic (complexLaw ν) ξ‖) =
      ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ : ℝ × ℝ, ‖characteristic2 μ ξ - characteristic2 ν ξ‖ := by
  let f : ℂ → ℝ := fun ξ => ‖charFun (complexLaw μ) ξ - charFun (complexLaw ν) ξ‖
  change (∫ ξ : ℂ, f ((2 * Real.pi) • ξ)) = _
  rw [Measure.integral_comp_smul_of_nonneg volume f (2 * Real.pi) (hR := by positivity)]
  simp only [Complex.finrank_real_complex, smul_eq_mul]
  congr 1
  have h := Complex.volume_preserving_equiv_real_prod.integral_comp
    Complex.measurableEquivRealProd.measurableEmbedding
    (fun ξ : ℝ × ℝ => ‖characteristic2 μ ξ - characteristic2 ν ξ‖)
  simpa only [f, charFun_complexLaw, Complex.measurableEquivRealProd_apply] using h

/-- Continuous densities of actual probability laws with integrable
characteristic functions obey the uniform inverse-Fourier difference
bound with the precise normalization constant. Source:
arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`. -/
theorem continuous_density_sub_abs_le
    (μ ν : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hI : Integrable (characteristic2 μ)) (hJ : Integrable (characteristic2 ν))
    {q r : ℝ × ℝ → ℝ} (hq : Continuous q) (hr : Continuous r)
    (hq0 : ∀ z, 0 ≤ q z) (hr0 : ∀ z, 0 ≤ r z)
    (hμ : μ = volume.withDensity (fun z => ENNReal.ofReal (q z)))
    (hν : ν = volume.withDensity (fun z => ENNReal.ofReal (r z))) (x : ℝ × ℝ) :
    |q x - r x| ≤ ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ, ‖characteristic2 μ ξ - characteristic2 ν ξ‖ := by
  rw [continuous_density_eq_inverseCharacteristic μ hI hq hq0 hμ,
    continuous_density_eq_inverseCharacteristic ν hJ hr hr0 hν]
  have h := inverseCharacteristic_sub_abs_le (complexLaw μ) (complexLaw ν)
    (integrable_charFun_complexLaw μ hI) (integrable_charFun_complexLaw ν hJ)
    (Complex.measurableEquivRealProd.symm x)
  rw [integral_norm_normalizedCharacteristic_sub_complexLaw] at h
  exact h

example : ∃ q r : ℝ × ℝ → ℝ, Continuous q ∧ Continuous r ∧
    IsDensityOf (Measure.pi fun _ : Fin 17 => lawY 3 0) (scaledSum 17) q ∧
    IsDensityOf (Measure.pi fun _ : Fin 18 => lawY 3 0) (scaledSum 18) r := by
  obtain ⟨q, hq, hqμ, _⟩ := exists_continuous_density_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)
  obtain ⟨r, hr, hrμ, _⟩ := exists_continuous_density_scaledSum_lawY (n := 18)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)
  exact ⟨q, r, hq, hr, hqμ, hrμ⟩

end Transformer.Modes
