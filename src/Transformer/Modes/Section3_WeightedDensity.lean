import Transformer.Modes.Section3_WeightedFourier
/-!
# Quadratically weighted comparison of actual continuous densities

The inversion step of arXiv:2412.09080v3, §5.4
`eq:higher-error-goal`, controls spatial weights by the integrals of
actual characteristic derivatives. Here both laws, their continuous
nonnegative densities, and all derivatives are literal objects.

The complex coordinate model supplies the Euclidean norm and the
Fourier differentiation formula. The frequency scale `2π` is kept
throughout: the second derivative scale cancels its volume Jacobian,
and the inverse Fourier bound contributes `(2π)⁻²`. Combining this
quadratic bound with the unweighted inversion estimate gives exactly
the spatial weight `1 + |x|²` used in §3 `thm:br` for `s = 2`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- The quadratic spatial weight is controlled by the actual second
characteristic derivative difference. Source: arXiv:2412.09080v3, §5.4
`eq:higher-error-goal`, order two. The coordinate map's operator norm
accounts for the project's product norm. -/
theorem continuous_density_sub_eucl_sq_abs_le
    (μ ν : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hI : Integrable (characteristic2 μ)) (hJ : Integrable (characteristic2 ν))
    (hf : ContDiff ℝ 2 (characteristic2 μ)) (hg : ContDiff ℝ 2 (characteristic2 ν))
    (hD1 : Integrable (fun ξ => ‖fderiv ℝ (characteristic2 μ) ξ - fderiv ℝ (characteristic2 ν) ξ‖))
    (hD2 : Integrable (fun ξ => ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ‖))
    {q r : ℝ × ℝ → ℝ} (hq : Continuous q) (hr : Continuous r)
    (hq0 : ∀ z, 0 ≤ q z) (hr0 : ∀ z, 0 ≤ r z)
    (hμ : μ = volume.withDensity (fun z => ENNReal.ofReal (q z)))
    (hν : ν = volume.withDensity (fun z => ENNReal.ofReal (r z))) (x : ℝ × ℝ) :
    eucl x ^ 2 * |q x - r x| ≤ ((2 * Real.pi) ^ 2)⁻¹ *
      ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 *
        ∫ ξ, ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ‖ := by
  let f : ℝ × ℝ → ℂ := fun ξ => characteristic2 μ ξ - characteristic2 ν ξ
  let g : ℂ → ℂ := fun ξ => f ((2 * Real.pi) • Complex.equivRealProdCLM ξ)
  let L : ℂ →L[ℝ] ℝ × ℝ := (2 * Real.pi) • Complex.equivRealProdCLM.toContinuousLinearMap
  have hfC : ContDiff ℝ 2 f := hf.sub hg
  have hgC : ContDiff ℝ 2 g := hfC.comp L.contDiff
  have he1 : fderiv ℝ f = fun ξ => fderiv ℝ (characteristic2 μ) ξ - fderiv ℝ (characteristic2 ν) ξ := by
    funext ξ
    exact fderiv_fun_sub (hf.differentiable (by norm_num) ξ) (hg.differentiable (by norm_num) ξ)
  have he2 : iteratedFDeriv ℝ 2 f = fun ξ =>
      iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ := by
    change iteratedFDeriv ℝ 2 (characteristic2 μ - characteristic2 ν) = _
    rw [iteratedFDeriv_sub hf hg]
    rfl
  have hIf (k : ℕ) (hk : k ≤ 2) : Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖) := by
    interval_cases k
    · have h0 : Integrable (fun ξ => ‖f ξ‖) := (hI.sub hJ).norm
      simpa only [norm_iteratedFDeriv_zero] using h0
    · simpa only [norm_iteratedFDeriv_one, he1] using hD1
    · simpa only [he2] using hD2
  have hIg (k : ℕ) (hk : k ≤ 2) : Integrable (iteratedFDeriv ℝ k g) :=
    (integrable_iteratedFDeriv_realProd_smul_and_bound f hfC hk (hIf k hk)
      (2 * Real.pi) (by positivity)).1
  have hG : g = normalizedCharacteristic (complexLaw μ) - normalizedCharacteristic (complexLaw ν) := by
    funext ξ
    have he (law : Measure (ℝ × ℝ)) : normalizedCharacteristic (complexLaw law) ξ =
        characteristic2 law ((2 * Real.pi) • Complex.equivRealProdCLM ξ) := by
      unfold normalizedCharacteristic
      rw [charFun_complexLaw]
      change characteristic2 law (Complex.equivRealProdCLM ((2 * Real.pi) • ξ)) = _
      rw [map_smul]
    change f ((2 * Real.pi) • Complex.equivRealProdCLM ξ) =
      normalizedCharacteristic (complexLaw μ) ξ - normalizedCharacteristic (complexLaw ν) ξ
    rw [he μ, he ν]
  let w : ℂ := Complex.measurableEquivRealProd.symm x
  let e : ℂ → ℂ := fun ξ => Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I)
  have he : AEStronglyMeasurable e := (by dsimp [e]; fun_prop : Continuous e).aestronglyMeasurable
  have heb : ∀ᵐ ξ ∂volume, ‖e ξ‖ ≤ (1 : ℝ) := ae_of_all _ fun ξ => by
    dsimp [e]
    rw [Complex.norm_exp_ofReal_mul_I]
  have hU := (integrable_normalizedCharacteristic (complexLaw μ) (integrable_charFun_complexLaw μ hI)).bdd_mul he heb
  have hV := (integrable_normalizedCharacteristic (complexLaw ν) (integrable_charFun_complexLaw ν hJ)).bdd_mul he heb
  have hFT : 𝓕 g w = 𝓕 (normalizedCharacteristic (complexLaw μ)) w -
      𝓕 (normalizedCharacteristic (complexLaw ν)) w := by
    rw [hG, Real.fourier_eq', Real.fourier_eq', Real.fourier_eq']
    simp only [smul_eq_mul]
    rw [← integral_sub hU hV]
    congr 1
    funext ξ
    dsimp [e]
    ring
  have hqr : |q x - r x| ≤ ‖𝓕 g w‖ := by
    rw [continuous_density_eq_inverseCharacteristic μ hI hq hq0 hμ,
      continuous_density_eq_inverseCharacteristic ν hJ hr hr0 hν]
    change |(𝓕 (normalizedCharacteristic (complexLaw μ)) w).re -
      (𝓕 (normalizedCharacteristic (complexLaw ν)) w).re| ≤ _
    rw [← Complex.sub_re, ← hFT]
    exact Complex.abs_re_le_norm _
  have hEucl : ‖w‖ = eucl x := by rw [Complex.norm_eq_sqrt_sq_add_sq]; rfl
  have hFg := norm_sq_mul_norm_fourier_le_second_derivative g hgC hIg w
  have hInt := (integrable_iteratedFDeriv_realProd_smul_and_bound (N := 2) f hfC (k := 2)
    (by omega) (hIf 2 (by omega)) (2 * Real.pi) (by positivity)).2
  have hInt' : (∫ ξ : ℂ, ‖iteratedFDeriv ℝ 2 g ξ‖) ≤
      ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 *
        ∫ ξ, ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ‖ := by
    convert hInt using 1
    rw [he2]
    have hc : (2 * Real.pi) ^ 2 ≠ 0 := by positivity
    field_simp
  calc
    _ ≤ ‖w‖ ^ 2 * ‖𝓕 g w‖ := by rw [hEucl]; exact mul_le_mul_of_nonneg_left hqr (sq_nonneg _)
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ : ℂ, ‖iteratedFDeriv ℝ 2 g ξ‖ := hFg
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ *
        (‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 *
          ∫ ξ, ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ‖) := by gcongr
    _ = _ := by ring

example : Integrable (characteristic2 stdGauss2) ∧ ContDiff ℝ 2 (characteristic2 stdGauss2) ∧
    Integrable (fun ξ => ‖fderiv ℝ (characteristic2 stdGauss2) ξ - fderiv ℝ (characteristic2 stdGauss2) ξ‖) ∧
    Integrable (fun ξ => ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖) ∧ Continuous phi2 ∧
    (∀ z, 0 ≤ phi2 z) ∧ stdGauss2 = volume.withDensity (fun z => ENNReal.ofReal (phi2 z)) := by
  have h1 : Integrable (fun ξ => fderiv ℝ (characteristic2 stdGauss2) ξ -
      fderiv ℝ (characteristic2 stdGauss2) ξ) :=
    integrable_characteristic2_stdGauss2_derivatives.1.sub integrable_characteristic2_stdGauss2_derivatives.1
  have h2 : Integrable (fun ξ => iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ) :=
    integrable_characteristic2_stdGauss2_derivatives.2.sub integrable_characteristic2_stdGauss2_derivatives.2
  exact ⟨integrable_characteristic2_stdGauss2,
    contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp)),
    h1.norm, h2.norm, by unfold phi2; fun_prop, isDensityOf_stdGauss2.nonneg, stdGauss2_eq_withDensity_phi2⟩

/-- The full spatial weight `1 + |x|²` is bounded by zeroth and second
characteristic derivative errors. Source: arXiv:2412.09080v3, §3 `thm:br`,
`s = 2`, and §5.4 `eq:higher-error-goal`. -/
theorem continuous_density_sub_weighted_abs_le
    (μ ν : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hI : Integrable (characteristic2 μ)) (hJ : Integrable (characteristic2 ν))
    (hf : ContDiff ℝ 2 (characteristic2 μ)) (hg : ContDiff ℝ 2 (characteristic2 ν))
    (hD1 : Integrable (fun ξ => ‖fderiv ℝ (characteristic2 μ) ξ - fderiv ℝ (characteristic2 ν) ξ‖))
    (hD2 : Integrable (fun ξ => ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ‖))
    {q r : ℝ × ℝ → ℝ} (hq : Continuous q) (hr : Continuous r)
    (hq0 : ∀ z, 0 ≤ q z) (hr0 : ∀ z, 0 ≤ r z)
    (hμ : μ = volume.withDensity (fun z => ENNReal.ofReal (q z)))
    (hν : ν = volume.withDensity (fun z => ENNReal.ofReal (r z))) (x : ℝ × ℝ) :
    (1 + eucl x ^ 2) * |q x - r x| ≤ ((2 * Real.pi) ^ 2)⁻¹ *
      ((∫ ξ, ‖characteristic2 μ ξ - characteristic2 ν ξ‖) +
        ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 *
          ∫ ξ, ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ‖) := by
  have h0 := continuous_density_sub_abs_le μ ν hI hJ hq hr hq0 hr0 hμ hν x
  have h2 := continuous_density_sub_eucl_sq_abs_le μ ν hI hJ hf hg hD1 hD2 hq hr hq0 hr0 hμ hν x
  calc
    _ = |q x - r x| + eucl x ^ 2 * |q x - r x| := by ring
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ * (∫ ξ, ‖characteristic2 μ ξ - characteristic2 ν ξ‖) +
        ((2 * Real.pi) ^ 2)⁻¹ * ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 2 *
          ∫ ξ, ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 ν) ξ‖ :=
      add_le_add h0 h2
    _ = _ := by ring

example : Integrable (characteristic2 stdGauss2) ∧ ContDiff ℝ 2 (characteristic2 stdGauss2) ∧
    Integrable (fun ξ => ‖fderiv ℝ (characteristic2 stdGauss2) ξ - fderiv ℝ (characteristic2 stdGauss2) ξ‖) ∧
    Integrable (fun ξ => ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖) ∧ Continuous phi2 ∧
    (∀ z, 0 ≤ phi2 z) ∧ stdGauss2 = volume.withDensity (fun z => ENNReal.ofReal (phi2 z)) := by
  have h1 : Integrable (fun ξ => fderiv ℝ (characteristic2 stdGauss2) ξ -
      fderiv ℝ (characteristic2 stdGauss2) ξ) :=
    integrable_characteristic2_stdGauss2_derivatives.1.sub integrable_characteristic2_stdGauss2_derivatives.1
  have h2 : Integrable (fun ξ => iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ) :=
    integrable_characteristic2_stdGauss2_derivatives.2.sub integrable_characteristic2_stdGauss2_derivatives.2
  exact ⟨integrable_characteristic2_stdGauss2,
    contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp)),
    h1.norm, h2.norm, by unfold phi2; fun_prop, isDensityOf_stdGauss2.nonneg, stdGauss2_eq_withDensity_phi2⟩

end Transformer.Modes
