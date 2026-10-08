import Transformer.Modes.Section3_CubicFourierInversion
import Transformer.Modes.Section3_CubicIntegralComparison
import Transformer.Modes.Section3_WeightedFourier

/-!
# The actual sum density after its first Edgeworth correction

The zero-derivative step of arXiv:2412.09080v3, §3 `thm:br`, `s = 3`,
now follows for the literal correction of §3.1 `eq:psi`. The actual
inverse Gaussian cubic coefficient has been identified with `ψ`, so
subtracting the three integrable Fourier integrals bounds the actual
continuous density error by the cubic-corrected characteristic error.

The volume-preserving complex coordinates and frequency scale `2π`
retain the precise factor `(2π)⁻²`. The spectral estimate for the actual
normalized sum then gives a uniform `C/n` density error. Continuous
densities also exist eventually under the same original assumptions.

These are unweighted consequences of `thm:br`. The additional spatial
factor `1 + |x|³` still requires controlling Fourier derivatives through
order three. The source's `edgeworth_three` statement is kept separately;
no derivative-error estimate or signed correction law is assumed here.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- The actual continuous density error after the signed cubic correction obeys the inverse-Fourier bound.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, §3.1 `eq:psi`,
and §5.4 `eq:higher-error-goal`, its zero-derivative consequence. -/
theorem continuous_density_sub_gaussian_cubic_abs_le
    (σ μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure σ] [IsProbabilityMeasure μ]
    (hI : Integrable (characteristic2 σ)) (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (hmom : MemLp id 3 μ) (r : ℝ)
    {q : ℝ × ℝ → ℝ} (hq : Continuous q) (hq0 : ∀ z, 0 ≤ q z)
    (hσ : σ = volume.withDensity (fun z => ENNReal.ofReal (q z))) (x : ℝ × ℝ) :
    |q x - phi2 x - r * psiOf μ x| ≤ ((2 * Real.pi) ^ 2)⁻¹ *
      ∫ ξ : ℝ × ℝ, ‖characteristic2 σ ξ - characteristic2 stdGauss2 ξ -
        (r : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ := by
  let f : ℝ × ℝ → ℂ := fun ξ => characteristic2 σ ξ - characteristic2 stdGauss2 ξ -
    (r : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ
  let g : ℂ → ℂ := fun ξ => f ((2 * Real.pi) • (ξ.re, ξ.im))
  let b : ℂ → ℂ := fun ξ => characteristic2 stdGauss2 ((2 * Real.pi) • (ξ.re, ξ.im)) *
    cubicCharacteristicCorrection μ ((2 * Real.pi) • (ξ.re, ξ.im))
  let w : ℂ := Complex.measurableEquivRealProd.symm x
  let e : ℂ → ℂ := fun ξ => Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I)
  have he : AEStronglyMeasurable e := (by dsimp [e]; fun_prop : Continuous e).aestronglyMeasurable
  have heb : ∀ᵐ ξ ∂volume, ‖e ξ‖ ≤ (1 : ℝ) := ae_of_all _ fun ξ => by
    dsimp only [e]
    rw [Complex.norm_exp_ofReal_mul_I]
  have hU := (integrable_normalizedCharacteristic (complexLaw σ) (integrable_charFun_complexLaw σ hI)).bdd_mul he heb
  have hV := (integrable_normalizedCharacteristic (complexLaw stdGauss2)
    (integrable_charFun_complexLaw stdGauss2 integrable_characteristic2_stdGauss2)).bdd_mul he heb
  have hZ := (integrable_gaussian_cubic_complex_coordinates μ hexp hmom).bdd_mul he heb
  have hN (law : Measure (ℝ × ℝ)) (ξ : ℂ) : normalizedCharacteristic (complexLaw law) ξ =
      characteristic2 law ((2 * Real.pi) • (ξ.re, ξ.im)) := by
    simp only [normalizedCharacteristic, charFun_complexLaw, Complex.smul_re, Complex.smul_im]
    rfl
  have hFT : (𝓕 g) w = (𝓕 (normalizedCharacteristic (complexLaw σ))) w -
      (𝓕 (normalizedCharacteristic (complexLaw stdGauss2))) w - (r : ℂ) * (𝓕 b) w := by
    rw [Real.fourier_eq', Real.fourier_eq', Real.fourier_eq', Real.fourier_eq']
    simp only [smul_eq_mul]
    rw [← integral_const_mul, ← integral_sub hU hV]
    rw [← integral_sub (f := fun ξ => e ξ * normalizedCharacteristic (complexLaw σ) ξ -
        e ξ * normalizedCharacteristic (complexLaw stdGauss2) ξ)
      (g := fun ξ => (r : ℂ) * (e ξ * b ξ)) (hU.sub hV) (hZ.const_mul (r : ℂ))]
    congr 1
    funext ξ
    dsimp only [g, f, b, e]
    rw [hN σ, hN stdGauss2]
    ring
  have hψ : (𝓕 b) w = ((psiOf μ x : ℝ) : ℂ) :=
    fourier_gaussian_cubic_eq_psiOf μ hμ hexp hmom x
  have hqr : |q x - phi2 x - r * psiOf μ x| ≤ ‖(𝓕 g) w‖ := by
    rw [continuous_density_eq_inverseCharacteristic σ hI hq hq0 hσ,
      ← inverseCharacteristic_complexLaw_stdGauss2 x]
    change |(𝓕 (normalizedCharacteristic (complexLaw σ)) w).re -
      (𝓕 (normalizedCharacteristic (complexLaw stdGauss2)) w).re - r * psiOf μ x| ≤ _
    have hr : (𝓕 g w).re = (𝓕 (normalizedCharacteristic (complexLaw σ)) w).re -
        (𝓕 (normalizedCharacteristic (complexLaw stdGauss2)) w).re - r * psiOf μ x := by
      rw [hFT, hψ]
      simp only [Complex.sub_re, Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, mul_zero, sub_zero]
    rw [← hr]
    exact Complex.abs_re_le_norm _
  calc
    _ ≤ ‖(𝓕 g) w‖ := hqr
    _ ≤ ∫ ξ : ℂ, ‖g ξ‖ := by
      rw [Real.fourier_eq']
      apply (norm_integral_le_integral_norm _).trans_eq
      apply integral_congr_ae
      exact ae_of_all _ fun ξ => by
        dsimp only
        rw [norm_smul, Complex.norm_exp_ofReal_mul_I, one_mul]
    _ = _ := integral_realProd_smul (fun ξ => ‖f ξ‖) (2 * Real.pi) (by positivity)

example : IsProbabilityMeasure stdGauss2 ∧ Integrable (characteristic2 stdGauss2) ∧
    IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    Continuous phi2 ∧ (∀ z, 0 ≤ phi2 z) ∧
    stdGauss2 = volume.withDensity (fun z => ENNReal.ofReal (phi2 z)) :=
  ⟨inferInstance, integrable_characteristic2_stdGauss2, isStandardized_stdGauss2,
    hasExpMoments_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    by unfold phi2; fun_prop, isDensityOf_stdGauss2.nonneg, stdGauss2_eq_withDensity_phi2⟩

/-- Every actual continuous normalized-sum density has uniform cubic-corrected error `O(1/n)`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, §3.1 `eq:psi`,
and §5.4 `eq:higher-error-goal`, its zero-derivative consequence. -/
theorem eventually_abs_scaledSum_density_sub_phi2_psiOf
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ,
      Continuous q → IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, |q x - phi2 x - (Real.sqrt n)⁻¹ * psiOf μ x| ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨C, hC, herror⟩ := eventually_integral_characteristic_scaledSum_sub_gaussian_cubic_rate μ hμ hmom hexp hcf
  refine ⟨((2 * Real.pi) ^ 2)⁻¹ * C, by positivity, ?_⟩
  filter_upwards [herror, eventually_integrable_characteristic_scaledSum μ hcf] with n hn hIn
  intro q hq hqd x
  have h := continuous_density_sub_gaussian_cubic_abs_le _ μ hIn hμ hexp
    (hmom.mono_exponent (by norm_num)) (Real.sqrt n)⁻¹ hq hqd.nonneg hqd.map_eq x
  calc
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ : ℝ × ℝ,
        ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
          characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ := h
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ * (C * (n : ℝ)⁻¹) := mul_le_mul_of_nonneg_left hn (by positivity)
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

/-- Actual continuous sum densities exist eventually and have cubic-corrected uniform error `O(1/n)`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, §3.1 `eq:psi`,
and §5.4 `eq:higher-error-goal`, its zero-derivative consequence. -/
theorem eventually_exists_scaledSum_density_gaussian_cubic_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∃ q : ℝ × ℝ → ℝ,
      Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q ∧
      ∀ x, |q x - phi2 x - (Real.sqrt n)⁻¹ * psiOf μ x| ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨C, hC, herror⟩ := eventually_abs_scaledSum_density_sub_phi2_psiOf μ hμ hmom hexp hcf
  refine ⟨C, hC, ?_⟩
  filter_upwards [herror, eventually_exists_continuous_density_scaledSum μ hcf] with n hn hex
  obtain ⟨q, hq, hqd, _⟩ := hex
  exact ⟨q, hq, hqd, hn q hq hqd⟩

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
