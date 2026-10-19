module

public import Transformer.Modes.Section3_CubicDensity
public import Transformer.Modes.Section3_CubicWeightedFourier
public import Transformer.Modes.Section3_CubicDerivativeIntegral

/-!
# Cubically weighted Edgeworth estimates for actual sum densities

The actual corrected characteristic error now has every analytic
input for arXiv:2412.09080v3, §3 `thm:br`, `s = 3`.
Its Fourier transform is the density error with the literal `psiOf`
correction from §3.1 `eq:psi`. Orders zero and three of the proved
Fourier bound supply the full spatial weight `1 + |x|³`.

The frequency coordinate map and scale retain the factor `(2π)⁻²`.
The actual error integrals are bounded by fixed-law constants over
`n`, so every continuous density of every sufficiently large
normalized sum satisfies the weighted inverse-n estimate.

The assumptions remain standardization, a fourth moment,
exponential moments for the literal cumulants, and the integrable
characteristic-power condition. Continuous densities also exist
eventually; neither their existence nor a derivative-error estimate
is added to the original `edgeworth_three` hypotheses.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- Cubic weighted Fourier inversion bounds the actual density after its signed correction.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, §3.1
`eq:psi`, and §5.4 `eq:higher-error-goal`, orders zero and three. -/
theorem continuous_density_sub_gaussian_cubic_weighted_abs_le
    (σ μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure σ] [IsProbabilityMeasure μ]
    (hI : Integrable (characteristic2 σ)) (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (hmom : MemLp id 3 μ) (r : ℝ)
    (hf : ContDiff ℝ 3 (fun η => characteristic2 σ η - characteristic2 stdGauss2 η -
      (r : ℂ) * characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η))
    (hJ : ∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖iteratedFDeriv ℝ k (fun η =>
      characteristic2 σ η - characteristic2 stdGauss2 η -
        (r : ℂ) * characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖))
    {q : ℝ × ℝ → ℝ} (hq : Continuous q) (hq0 : ∀ z, 0 ≤ q z)
    (hσ : σ = volume.withDensity (fun z => ENNReal.ofReal (q z))) (x : ℝ × ℝ) :
    (1 + eucl x ^ 3) * |q x - phi2 x - r * psiOf μ x| ≤ ((2 * Real.pi) ^ 2)⁻¹ *
      ((∫ ξ, ‖characteristic2 σ ξ - characteristic2 stdGauss2 ξ -
        (r : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) +
      ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 3 *
        ∫ ξ, ‖iteratedFDeriv ℝ 3 (fun η => characteristic2 σ η - characteristic2 stdGauss2 η -
          (r : ℂ) * characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) := by
  let f := fun η => characteristic2 σ η - characteristic2 stdGauss2 η -
    (r : ℂ) * characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η
  calc
    _ ≤ (1 + eucl x ^ 3) * ‖𝓕 (fun ξ : ℂ => f ((2 * Real.pi) • Complex.equivRealProdCLM ξ))
        (Complex.measurableEquivRealProd.symm x)‖ := mul_le_mul_of_nonneg_left
      (continuous_density_sub_gaussian_cubic_abs_le_norm_fourier σ μ hI hμ hexp hmom r hq hq0 hσ x)
      (by unfold eucl; positivity)
    _ ≤ _ := weighted_norm_fourier_realProd_le f hf hJ x

example : IsProbabilityMeasure stdGauss2 ∧ Integrable (characteristic2 stdGauss2) ∧
    IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    Continuous phi2 ∧ (∀ z, 0 ≤ phi2 z) ∧
    stdGauss2 = volume.withDensity (fun z => ENNReal.ofReal (phi2 z)) ∧
    let f := fun η => characteristic2 stdGauss2 η - characteristic2 stdGauss2 η -
      (0 : ℂ) * characteristic2 stdGauss2 η * cubicCharacteristicCorrection stdGauss2 η
    ContDiff ℝ 3 f ∧ ∀ k : ℕ, Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖) := by
  refine ⟨inferInstance, integrable_characteristic2_stdGauss2, isStandardized_stdGauss2,
    hasExpMoments_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    by unfold phi2; fun_prop, isDensityOf_stdGauss2.nonneg, stdGauss2_eq_withDensity_phi2, ?_, ?_⟩
  · simpa only [sub_self, zero_mul, sub_zero] using
      (contDiff_const : ContDiff ℝ 3 (fun _ : ℝ × ℝ => (0 : ℂ)))
  · intro k
    simp only [sub_self, zero_mul, iteratedFDeriv_fun_zero, Pi.zero_apply, norm_zero]
    fun_prop

/-- Every actual continuous normalized-sum density has cubic weighted corrected error bounded by a constant over n.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, §3.1
`eq:psi`, and §5.4 `eq:higher-error-goal`, with the original hypotheses. -/
theorem eventually_weighted_abs_scaledSum_density_sub_phi2_psiOf
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ,
      Continuous q → IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 3) * |q x - phi2 x - (Real.sqrt n)⁻¹ * psiOf μ x| ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨C0, C3, hC0, hC3, hinputs⟩ :=
    eventually_characteristic_scaledSum_cubic_error_fourier_inputs μ hμ hmom hexp hcf
  refine ⟨((2 * Real.pi) ^ 2)⁻¹ * (C0 + ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 3 * C3), by positivity, ?_⟩
  filter_upwards [hinputs, eventually_integrable_characteristic_scaledSum μ hcf] with n hn hIn
  intro q hq hqd x
  have h := continuous_density_sub_gaussian_cubic_weighted_abs_le _ μ hIn hμ hexp
    (hmom.mono_exponent (by norm_num)) (Real.sqrt n)⁻¹ hn.1 hn.2.1 hq hqd.nonneg hqd.map_eq x
  calc
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ *
        ((∫ ξ, ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
          characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) +
        ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 3 * ∫ ξ, ‖iteratedFDeriv ℝ 3 (fun η =>
          characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
            characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
              characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) := h
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ *
        (C0 * (n : ℝ)⁻¹ + ‖Complex.equivRealProdCLM.toContinuousLinearMap‖ ^ 3 * (C3 * (n : ℝ)⁻¹)) := by
      gcongr
      · exact hn.2.2.1
      · exact hn.2.2.2
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

/-- Actual continuous normalized-sum densities eventually exist and attain the cubic weighted inverse-n rate.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`,
including the original integrable-power condition for density existence. -/
theorem eventually_exists_scaledSum_density_weighted_gaussian_cubic_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∃ q : ℝ × ℝ → ℝ,
      Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q ∧
      ∀ x, (1 + eucl x ^ 3) * |q x - phi2 x - (Real.sqrt n)⁻¹ * psiOf μ x| ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨C, hC, herror⟩ := eventually_weighted_abs_scaledSum_density_sub_phi2_psiOf μ hμ hmom hexp hcf
  refine ⟨C, hC, ?_⟩
  filter_upwards [herror, eventually_exists_continuous_density_scaledSum μ hcf] with n hn hex
  obtain ⟨q, hq, hqd, _⟩ := hex
  exact ⟨q, hq, hqd, hn q hq hqd⟩

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

/-- The corrected density errors converge uniformly with the full cubic spatial weight.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`,
the uniform convergence consequence of the inverse-n estimate. -/
theorem eventually_weighted_abs_scaledSum_density_sub_phi2_psiOf_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ,
      Continuous q → IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, (1 + eucl x ^ 3) * |q x - phi2 x - (Real.sqrt n)⁻¹ * psiOf μ x| ≤ ε := by
  obtain ⟨C, _, herror⟩ := eventually_weighted_abs_scaledSum_density_sub_phi2_psiOf μ hμ hmom hexp hcf
  have hlim : Tendsto (fun n : ℕ => C * (n : ℝ)⁻¹) atTop (nhds 0) := by
    simpa only [mul_zero, Function.comp_def] using
      (tendsto_inv_atTop_zero.comp
        (tendsto_natCast_atTop_atTop : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop)).const_mul C
  filter_upwards [herror, hlim.eventually (gt_mem_nhds hε)] with n hn hεn
  intro q hq hqd x
  exact (hn q hq hqd x).trans hεn.le

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2, one_pos⟩

end Transformer.Modes
