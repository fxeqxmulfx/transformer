import Transformer.Modes.Section3_DensityLimit

/-!
# Density inversion for the actual probability law

The continuous-density argument in arXiv:2412.09080v3, §5.5, is supplied
here for any probability law with an integrable characteristic function.
The candidate is the literal inverse integral, already proved continuous,
real, and nonnegative as a limit of actual Gaussian smoothings.

The preceding module proves total mass at most one by Fatou's lemma
and hence integrability.
Both the normalized characteristic function and its Fourier transform
are therefore integrable. Continuous Fourier inversion applies to the
characteristic function itself. It identifies the characteristic function
of the finite measure defined by the candidate density with the original
one. Uniqueness of characteristic functions then proves equality of the
actual measures, and their total mass is exactly one.

In particular the proof neither assumes a density nor infers Fourier
integrability merely from boundedness of a density. The source's latter
inference in §5.4 is invalid; integrability is provided by the separately
proved actual characteristic-function estimates. The examples use those
estimates for the actual normalized sum, without an unproved premise.

Transporting the density through the volume-preserving coordinate
equivalence returns a bounded continuous density on the paper's `ℝ²`.
For the actual `lawY β t` normalized sum the corrected sufficient threshold
is `n > 4(β + 1)`, for every positive bandwidth and fixed translation.
The source's fixed-five-summands claim is false at large bandwidth, so this
positive bandwidth-dependent result is named and documented separately.
Every law satisfying `HasIntegrableCharFun` also has these densities
eventually; the weighted Edgeworth remainder estimates remain separate.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped FourierTransform Topology ENNReal

namespace Transformer.Modes

/-- The inverse integral is a density of the actual probability law.
Its proved integrability permits continuous Fourier inversion of the
normalized characteristic function. The resulting characteristic-function
equality identifies the finite measures. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem measure_eq_withDensity_inverseCharacteristic (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) :
    μ = volume.withDensity (fun w => ENNReal.ofReal (inverseCharacteristic μ w)) := by
  let ν : Measure ℂ := volume.withDensity (fun w => ENNReal.ofReal (inverseCharacteristic μ w))
  have hqI := integrable_inverseCharacteristic μ hI
  let : IsFiniteMeasure ν := isFiniteMeasure_withDensity_ofReal hqI.hasFiniteIntegral
  have hψI := integrable_normalizedCharacteristic μ hI
  have hψc : Continuous (normalizedCharacteristic μ) := by
    unfold normalizedCharacteristic
    fun_prop
  have hQI : Integrable (𝓕 (normalizedCharacteristic μ)) := by
    have he : 𝓕 (normalizedCharacteristic μ) = fun w => (inverseCharacteristic μ w : ℂ) := by
      funext w
      exact (ofReal_inverseCharacteristic μ hI w).symm
    rw [he]
    exact hqI.ofReal
  have hinv := hψc.fourierInv_fourier_eq hψI hQI
  apply Measure.ext_of_charFun
  funext ξ
  have h := congrFun hinv ((2 * Real.pi)⁻¹ • ξ)
  have he : normalizedCharacteristic μ ((2 * Real.pi)⁻¹ • ξ) = charFun μ ξ := by
    unfold normalizedCharacteristic
    rw [smul_smul, mul_inv_cancel₀ (by positivity : (2 * Real.pi : ℝ) ≠ 0), one_smul]
  rw [he] at h
  rw [← h, Real.fourierInv_eq', charFun_apply,
    integral_withDensity_eq_integral_toReal_smul
      (f := fun w => ENNReal.ofReal (inverseCharacteristic μ w))
      (ENNReal.continuous_ofReal.measurable.comp (continuous_inverseCharacteristic μ hI).measurable)
      (ae_of_all _ fun w => ENNReal.ofReal_lt_top)]
  apply integral_congr_ae
  exact ae_of_all _ fun w => by
    dsimp only
    rw [ofReal_inverseCharacteristic μ hI w |>.symm,
      ENNReal.toReal_ofReal (inverseCharacteristic_nonneg μ hI w)]
    rw [inner_smul_right]
    have hh : 2 * Real.pi * ((2 * Real.pi)⁻¹ * inner ℝ w ξ) = inner ℝ w ξ := by
      rw [← mul_assoc, mul_inv_cancel₀ (by positivity : (2 * Real.pi : ℝ) ≠ 0), one_mul]
    rw [hh, Complex.real_smul]
    simp only [smul_eq_mul]
    ring

example := measure_eq_withDensity_inverseCharacteristic _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

/-- The density has mass exactly one, by its proved measure identity
with the original probability law. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem integral_inverseCharacteristic (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) : ∫ w, inverseCharacteristic μ w = 1 := by
  have hm := congrArg (fun ν : Measure ℂ => ν Set.univ)
    (measure_eq_withDensity_inverseCharacteristic μ hI)
  rw [measure_univ, withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ,
    ← ofReal_integral_eq_lintegral_ofReal (integrable_inverseCharacteristic μ hI)
      (ae_of_all _ (inverseCharacteristic_nonneg μ hI))] at hm
  have h := congrArg ENNReal.toReal hm
  simpa only [ENNReal.toReal_one, ENNReal.toReal_ofReal
    (integral_nonneg (inverseCharacteristic_nonneg μ hI))] using h.symm

example := integral_inverseCharacteristic _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

/-- An integrable literal characteristic function gives a bounded
continuous nonnegative density of the actual law on `ℝ²`. The density is
constructed by inversion in complex coordinates and transported through
the volume-preserving equivalence. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem exists_bounded_continuous_density_of_integrable_characteristic2
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hI : Integrable (characteristic2 μ)) :
    ∃ q : ℝ × ℝ → ℝ, Continuous q ∧ (∀ z, 0 ≤ q z) ∧
      μ = volume.withDensity (fun z => ENNReal.ofReal (q z)) ∧
        ∃ C : ℝ, ∀ z, |q z| ≤ C := by
  have hC := integrable_charFun_complexLaw μ hI
  let q : ℝ × ℝ → ℝ := fun z => inverseCharacteristic (complexLaw μ)
    (Complex.measurableEquivRealProd.symm z)
  have hq : Continuous q :=
    (continuous_inverseCharacteristic (complexLaw μ) hC).comp
      Complex.equivRealProdCLM.symm.continuous
  refine ⟨q, hq, fun z => inverseCharacteristic_nonneg (complexLaw μ) hC _, ?_,
    ∫ ξ, ‖normalizedCharacteristic (complexLaw μ) ξ‖, fun z => inverseCharacteristic_abs_le _ _⟩
  rw [← complexLaw_map_realProd μ, measure_eq_withDensity_inverseCharacteristic (complexLaw μ) hC,
    map_realProd_withDensity _ (continuous_inverseCharacteristic (complexLaw μ) hC).measurable]

example := exists_bounded_continuous_density_of_integrable_characteristic2 _
  (integrable_characteristic_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

/-- The actual `n^(-1/2) Σ Yᵢ(t)` has a bounded continuous density
for `β > 0` and `n > 4(β + 1)`. The source asserts the fixed threshold
of five summands, which is false at large bandwidth. This is an explicitly
corrected sufficient threshold derived from the actual Fourier bound,
with the original normalized sum and arbitrary fixed translation retained.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem exists_continuous_density_scaledSum_lawY {β : ℝ} (hβ : 0 < β) (t : ℝ)
    {n : ℕ} (hn : 4 * (β + 1) < (n : ℝ)) :
    ∃ q, Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => lawY β t) (scaledSum n) q ∧
      ∃ C : ℝ, ∀ z, |q z| ≤ C := by
  obtain ⟨q, hq, hnq, heq, hb⟩ :=
    exists_bounded_continuous_density_of_integrable_characteristic2
      ((Measure.pi fun _ : Fin n => lawY β t).map (scaledSum n))
      (integrable_characteristic_scaledSum_lawY hβ t hn)
  exact ⟨q, hq, ⟨hq.measurable, hnq, heq⟩, hb⟩

example := exists_continuous_density_scaledSum_lawY (n := 17)
  (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)

/-- Any probability law satisfying the source's integrable-characteristic
power condition has bounded continuous densities of its normalized sums
for all sufficiently large `n`. This constructs those densities without
standardization or moment hypotheses. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem eventually_exists_continuous_density_scaledSum
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop, ∃ q,
      Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q ∧
        ∃ C : ℝ, ∀ z, |q z| ≤ C := by
  filter_upwards [eventually_integrable_characteristic_scaledSum μ hcf] with n hn
  obtain ⟨q, hq, hnq, heq, hb⟩ :=
    exists_bounded_continuous_density_of_integrable_characteristic2
      ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) hn
  exact ⟨q, hq, ⟨hq.measurable, hnq, heq⟩, hb⟩

example := eventually_exists_continuous_density_scaledSum stdGauss2
  hasIntegrableCharFun_stdGauss2

end Transformer.Modes
