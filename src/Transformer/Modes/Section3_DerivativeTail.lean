import Transformer.Modes.Section3_DerivativeMomentBounds
import Transformer.Modes.Section3_GaussianDensity
/-!
# Integrable exterior tails of normalized-sum derivatives

The derivative version of arXiv:2412.09080v3, §5.4 `eq:big-z-exp`,
retains a characteristic norm power `ν` that is genuinely integrable.
The remaining `n - j - ν` factors supply the exterior damping for
derivative order `j`. A general continuous norm-majorized function is
first treated, then the proved first and second sum derivatives are used.

Finite second moment gives the derivative bounds. The hypothesis on an
integrable characteristic power supplies integrability over the entire
frequency space, including its infinite-volume exterior. The exact
frequency Jacobian is `n` in dimension two. Dependence on the law and
the exterior gap is explicit; the source's inconsistent block scaling
is replaced by the actual normalized-sum characteristic formula.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Retaining an integrable characteristic power justifies the exterior norm integral.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, the norm-majorant
step after `j` derivatives, with the exact two-dimensional Jacobian. -/
theorem integrable_power_majorant_and_tail {F : Type*} [NormedAddCommGroup F]
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {ν j n : ℕ} (hν : 1 ≤ ν)
    (hn : ν + j ≤ n) (hI : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν))
    (f : ℝ × ℝ → F) (hf : Continuous f) {A : ℝ} (hA : 0 ≤ A)
    (hbound : ∀ ξ, ‖f ξ‖ ≤ A * ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ (n - j))
    {a ε : ℝ} (hε : 0 ≤ ε)
    (hgap : ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ ε) :
    Integrable f ∧ (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, ‖f ξ‖) ≤
      A * (n : ℝ) * ε ^ (n - j - ν) * ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν := by
  have hn1 : 1 ≤ n := by omega
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hIJ := (integrable_characteristic2_pow_of_le μ (show ν ≤ n - j by omega) hI).comp_smul
    (inv_ne_zero hs.ne')
  have hIf : Integrable f := (hIJ.const_mul A).mono' hf.aestronglyMeasurable (ae_of_all _ hbound)
  refine ⟨hIf, ?_⟩
  let g : ℝ × ℝ → ℝ := fun ξ =>
    A * ε ^ (n - j - ν) * ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ ν
  have hIg : Integrable g := (hI.comp_smul (inv_ne_zero hs.ne')).const_mul _
  have hS : MeasurableSet {ξ : ℝ × ℝ | a * Real.sqrt n ≤ ‖ξ‖} :=
    (isClosed_le continuous_const continuous_norm).measurableSet
  calc
    _ ≤ ∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, g ξ := by
      apply setIntegral_mono_on hIf.norm.integrableOn hIg.integrableOn hS
      intro ξ hξ
      change a * Real.sqrt n ≤ ‖ξ‖ at hξ
      have hη : a ≤ ‖(Real.sqrt (n : ℝ))⁻¹ • ξ‖ := by
        rw [norm_smul, Real.norm_of_nonneg (inv_nonneg.mpr hs.le)]
        exact (le_inv_mul_iff₀ hs).mpr (by simpa only [mul_comm] using hξ)
      have hp : ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ (n - j) =
          ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ (n - j - ν) *
            ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ ν := by
        rw [← pow_add, Nat.sub_add_cancel (show ν ≤ n - j by omega)]
      calc
        _ ≤ A * ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ (n - j) := hbound ξ
        _ ≤ g ξ := by
          rw [hp]
          dsimp [g]
          have hpow := pow_le_pow_left₀ (norm_nonneg _) (hgap _ hη) (n - j - ν)
          calc
            _ ≤ A * (ε ^ (n - j - ν) * ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ ν) := by gcongr
            _ = _ := by ring
    _ ≤ ∫ ξ : ℝ × ℝ, g ξ := setIntegral_le_integral hIg (ae_of_all _ fun ξ => by dsimp [g]; positivity)
    _ = _ := by
      dsimp [g]
      rw [integral_const_mul, integral_characteristic2_pow_scaledFrequency μ hn1 ν]
      ring

example : Integrable (fun ξ : ℝ × ℝ => characteristic2 stdGauss2 ((Real.sqrt 3)⁻¹ • ξ)) := by
  obtain ⟨ε, hε, _, hgap⟩ := characteristic2_gap_away_zero stdGauss2
    hasIntegrableCharFun_stdGauss2 (a := 1) one_pos
  have hc : Continuous (fun ξ : ℝ × ℝ => characteristic2 stdGauss2 ((Real.sqrt 3)⁻¹ • ξ)) :=
    (continuous_characteristic2 stdGauss2).comp (by fun_prop)
  exact (integrable_power_majorant_and_tail stdGauss2 (ν := 1) (j := 2) (n := 3)
    (by omega) (by omega) (by simpa using integrable_characteristic2_stdGauss2.norm)
    _ hc (A := 1) (by norm_num) (by intro ξ; simp) hε.le hgap).1

/-- The first actual sum derivative is integrable and has an exterior geometric bound.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, derivative order one.
The integrable power threshold is `ν + 1` and the frequency scale is literal. -/
theorem integrable_fderiv_characteristic_scaledSum_and_tail
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ)
    {ν n : ℕ} (hν : 1 ≤ ν) (hn : ν + 1 ≤ n)
    (hI : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν)) {a ε : ℝ} (hε : 0 ≤ ε)
    (hgap : ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ ε) :
    Integrable (fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) ∧
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖) ≤
      (n : ℝ) ^ 2 * (Real.sqrt n)⁻¹ * (2 * ∫ z, ‖z‖ ∂μ) *
        ε ^ (n - 1 - ν) * ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν := by
  have hM : 0 ≤ ∫ z, ‖z‖ ∂μ := integral_nonneg fun z => norm_nonneg z
  have h := integrable_power_majorant_and_tail μ hν (j := 1) hn hI _
    ((contDiff_characteristic_scaledSum μ hmom n).continuous_fderiv (by norm_num))
    (by positivity) (norm_fderiv_characteristic_scaledSum_le μ hmom n) hε hgap
  refine ⟨h.1, h.2.trans_eq ?_⟩
  ring

example : ∃ (ν : ℕ) (ε : ℝ), 1 ≤ ν ∧ ν + 1 ≤ ν + 1 ∧
    IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 ∧
    Integrable (fun ξ => ‖characteristic2 stdGauss2 ξ‖ ^ ν) ∧ 0 ≤ ε ∧
    ∀ ξ : ℝ × ℝ, 1 ≤ ‖ξ‖ → ‖characteristic2 stdGauss2 ξ‖ ≤ ε := by
  obtain ⟨ν, hν, hI⟩ := hasIntegrableCharFun_stdGauss2
  obtain ⟨ε, hε, _, hgap⟩ := characteristic2_gap_away_zero stdGauss2 hasIntegrableCharFun_stdGauss2 one_pos
  exact ⟨ν, ε, hν, le_rfl, inferInstance, IsGaussian.memLp_id _ _ (by simp), hI, hε.le, hgap⟩

/-- The second actual sum derivative is integrable and has an exterior geometric bound.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, derivative order two.
The integrable power threshold is `ν + 2` and both product-rule terms are bounded. -/
theorem integrable_iteratedFDeriv_two_characteristic_scaledSum_and_tail
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ)
    {ν n : ℕ} (hν : 1 ≤ ν) (hn : ν + 2 ≤ n)
    (hI : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν)) {a ε : ℝ} (hε : 0 ≤ ε)
    (hgap : ∀ ξ : ℝ × ℝ, a ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ ε) :
    Integrable (iteratedFDeriv ℝ 2
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) ∧
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ 2
          (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖) ≤
      (n : ℝ) ^ 2 * ((2 * ∫ z, ‖z‖ ∂μ) ^ 2 + 4 * ∫ z, ‖z‖ ^ 2 ∂μ) *
        ε ^ (n - 2 - ν) * ∫ ξ, ‖characteristic2 μ ξ‖ ^ ν := by
  have hM2 : 0 ≤ ∫ z, ‖z‖ ^ 2 ∂μ := integral_nonneg fun z => sq_nonneg _
  have h := integrable_power_majorant_and_tail μ hν (j := 2) hn hI _
    ((contDiff_characteristic_scaledSum μ hmom n).continuous_iteratedFDeriv (by norm_num))
    (by positivity) (norm_iteratedFDeriv_two_characteristic_scaledSum_le μ hmom (by omega)) hε hgap
  refine ⟨h.1, h.2.trans_eq ?_⟩
  ring

example : ∃ (ν : ℕ) (ε : ℝ), 1 ≤ ν ∧ ν + 2 ≤ ν + 2 ∧
    IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 ∧
    Integrable (fun ξ => ‖characteristic2 stdGauss2 ξ‖ ^ ν) ∧ 0 ≤ ε ∧
    ∀ ξ : ℝ × ℝ, 1 ≤ ‖ξ‖ → ‖characteristic2 stdGauss2 ξ‖ ≤ ε := by
  obtain ⟨ν, hν, hI⟩ := hasIntegrableCharFun_stdGauss2
  obtain ⟨ε, hε, _, hgap⟩ := characteristic2_gap_away_zero stdGauss2 hasIntegrableCharFun_stdGauss2 one_pos
  exact ⟨ν, ε, hν, le_rfl, inferInstance, IsGaussian.memLp_id _ _ (by simp), hI, hε.le, hgap⟩

/-- The original integrable-characteristic-power condition eventually gives all
Fourier derivatives needed for the spatial weight of order two. Source:
arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4 `eq:higher-error-goal`.
The derivatives are integrable at the actual threshold `ν + 2`. -/
theorem eventually_integrable_characteristic_scaledSum_derivatives
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ)
    (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop,
      Integrable (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ∧
      Integrable (fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) ∧
      Integrable (iteratedFDeriv ℝ 2 (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)))) := by
  obtain ⟨ν, hν, hI⟩ := hcf
  filter_upwards [eventually_ge_atTop (ν + 2)] with n hn
  have hgap : ∀ ξ : ℝ × ℝ, 0 ≤ ‖ξ‖ → ‖characteristic2 μ ξ‖ ≤ (1 : ℝ) :=
    fun ξ _ => norm_characteristic2_le_one μ ξ
  refine ⟨integrable_characteristic_scaledSum_of_power μ (by omega)
    (integrable_characteristic2_pow_of_le μ (by omega) hI), ?_, ?_⟩
  · exact (integrable_fderiv_characteristic_scaledSum_and_tail μ hmom hν (by omega) hI zero_le_one hgap).1
  · exact (integrable_iteratedFDeriv_two_characteristic_scaledSum_and_tail μ hmom hν (by omega) hI zero_le_one hgap).1

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
