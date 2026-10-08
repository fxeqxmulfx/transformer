import Transformer.Modes.Section3_DerivativeTailRate
import Transformer.Modes.Section3_SecondDerivativeFrequency
/-!
# Global integrals of Gaussian characteristic-derivative errors

The first two derivative versions of arXiv:2412.09080v3, §5.4
`eq:higher-error-goal`, follow by combining the local Gaussian majorants
of `eq:br-9.10` with the proved geometric exterior rates of `eq:big-z-exp`.
Every derivative in the difference is integrable for sufficiently large
sample counts under the original integrable-characteristic-power condition.

The polynomial times Gaussian majorants have finite integrals over the
entire frequency space. Splitting into the growing ball and its complement
then gives the rate `1 / sqrt n` for the full derivative-error integrals.
No boundary contribution or unproved input is assumed; the abstract
integral comparison is instantiated with the actual characteristic functions.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Integrating a local majorant and both exterior tails controls the whole error.
Source: arXiv:2412.09080v3, §5.4 `eq:small-z`, `eq:big-z-exp`,
and `eq:big-phi-poly`, the frequency-region split. -/
theorem integral_norm_sub_le_of_local_and_tail {F : Type*} [NormedAddCommGroup F]
    (f g : ℝ × ℝ → F) (hIf : Integrable f) (hIg : Integrable g)
    (P : ℝ × ℝ → ℝ) (hIP : Integrable P) (hP : ∀ ξ, 0 ≤ P ξ)
    {a A B D : ℝ} (hA : 0 ≤ A) (n : ℕ)
    (hlow : ∀ ξ : ℝ × ℝ, ‖ξ‖ ≤ a * Real.sqrt n → ‖f ξ - g ξ‖ ≤ A * (Real.sqrt n)⁻¹ * P ξ)
    (hTf : (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, ‖f ξ‖) ≤ B * (Real.sqrt n)⁻¹)
    (hTg : (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖}, ‖g ξ‖) ≤ D * (Real.sqrt n)⁻¹) :
    (∫ ξ, ‖f ξ - g ξ‖) ≤ (A * (∫ ξ, P ξ) + B + D) * (Real.sqrt n)⁻¹ := by
  have hIe : Integrable (fun ξ => ‖f ξ - g ξ‖) := (hIf.sub hIg).norm
  let S : Set (ℝ × ℝ) := {ξ | ‖ξ‖ ≤ a * Real.sqrt n}
  let E : Set (ℝ × ℝ) := {ξ | a * Real.sqrt n ≤ ‖ξ‖}
  have hS : MeasurableSet S := (isClosed_le continuous_norm continuous_const).measurableSet
  have hE : MeasurableSet E := (isClosed_le continuous_const continuous_norm).measurableSet
  have hsub : Sᶜ ⊆ E := by
    intro ξ hξ
    change ¬‖ξ‖ ≤ a * Real.sqrt n at hξ
    exact le_of_lt (lt_of_not_ge hξ)
  have hlowI : (∫ ξ in S, ‖f ξ - g ξ‖) ≤ A * (Real.sqrt n)⁻¹ * ∫ ξ, P ξ := by
    calc
      _ ≤ ∫ ξ in S, (A * (Real.sqrt n)⁻¹) * P ξ := by
        apply setIntegral_mono_on hIe.integrableOn (hIP.const_mul _).integrableOn hS
        exact fun ξ hξ => hlow ξ hξ
      _ ≤ ∫ ξ, (A * (Real.sqrt n)⁻¹) * P ξ :=
        setIntegral_le_integral (hIP.const_mul _) (ae_of_all _ fun ξ => mul_nonneg (by positivity) (hP ξ))
      _ = _ := integral_const_mul _ _
  have hhighI : (∫ ξ in Sᶜ, ‖f ξ - g ξ‖) ≤ (B + D) * (Real.sqrt n)⁻¹ := by
    calc
      _ ≤ ∫ ξ in E, ‖f ξ - g ξ‖ := setIntegral_mono_set hIe.integrableOn
        (ae_of_all _ fun ξ => norm_nonneg _) (ae_of_all _ fun ξ hξ => hsub hξ)
      _ ≤ ∫ ξ in E, ‖f ξ‖ + ‖g ξ‖ := by
        apply setIntegral_mono_on hIe.integrableOn (hIf.norm.add hIg.norm).integrableOn hE
        exact fun ξ _ => norm_sub_le _ _
      _ = (∫ ξ in E, ‖f ξ‖) + ∫ ξ in E, ‖g ξ‖ := integral_add hIf.norm.integrableOn hIg.norm.integrableOn
      _ ≤ B * (Real.sqrt n)⁻¹ + D * (Real.sqrt n)⁻¹ := add_le_add hTf hTg
      _ = _ := by ring
  rw [← integral_add_compl hS hIe]
  exact (add_le_add hlowI hhighI).trans_eq (by ring)

example : (∫ ξ : ℝ × ℝ, ‖characteristic2 stdGauss2 ξ - characteristic2 stdGauss2 ξ‖) ≤
    2 * ∫ ξ : ℝ × ℝ in {ξ | 1 ≤ ‖ξ‖}, ‖characteristic2 stdGauss2 ξ‖ := by
  have hI := integrable_characteristic2_stdGauss2
  let B : ℝ := ∫ ξ : ℝ × ℝ in {ξ | 1 ≤ ‖ξ‖}, ‖characteristic2 stdGauss2 ξ‖
  have h := integral_norm_sub_le_of_local_and_tail (characteristic2 stdGauss2)
    (characteristic2 stdGauss2) hI hI (fun ξ => ‖characteristic2 stdGauss2 ξ‖) hI.norm
    (fun ξ => norm_nonneg _) (a := 1) (A := 0) (B := B) (D := B) (by norm_num) 1
    (by intro ξ hξ; simp) (by simp [B]) (by simp [B])
  simpa [B, two_mul] using h

/-- The global first-derivative Gaussian error integral is `O(1 / sqrt n)`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4
`eq:higher-error-goal`, derivative order one, under the original hypotheses. -/
theorem eventually_integral_norm_fderiv_characteristic_scaledSum_sub_gaussian
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      Integrable (fun ξ => fderiv ℝ (characteristic2
          ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ - fderiv ℝ (characteristic2 stdGauss2) ξ) ∧
      (∫ ξ, ‖fderiv ℝ (characteristic2
          ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ - fderiv ℝ (characteristic2 stdGauss2) ξ‖) ≤
        C * (Real.sqrt n)⁻¹ := by
  obtain ⟨a, A, ha, hA, hlow⟩ := exists_fderiv_characteristic_scaledSum_sub_gaussian_bound μ hμ hmom
  obtain ⟨B, hB, hTf⟩ := eventually_integral_fderiv_characteristic_scaledSum_tail_rate μ hμ.memLp hcf ha
  obtain ⟨D, hD, hTg⟩ := eventually_integral_characteristic2_stdGauss2_derivative_tail_rate ha
  let P : ℝ × ℝ → ℝ := fun ξ => (‖ξ‖ ^ 2 + ‖ξ‖ ^ 4) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)
  have hIP : Integrable P := by
    have h2 := integrable_norm_pow_gaussian_frequency 2 (b := 1 / 8) (by norm_num)
    have h4 := integrable_norm_pow_gaussian_frequency 4 (b := 1 / 8) (by norm_num)
    convert h2.add h4 using 1
    funext ξ
    dsimp [P]
    have he : Real.exp (-(1 / 8) * (ξ.1 ^ 2 + ξ.2 ^ 2)) =
        Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by congr 1; ring
    rw [he]
    ring
  have hP : ∀ ξ, 0 ≤ P ξ := by intro ξ; dsimp [P]; positivity
  have hPI : 0 ≤ ∫ ξ, P ξ := integral_nonneg hP
  refine ⟨A * (∫ ξ, P ξ) + B + D, by positivity, ?_⟩
  filter_upwards [hTf, hTg, eventually_ge_atTop 4] with n hnF hnG hn
  have hIG := integrable_characteristic2_stdGauss2_derivatives.1
  refine ⟨hnF.1.sub hIG, ?_⟩
  apply integral_norm_sub_le_of_local_and_tail _ _ hnF.1 hIG P hIP hP hA n
    (by intro ξ hξ; simpa only [P, mul_assoc] using hlow n hn ξ hξ) hnF.2
  exact (le_add_of_nonneg_right (integral_nonneg fun ξ => norm_nonneg _)).trans hnG

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

/-- The global second-derivative Gaussian error integral is `O(1 / sqrt n)`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4
`eq:higher-error-goal`, derivative order two, under the original hypotheses. -/
theorem eventually_integral_norm_iteratedFDeriv_two_characteristic_scaledSum_sub_gaussian
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      Integrable (fun ξ => iteratedFDeriv ℝ 2 (characteristic2
          ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ - iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ) ∧
      (∫ ξ, ‖iteratedFDeriv ℝ 2 (characteristic2
          ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ - iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖) ≤
        C * (Real.sqrt n)⁻¹ := by
  obtain ⟨a, A, ha, hA, hlow⟩ := exists_iteratedFDeriv_two_characteristic_scaledSum_sub_gaussian_bound μ hμ hmom
  obtain ⟨B, hB, hTf⟩ := eventually_integral_iteratedFDeriv_two_characteristic_scaledSum_tail_rate μ hμ.memLp hcf ha
  obtain ⟨D, hD, hTg⟩ := eventually_integral_characteristic2_stdGauss2_derivative_tail_rate ha
  let P : ℝ × ℝ → ℝ := fun ξ => (‖ξ‖ + ‖ξ‖ ^ 3 + ‖ξ‖ ^ 5) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)
  have hIP : Integrable P := by
    have h1 := integrable_norm_pow_gaussian_frequency 1 (b := 1 / 8) (by norm_num)
    have h3 := integrable_norm_pow_gaussian_frequency 3 (b := 1 / 8) (by norm_num)
    have h5 := integrable_norm_pow_gaussian_frequency 5 (b := 1 / 8) (by norm_num)
    convert (h1.add h3).add h5 using 1
    funext ξ
    dsimp [P]
    have he : Real.exp (-(1 / 8) * (ξ.1 ^ 2 + ξ.2 ^ 2)) =
        Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by congr 1; ring
    rw [he]
    ring
  have hP : ∀ ξ, 0 ≤ P ξ := by intro ξ; dsimp [P]; positivity
  have hPI : 0 ≤ ∫ ξ, P ξ := integral_nonneg hP
  refine ⟨A * (∫ ξ, P ξ) + B + D, by positivity, ?_⟩
  filter_upwards [hTf, hTg, eventually_ge_atTop 6] with n hnF hnG hn
  have hIG := integrable_characteristic2_stdGauss2_derivatives.2
  refine ⟨hnF.1.sub hIG, ?_⟩
  apply integral_norm_sub_le_of_local_and_tail _ _ hnF.1 hIG P hIP hP hA n
    (by intro ξ hξ; simpa only [P, mul_assoc] using hlow n hn ξ hξ) hnF.2
  exact (le_add_of_nonneg_left (integral_nonneg fun ξ => norm_nonneg _)).trans hnG

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
