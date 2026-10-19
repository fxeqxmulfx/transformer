module

public import Transformer.Modes.Section3_CubicDerivativeFrequency
public import Transformer.Modes.Section3_CubicDerivativeTail
public import Transformer.Modes.Section3_CubicIntegralComparison

/-!
# The full integral of the actual cubic-corrected third derivative

The local and exterior estimates now prove the third-derivative
step of arXiv:2412.09080v3, §5.4 `eq:higher-error-goal` for
`s = 3`. On the small-frequency ball, the actual derivative is
bounded by `C/n` times an integrable Gaussian polynomial. On its
complement, the proved closed-exterior estimate applies.

The boundary requires no separate measure-zero argument: the
complement of the closed small ball is contained in the closed
exterior. Both integrals are of the actual derivative norm, whose
integrability was proved from the original characteristic-power
condition. A finite fourth moment gives the local Taylor estimate.

The final result assembles all inputs for cubic Fourier inversion:
three continuous derivatives, their four integrable norms, and the
inverse-n bounds for orders zero and three. Exponential moments
enter only the existing zero-order comparison and the subsequent
identification of the inverse cubic coefficient with `psiOf`.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The full norm integral of the actual corrected third derivative is eventually bounded by a constant over n.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`,
`eq:small-z`, and the exterior estimates, at derivative order three. -/
theorem eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      Integrable (fun ξ => ‖iteratedFDeriv ℝ 3 (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) ∧
      (∫ ξ : ℝ × ℝ, ‖iteratedFDeriv ℝ 3 (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨a, A, ha, _, hA, hsmall⟩ :=
    exists_iteratedFDeriv_characteristic_scaledSum_cubic_error_gaussian_bound μ hμ hmom
  obtain ⟨Ct, hCt, htail⟩ :=
    eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_tail μ
      (hmom.mono_exponent (by norm_num)) hcf ha
  let W := fun ξ : ℝ × ℝ => (1 + ‖ξ‖) ^ 13 * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)
  have hIW : Integrable W := by
    convert integrable_one_add_norm_pow_gaussian_frequency 13 (b := 1 / 8) (by norm_num) using 1
    funext ξ
    dsimp only [W]
    congr 2
    ring
  have hW0 : 0 ≤ ∫ ξ, W ξ := integral_nonneg fun ξ => by dsimp [W]; positivity
  refine ⟨A * (∫ ξ, W ξ) + Ct, by positivity, ?_⟩
  filter_upwards [htail, eventually_ge_atTop 16] with n hT hn
  let f := fun ξ => ‖iteratedFDeriv ℝ 3 (fun η =>
    characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
      characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
        characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖
  have hIf : Integrable f := hT.1
  let S : Set (ℝ × ℝ) := {ξ | ‖ξ‖ ≤ a * Real.sqrt n}
  let E : Set (ℝ × ℝ) := {ξ | a * Real.sqrt n ≤ ‖ξ‖}
  have hS : MeasurableSet S := (isClosed_le continuous_norm continuous_const).measurableSet
  have hsub : Sᶜ ⊆ E := by
    intro ξ hξ
    change ¬‖ξ‖ ≤ a * Real.sqrt n at hξ
    exact le_of_lt (lt_of_not_ge hξ)
  have hlow : (∫ ξ in S, f ξ) ≤ A * (n : ℝ)⁻¹ * ∫ ξ, W ξ := by
    calc
      _ ≤ ∫ ξ in S, (A * (n : ℝ)⁻¹) * W ξ := by
        apply setIntegral_mono_on hIf.integrableOn (hIW.const_mul _).integrableOn hS
        intro ξ hξ
        simpa only [f, W, mul_assoc] using hsmall n hn ξ hξ 3 le_rfl
      _ ≤ ∫ ξ, (A * (n : ℝ)⁻¹) * W ξ :=
        setIntegral_le_integral (hIW.const_mul _) (ae_of_all _ fun ξ => by dsimp [W]; positivity)
      _ = _ := integral_const_mul _ _
  have hhigh : (∫ ξ in Sᶜ, f ξ) ≤ Ct * (n : ℝ)⁻¹ := by
    calc
      _ ≤ ∫ ξ in E, f ξ := setIntegral_mono_set hIf.integrableOn
        (ae_of_all _ fun ξ => norm_nonneg _) (ae_of_all _ fun ξ hξ => hsub hξ)
      _ ≤ _ := hT.2
  refine ⟨hIf, ?_⟩
  change (∫ ξ, f ξ) ≤ _
  rw [← integral_add_compl hS hIf]
  calc
    _ ≤ A * (n : ℝ)⁻¹ * (∫ ξ, W ξ) + Ct * (n : ℝ)⁻¹ := add_le_add hlow hhigh
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

/-- The full actual cubic-corrected third derivative tends to zero in the norm integral.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal`,
the convergence consequence of its proved inverse-n rate. -/
theorem tendsto_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_zero
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hcf : HasIntegrableCharFun μ) :
    Tendsto (fun n : ℕ => ∫ ξ : ℝ × ℝ,
      ‖iteratedFDeriv ℝ 3 (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) atTop (nhds 0) := by
  obtain ⟨C, _, hrate⟩ :=
    eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_rate μ hμ hmom hcf
  have hlim : Tendsto (fun n : ℕ => C * (n : ℝ)⁻¹) atTop (nhds 0) := by
    simpa only [mul_zero, Function.comp_def] using
      (tendsto_inv_atTop_zero.comp
        (tendsto_natCast_atTop_atTop : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop)).const_mul C
  apply squeeze_zero' (Eventually.of_forall fun n => integral_nonneg fun ξ => norm_nonneg _) _ hlim
  filter_upwards [hrate] with n hn
  exact hn.2

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2⟩

/-- All analytic inputs for cubic weighted Fourier inversion follow from the original fixed-law hypotheses.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and
§5.4 `eq:higher-error-goal`, assembling orders zero through three. -/
theorem eventually_characteristic_scaledSum_cubic_error_fourier_inputs
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C0 C3 : ℝ, 0 ≤ C0 ∧ 0 ≤ C3 ∧ ∀ᶠ n : ℕ in atTop,
      let f := fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η
      ContDiff ℝ 3 f ∧
        (∀ k : ℕ, k ≤ 3 → Integrable (fun ξ => ‖iteratedFDeriv ℝ k f ξ‖)) ∧
        (∫ ξ, ‖f ξ‖) ≤ C0 * (n : ℝ)⁻¹ ∧
        (∫ ξ, ‖iteratedFDeriv ℝ 3 f ξ‖) ≤ C3 * (n : ℝ)⁻¹ := by
  have hmom3 : MemLp id 3 μ := hmom.mono_exponent (by norm_num)
  obtain ⟨C0, hC0, hzero⟩ := eventually_integral_characteristic_scaledSum_sub_gaussian_cubic_rate μ hμ hmom hexp hcf
  obtain ⟨C3, hC3, hthird⟩ :=
    eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_rate μ hμ hmom hcf
  refine ⟨C0, C3, hC0, hC3, ?_⟩
  filter_upwards [hzero, hthird,
    eventually_integrable_norm_iteratedFDeriv_characteristic_scaledSum_cubic_error μ hmom3 hcf]
    with n h0 h3 hI
  exact ⟨contDiff_characteristic_scaledSum_cubic_error μ hmom3 n, hI, h0, h3.2⟩

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
