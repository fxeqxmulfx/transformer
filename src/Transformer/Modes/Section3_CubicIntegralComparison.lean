import Transformer.Modes.Section3_CubicCorrectionTail
/-!
# The full integrated cubic Edgeworth characteristic error

Combining the small-frequency estimate with the three exterior bounds
proves the zero-derivative `s = 3` step of arXiv:2412.09080v3,
§5.4 `eq:higher-error-goal`. The actual normalized-sum characteristic
function is compared with its Gaussian function and the literal cubic
correction `G(ξ) B(ξ) / sqrt n`. The full norm integral is eventually
bounded by a nonnegative fixed-law constant divided by `n`.

The required inputs are standardization, a finite fourth moment,
exponential moments to identify the actual cumulants, and the paper's
integrable-characteristic-power condition. The small ball includes its
boundary; its complement is contained in the closed exterior. This
permits combining the bounds without a separate null-boundary argument.

Every term used in the integral comparison is proved integrable. The
constant retains its dependence on the fixed law, the small-frequency
radius, and finite Gaussian polynomial moments. The integral also tends
to zero as a consequence of its inverse-n rate. Identifying the inverse
spectral correction with `ψ`, and controlling third-order Fourier
derivatives for the spatial weight, are further proof steps.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The full norm integral of the actual cubic-corrected characteristic error is `O(1/n)`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:higher-error-goal`, its zero-derivative fixed-law step. -/
theorem eventually_integral_characteristic_scaledSum_sub_gaussian_cubic_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      (∫ ξ : ℝ × ℝ,
        ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
          characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨a, A, ha, hA, hsmall⟩ := exists_characteristic_scaledSum_sub_gaussian_cubic_bound μ hμ hmom
  obtain ⟨Ct, hCt, htail⟩ := eventually_integral_characteristic_scaledSum_tail_rate_inv_nat μ hcf ha
  obtain ⟨Cb, hCb, hBtail⟩ := exists_integral_scaled_gaussian_cubic_tail_rate_inv_nat μ
    (hmom.mono_exponent (by norm_num)) hexp ha
  let Cg : ℝ := (a ^ 2)⁻¹ * ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ 2 * ‖characteristic2 stdGauss2 ξ‖
  let W : ℝ × ℝ → ℝ := fun ξ => (‖ξ‖ ^ 4 + ‖ξ‖ ^ 6) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)
  have hCg : 0 ≤ Cg := by
    have hM : 0 ≤ ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ 2 * ‖characteristic2 stdGauss2 ξ‖ := integral_nonneg fun ξ => by positivity
    dsimp [Cg]; positivity
  have hIW : Integrable W := by
    convert (integrable_norm_pow_gaussian_frequency 4 (b := 1 / 8) (by norm_num)).add
      (integrable_norm_pow_gaussian_frequency 6 (b := 1 / 8) (by norm_num)) using 1
    funext ξ
    dsimp only [W, Pi.add_apply]
    have he : -(ξ.1 ^ 2 + ξ.2 ^ 2) / 8 = -(1 / 8 : ℝ) * (ξ.1 ^ 2 + ξ.2 ^ 2) := by ring
    rw [he]
    ring
  have hW0 : 0 ≤ ∫ ξ, W ξ := integral_nonneg fun ξ => by dsimp [W]; positivity
  have hIg : Integrable (fun ξ : ℝ × ℝ => ‖characteristic2 stdGauss2 ξ‖) := by
    simpa only [pow_zero, one_mul] using integrable_norm_pow_characteristic2_stdGauss2 0
  have hIb : Integrable (fun ξ : ℝ × ℝ => ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) := by
    simpa only [pow_zero, one_mul] using
      (integrable_norm_pow_gaussian_cubic_and_bound μ (hmom.mono_exponent (by norm_num)) hexp 0).1
  refine ⟨A * (∫ ξ, W ξ) + Ct + Cg + Cb, by positivity, ?_⟩
  filter_upwards [eventually_integrable_characteristic_scaledSum μ hcf,
    eventually_integrable_characteristic_scaledSum_cubic_error μ (hmom.mono_exponent (by norm_num)) hexp hcf,
    htail, eventually_ge_atTop 4] with n hISum hIerror htailn hn
  have hn1 : 1 ≤ n := by omega
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  let f : ℝ × ℝ → ℝ := fun ξ =>
    ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
      characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
        characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖
  have hIf : Integrable f := hIerror.norm
  have he (ξ : ℝ × ℝ) : ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ =
      (Real.sqrt n)⁻¹ * ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ := by
    rw [mul_assoc, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
  have hIbn : Integrable (fun ξ : ℝ × ℝ => ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
      characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) := by
    simpa only [he] using hIb.const_mul (Real.sqrt n)⁻¹
  let S : Set (ℝ × ℝ) := {ξ | ‖ξ‖ ≤ a * Real.sqrt n}
  let E : Set (ℝ × ℝ) := {ξ | a * Real.sqrt n ≤ ‖ξ‖}
  have hS : MeasurableSet S := (isClosed_le continuous_norm continuous_const).measurableSet
  have hE : MeasurableSet E := (isClosed_le continuous_const continuous_norm).measurableSet
  have hsub : Sᶜ ⊆ E := by
    intro ξ hξ
    change ¬‖ξ‖ ≤ a * Real.sqrt n at hξ
    change a * Real.sqrt n ≤ ‖ξ‖
    exact le_of_lt (lt_of_not_ge hξ)
  have hlow : (∫ ξ in S, f ξ) ≤ A * (n : ℝ)⁻¹ * ∫ ξ, W ξ := by
    calc
      _ ≤ ∫ ξ in S, (A * (n : ℝ)⁻¹) * W ξ := by
        apply setIntegral_mono_on hIf.integrableOn (hIW.const_mul _).integrableOn hS
        intro ξ hξ
        simpa only [f, W, mul_assoc] using hsmall n hn ξ hξ
      _ ≤ ∫ ξ, (A * (n : ℝ)⁻¹) * W ξ :=
        setIntegral_le_integral (hIW.const_mul _) (ae_of_all _ fun ξ => by dsimp [W]; positivity)
      _ = _ := integral_const_mul _ _
  have hhigh : (∫ ξ in Sᶜ, f ξ) ≤ (Ct + Cg + Cb) * (n : ℝ)⁻¹ := by
    calc
      _ ≤ ∫ ξ in E, f ξ := setIntegral_mono_set hIf.integrableOn
        (ae_of_all _ fun ξ => norm_nonneg _) (ae_of_all _ fun ξ hξ => hsub hξ)
      _ ≤ ∫ ξ in E,
          (‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖ +
            ‖characteristic2 stdGauss2 ξ‖) +
              ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ := by
        apply setIntegral_mono_on hIf.integrableOn ((hISum.norm.add hIg).add hIbn).integrableOn hE
        intro ξ _
        dsimp [f]
        exact (norm_sub_le _ _).trans (add_le_add (norm_sub_le _ _) le_rfl)
      _ = (∫ ξ in E, ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖) +
          (∫ ξ in E, ‖characteristic2 stdGauss2 ξ‖) +
            ∫ ξ in E, ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ := by
        rw [integral_add (f := fun ξ : ℝ × ℝ =>
          ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖ + ‖characteristic2 stdGauss2 ξ‖)
          (hISum.norm.add hIg).integrableOn hIbn.integrableOn,
          integral_add (f := fun ξ : ℝ × ℝ =>
            ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ‖) hISum.norm.integrableOn hIg.integrableOn]
      _ ≤ Ct * (n : ℝ)⁻¹ + Cg * (n : ℝ)⁻¹ + Cb * (n : ℝ)⁻¹ :=
        add_le_add (add_le_add htailn (integral_characteristic2_stdGauss2_tail_le_inv_nat ha hn1)) (hBtail n hn1)
      _ = _ := by ring
  change (∫ ξ, f ξ) ≤ _
  rw [← integral_add_compl hS hIf]
  calc
    _ ≤ A * (n : ℝ)⁻¹ * (∫ ξ, W ξ) + (Ct + Cg + Cb) * (n : ℝ)⁻¹ := add_le_add hlow hhigh
    _ = _ := by ring


example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

/-- The actual cubic-corrected characteristic error tends to zero in the norm integral.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, the qualitative
consequence of the proved zero-derivative inverse-n rate. -/
theorem tendsto_integral_characteristic_scaledSum_cubic_error_zero
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    Tendsto (fun n : ℕ => ∫ ξ : ℝ × ℝ,
      ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
        characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
          characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) atTop (nhds 0) := by
  obtain ⟨C, _, hB⟩ := eventually_integral_characteristic_scaledSum_sub_gaussian_cubic_rate μ hμ hmom hexp hcf
  have hlim : Tendsto (fun n : ℕ => C * (n : ℝ)⁻¹) atTop (nhds 0) := by
    simpa only [mul_zero, Function.comp_def] using
      (tendsto_inv_atTop_zero.comp
        (tendsto_natCast_atTop_atTop : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop)).const_mul C
  refine squeeze_zero' ?_ hB hlim
  filter_upwards [] with n
  exact integral_nonneg fun ξ => norm_nonneg _

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
