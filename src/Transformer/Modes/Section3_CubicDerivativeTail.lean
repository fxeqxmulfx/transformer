import Transformer.Modes.Section3_CubicErrorRegularity
import Transformer.Modes.Section3_CubicDerivativeTailRates

/-!
# Integral envelopes for the cubic-corrected third derivative

The local envelope in arXiv:2412.09080v3, §5.4 `eq:small-z`,
contains a polynomial times Gaussian decay. Adding one to the
frequency norm preserves its integrability at every polynomial
degree. This accommodates the common envelope of all four actual
derivative orders without a singularity at zero.

Outside the small-frequency ball, `eq:big-z-exp`, `eq:big-phi-poly`,
and `eq:big-z-poly` bound the three terms of the literal corrected
third derivative. Their already proved exterior inverse-n rates
are now combined. The scaled cubic term retains its actual factor
`1 / sqrt n`, which is at most one for positive sample counts.

The conclusion includes integrability of the actual derivative
norm. A finite third moment and the original integrable-power
condition suffice; standardization and fourth moments will enter
when combining this exterior estimate with the local bound.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- A polynomial in one plus the frequency norm remains integrable against Gaussian decay.
Source: arXiv:2412.09080v3, §5.4 `eq:small-z`, the
Gaussian polynomial envelope of the corrected derivatives. -/
theorem integrable_one_add_norm_pow_gaussian_frequency (p : ℕ) {b : ℝ} (hb : 0 < b) :
    Integrable (fun ξ : ℝ × ℝ => (1 + ‖ξ‖) ^ p * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
  have hI0 : Integrable (fun ξ : ℝ × ℝ => Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
    simpa only [pow_zero, one_mul] using integrable_norm_pow_gaussian_frequency 0 hb
  have hIp := integrable_norm_pow_gaussian_frequency p hb
  have hpoly (ξ : ℝ × ℝ) : (1 + ‖ξ‖) ^ p ≤ (2 : ℝ) ^ p * (1 + ‖ξ‖ ^ p) := by
    rcases le_total ‖ξ‖ 1 with h | h
    · calc
        _ ≤ (2 : ℝ) ^ p := pow_le_pow_left₀ (by positivity) (by linarith) p
        _ ≤ _ := by nlinarith [pow_nonneg (norm_nonneg ξ) p,
          mul_nonneg (show 0 ≤ (2 : ℝ) ^ p by positivity) (pow_nonneg (norm_nonneg ξ) p)]
    · calc
        _ ≤ (2 * ‖ξ‖) ^ p := pow_le_pow_left₀ (by positivity) (by linarith) p
        _ = (2 : ℝ) ^ p * ‖ξ‖ ^ p := mul_pow _ _ _
        _ ≤ _ := by nlinarith [show 0 ≤ (2 : ℝ) ^ p by positivity]
  apply ((hI0.add hIp).const_mul ((2 : ℝ) ^ p)).mono' (by fun_prop)
  refine ae_of_all _ fun ξ => ?_
  rw [Real.norm_of_nonneg (by positivity)]
  calc
    _ ≤ ((2 : ℝ) ^ p * (1 + ‖ξ‖ ^ p)) * Real.exp (-b * (ξ.1 ^ 2 + ξ.2 ^ 2)) :=
      mul_le_mul_of_nonneg_right (hpoly ξ) (Real.exp_pos _).le
    _ = _ := by dsimp only [Pi.add_apply]; ring

example : (0 : ℝ) < 1 / 8 := by norm_num

/-- The actual cubic-corrected third derivative has an inverse-n exterior norm integral.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`,
`eq:big-phi-poly`, and `eq:big-z-poly`, at derivative order three. -/
theorem eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_tail
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hcf : HasIntegrableCharFun μ) {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      Integrable (fun ξ => ‖iteratedFDeriv ℝ 3 (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) ∧
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ 3 (fun η =>
          characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
            characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
              characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨Ct, hCt, htail⟩ :=
    eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_tail_rate_inv_nat μ hmom hcf ha
  obtain ⟨Cg, hCg, hGtail⟩ := exists_integral_characteristic2_stdGauss2_derivative_tail_rate_inv_nat ha
  obtain ⟨Cb, hCb, hBtail⟩ := exists_integrable_third_gaussian_cubicCorrection_tail_rate_inv_nat μ hmom ha
  have hIG : Integrable (fun ξ => ‖iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ‖) := by
    simpa only [pow_zero, one_mul] using integrable_norm_pow_iteratedFDeriv_characteristic2_stdGauss2 0 3 le_rfl
  refine ⟨Ct + Cg + Cb, by positivity, ?_⟩
  filter_upwards [htail, eventually_integrable_norm_iteratedFDeriv_characteristic_scaledSum_cubic_error μ hmom hcf,
    eventually_ge_atTop 1] with n hT hIerror hn
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  let c : ℝ := (Real.sqrt n)⁻¹
  have hc : 0 ≤ c := by dsimp [c]; positivity
  have hc1 : c ≤ 1 := (inv_le_one₀ hs).mpr (Real.one_le_sqrt.mpr (by exact_mod_cast hn))
  let f := fun ξ => ‖iteratedFDeriv ℝ 3 (fun η =>
    characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
      characteristic2 stdGauss2 η - (c : ℂ) * characteristic2 stdGauss2 η *
        cubicCharacteristicCorrection μ η) ξ‖
  let u := fun ξ => ‖iteratedFDeriv ℝ 3
    (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖
  let g := fun ξ => ‖iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ‖
  let b := fun ξ => ‖iteratedFDeriv ℝ 3
    (fun η => characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖
  let S : Set (ℝ × ℝ) := {ξ | a * Real.sqrt n ≤ ‖ξ‖}
  have hS : MeasurableSet S := (isClosed_le continuous_const continuous_norm).measurableSet
  have hIB : Integrable b := (hBtail n hn).1.norm
  have hsum : Integrable (fun ξ => u ξ + g ξ + c * b ξ) := (hT.1.norm.add hIG).add (hIB.const_mul c)
  have hpoint (ξ : ℝ × ℝ) : f ξ ≤ u ξ + g ξ + c * b ξ := by
    dsimp [f, u, g, b, c]
    rw [iteratedFDeriv_characteristic_scaledSum_cubic_error μ hmom n 3 le_rfl]
    have he : ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) • iteratedFDeriv ℝ 3
        (fun η => characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖ =
        (Real.sqrt n)⁻¹ * ‖iteratedFDeriv ℝ 3
          (fun η => characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖ := by
      rw [norm_smul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    calc
      _ ≤ ‖iteratedFDeriv ℝ 3 (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ -
          iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ‖ +
            ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) • iteratedFDeriv ℝ 3
              (fun η => characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖ := norm_sub_le _ _
      _ ≤ _ := by rw [he]; exact add_le_add (norm_sub_le _ _) le_rfl
  refine ⟨hIerror 3 le_rfl, ?_⟩
  change (∫ ξ in S, f ξ) ≤ _
  calc
    _ ≤ ∫ ξ in S, u ξ + g ξ + c * b ξ :=
      setIntegral_mono_on (hIerror 3 le_rfl).integrableOn hsum.integrableOn hS (fun ξ _ => hpoint ξ)
    _ = (∫ ξ in S, u ξ) + (∫ ξ in S, g ξ) + c * ∫ ξ in S, b ξ := by
      rw [integral_add (f := fun ξ => u ξ + g ξ) (g := fun ξ => c * b ξ)
        (hT.1.norm.add hIG).integrableOn (hIB.const_mul c).integrableOn,
        integral_add (f := u) (g := g) hT.1.norm.integrableOn hIG.integrableOn, integral_const_mul]
    _ ≤ Ct * (n : ℝ)⁻¹ + Cg * (n : ℝ)⁻¹ + c * (Cb * (n : ℝ)⁻¹) :=
      add_le_add (add_le_add hT.2 (hGtail n hn 3 le_rfl))
        (mul_le_mul_of_nonneg_left (hBtail n hn).2 hc)
    _ ≤ Ct * (n : ℝ)⁻¹ + Cg * (n : ℝ)⁻¹ + Cb * (n : ℝ)⁻¹ := by
      apply add_le_add le_rfl
      simpa only [one_mul] using mul_le_mul_of_nonneg_right hc1
        (mul_nonneg hCb (inv_nonneg.mpr hn0.le))
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2, one_pos⟩

/-- The actual corrected third-derivative exterior integral tends to zero.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp` and
`eq:big-z-poly`, the convergence consequence of the inverse-n bound. -/
theorem tendsto_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_tail_zero
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hcf : HasIntegrableCharFun μ) {a : ℝ} (ha : 0 < a) :
    Tendsto (fun n : ℕ => ∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
      ‖iteratedFDeriv ℝ 3 (fun η =>
        characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) η -
          characteristic2 stdGauss2 η - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 η * cubicCharacteristicCorrection μ η) ξ‖) atTop (nhds 0) := by
  obtain ⟨C, _, htail⟩ :=
    eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_cubic_error_tail μ hmom hcf ha
  have hlim : Tendsto (fun n : ℕ => C * (n : ℝ)⁻¹) atTop (nhds 0) := by
    simpa only [mul_zero, Function.comp_def] using
      (tendsto_inv_atTop_zero.comp
        (tendsto_natCast_atTop_atTop : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop)).const_mul C
  apply squeeze_zero' (Eventually.of_forall fun n => integral_nonneg fun ξ => norm_nonneg _) _ hlim
  filter_upwards [htail] with n hn
  exact hn.2

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasIntegrableCharFun_stdGauss2, one_pos⟩

end Transformer.Modes
