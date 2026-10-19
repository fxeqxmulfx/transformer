module

public import Transformer.Modes.Section3_ThirdDerivativeTail
public import Transformer.Modes.Section3_FirstDerivativeFrequency

/-!
# Gaussian decay through the actual third characteristic derivative

The Gaussian exterior estimate in arXiv:2412.09080v3, §5.4
`eq:big-phi-poly`, is needed through order three. The literal Gaussian
normalized-sum identity with four summands preserves the Gaussian
characteristic function. The proved sum derivative bounds therefore
retain at least one Gaussian factor at half frequency, giving the
common envelope `C exp(-|ξ|²/8)` at every order from zero to three.

This argument uses the actual derivative rather than imposing a
Gaussian derivative formula. It also derives third-derivative Fourier
integrability and its inverse-n exterior rate from the actual sum's
integrable-power condition. The envelope will permit polynomial
weights when differentiating the Gaussian-damped cubic correction.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- All actual Gaussian characteristic derivatives through order three share a Gaussian envelope.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`, with a
nonsharp damping exponent `1/8` from the literal four-summand identity. -/
theorem exists_norm_iteratedFDeriv_characteristic2_stdGauss2_gaussian_bound :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ k : ℕ, k ≤ 3 → ∀ ξ : ℝ × ℝ,
      ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖ ≤
        C * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  let A : ℝ := 2 * ∫ z, ‖z‖ ∂stdGauss2
  let B : ℝ := 4 * ∫ z, ‖z‖ ^ 2 ∂stdGauss2
  let D : ℝ := 8 * ∫ z, ‖z‖ ^ 3 ∂stdGauss2
  let C1 : ℝ := 2 * A
  let C2 : ℝ := 4 * (A ^ 2 + B)
  let C3 : ℝ := 64 * (A ^ 3 + 3 * A * B + D)
  let C : ℝ := 1 + C1 + C2 + C3
  have hM1 : 0 ≤ ∫ z, ‖z‖ ∂stdGauss2 := integral_nonneg fun z => norm_nonneg z
  have hM2 : 0 ≤ ∫ z, ‖z‖ ^ 2 ∂stdGauss2 := integral_nonneg fun z => sq_nonneg _
  have hM3 : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hC1 : 0 ≤ C1 := by dsimp [C1, A]; positivity
  have hC2 : 0 ≤ C2 := by dsimp [C2, B]; positivity
  have hC3 : 0 ≤ C3 := by dsimp [C3, A, B, D]; positivity
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have he : characteristic2 ((Measure.pi fun _ : Fin 4 => stdGauss2).map (scaledSum 4)) =
      characteristic2 stdGauss2 := by
    funext ξ
    rw [characteristic_scaledSum, characteristic2_stdGauss2_scaled_power 4 (by omega)]
  have hscale : (Real.sqrt (4 : ℝ))⁻¹ = 1 / 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num : (0 : ℝ) ≤ 2)]
    norm_num
  refine ⟨C, hC, ?_⟩
  intro k hk ξ
  let r : ℝ := ‖characteristic2 stdGauss2 ((1 / 2 : ℝ) • ξ)‖
  have hr : 0 ≤ r := norm_nonneg _
  have hr1 : r ≤ 1 := norm_characteristic2_le_one stdGauss2 _
  have hrG : r = Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
    dsimp [r]
    rw [characteristic2_stdGauss2, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _)]
    congr 1
    simp only [Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
    ring
  have h1 := norm_fderiv_characteristic_scaledSum_le stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) 4 ξ
  have h2 := norm_iteratedFDeriv_two_characteristic_scaledSum_le stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) (n := 4) (by omega) ξ
  have h3 := norm_iteratedFDeriv_three_characteristic_scaledSum_le stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) (n := 4) (by omega) ξ
  rw [he] at h1 h2 h3
  norm_num at h1 h2 h3
  rw [hscale] at h1 h2 h3
  have h1' : ‖fderiv ℝ (characteristic2 stdGauss2) ξ‖ ≤ C1 * r := by
    calc
      _ ≤ C1 * r ^ 3 := by convert h1 using 1; dsimp [C1, A, r]; ring
      _ ≤ C1 * r := mul_le_mul_of_nonneg_left
        (by simpa using pow_le_pow_of_le_one hr hr1 (by omega : 1 ≤ 3)) hC1
  have h2' : ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖ ≤ C2 * r := by
    calc
      _ ≤ C2 * r ^ 2 := h2
      _ ≤ C2 * r := mul_le_mul_of_nonneg_left
        (by simpa using pow_le_pow_of_le_one hr hr1 (by omega : 1 ≤ 2)) hC2
  have h3' : ‖iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ‖ ≤ C3 * r := h3
  rw [← hrG]
  interval_cases k
  · rw [norm_iteratedFDeriv_zero]
    have hval : ‖characteristic2 stdGauss2 ξ‖ = r ^ 4 := by
      rw [← characteristic2_stdGauss2_scaled_power 4 (by omega) ξ, norm_pow]
      norm_num only [Nat.cast_ofNat]
      rw [hscale]
    calc
      _ = r ^ 4 := hval
      _ ≤ r := by simpa using pow_le_pow_of_le_one hr hr1 (by omega : 1 ≤ 4)
      _ = 1 * r := (one_mul r).symm
      _ ≤ C * r := by dsimp [C]; gcongr; linarith
  · rw [norm_iteratedFDeriv_one]
    exact h1'.trans (mul_le_mul_of_nonneg_right (by dsimp [C]; linarith) hr)
  · exact h2'.trans (mul_le_mul_of_nonneg_right (by dsimp [C]; linarith) hr)
  · exact h3'.trans (mul_le_mul_of_nonneg_right (by dsimp [C]; linarith) hr)

/-- Every polynomial norm weight of each actual Gaussian characteristic derivative through order three is integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly` and
`eq:big-z-poly`, the finite Gaussian moment integrals used for the exterior. -/
theorem integrable_norm_pow_iteratedFDeriv_characteristic2_stdGauss2
    (l k : ℕ) (hk : k ≤ 3) :
    Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ ^ l *
      ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖) := by
  obtain ⟨C, _, hbound⟩ := exists_norm_iteratedFDeriv_characteristic2_stdGauss2_gaussian_bound
  have he (ξ : ℝ × ℝ) : Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) =
      Real.exp (-(1 / 8 : ℝ) * (ξ.1 ^ 2 + ξ.2 ^ 2)) := by congr 1; ring
  have hI : Integrable (fun ξ : ℝ × ℝ =>
      C * (‖ξ‖ ^ l * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8))) := by
    simp_rw [he]
    exact (integrable_norm_pow_gaussian_frequency l (by norm_num : (0 : ℝ) < 1 / 8)).const_mul C
  have hcont : Continuous (iteratedFDeriv ℝ k (characteristic2 stdGauss2)) :=
    ContDiff.continuous_iteratedFDeriv (by exact_mod_cast hk)
      (contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp)))
  apply hI.mono' ((continuous_norm.pow l).mul hcont.norm).aestronglyMeasurable
  exact ae_of_all _ fun ξ => by
    change ‖‖ξ‖ ^ l * ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖‖ ≤
      C * (‖ξ‖ ^ l * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8))
    rw [Real.norm_of_nonneg (by positivity :
      0 ≤ ‖ξ‖ ^ l * ‖iteratedFDeriv ℝ k (characteristic2 stdGauss2) ξ‖)]
    calc
      _ ≤ ‖ξ‖ ^ l * (C * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)) :=
        mul_le_mul_of_nonneg_left (hbound k hk ξ) (by positivity)
      _ = _ := by ring

example : (0 : ℕ) ≤ 3 := by omega

/-- The actual third Gaussian characteristic derivative is integrable over the whole frequency space.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal` and
`eq:big-phi-poly`, the third Gaussian derivative in Fourier inversion. -/
theorem integrable_iteratedFDeriv_three_characteristic2_stdGauss2 :
    Integrable (iteratedFDeriv ℝ 3 (characteristic2 stdGauss2)) := by
  obtain ⟨n, hI, hn⟩ :=
    ((eventually_integrable_iteratedFDeriv_three_characteristic_scaledSum stdGauss2
      (IsGaussian.memLp_id _ _ (by simp)) hasIntegrableCharFun_stdGauss2).and
      (eventually_ge_atTop 1)).exists
  have he : characteristic2 ((Measure.pi fun _ : Fin n => stdGauss2).map (scaledSum n)) =
      characteristic2 stdGauss2 := by
    funext ξ
    rw [characteristic_scaledSum, characteristic2_stdGauss2_scaled_power n hn]
  rw [he] at hI
  exact hI

/-- The actual third Gaussian characteristic derivative has inverse-n norm integral on the growing exterior.
Source: arXiv:2412.09080v3, §5.4 `eq:big-phi-poly`, derivative order
three at radius `a sqrt n`. The Gaussian normalized-sum identity
transfers the proved genuine third-derivative tail rate. -/
theorem eventually_integral_iteratedFDeriv_three_characteristic2_stdGauss2_tail_rate_inv_nat
    {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop,
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ‖) ≤ C * (n : ℝ)⁻¹ := by
  obtain ⟨C, hC, hT⟩ :=
    eventually_integral_iteratedFDeriv_three_characteristic_scaledSum_tail_rate_inv_nat stdGauss2
      (IsGaussian.memLp_id _ _ (by simp)) hasIntegrableCharFun_stdGauss2 ha
  refine ⟨C, hC, ?_⟩
  filter_upwards [hT, eventually_ge_atTop 1] with n hnT hn
  have he : characteristic2 ((Measure.pi fun _ : Fin n => stdGauss2).map (scaledSum n)) =
      characteristic2 stdGauss2 := by
    funext ξ
    rw [characteristic_scaledSum, characteristic2_stdGauss2_scaled_power n hn]
  rw [he] at hnT
  exact hnT.2

example : (0 : ℝ) < 1 := one_pos

end Transformer.Modes
