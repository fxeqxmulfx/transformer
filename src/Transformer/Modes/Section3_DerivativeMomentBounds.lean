import Transformer.Modes.Section3_PowerDerivatives
import Transformer.Modes.Section3_DerivativeMoments
import Transformer.Modes.Section3_FourierTail
/-!
# Moment bounds for normalized-sum characteristic derivatives

The large-frequency product rule in arXiv:2412.09080v3, §5.4
`eq:big-z-exp`, leaves the undifferentiated characteristic factors.
These global operator bounds make those factors explicit for the first
two derivatives of the actual normalized-sum characteristic function.

The first derivative retains power `n - 1`. For the second derivative,
the two product-rule terms can both be bounded using power `n - 2`,
since every probability characteristic function has modulus at most one.
The displayed coefficients depend on the actual first and second norm
moments; finite second moment supplies them and continuous derivatives.
These bounds are the inputs to integrability and exterior damping,
without inferring Fourier integrability from a bounded density.

The first coefficient is `n / sqrt n`. The second-order coefficient
with two first derivatives simplifies to `n - 1`, and the coefficient
with one second derivative simplifies to one. The displayed upper
bound replaces these by `n`, retaining one common characteristic
power for subsequent integration over the whole frequency space.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Finite second moment gives two continuous derivatives of the actual sum.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
the differentiability used by the normalized-sum product rule. -/
theorem contDiff_characteristic_scaledSum (μ : Measure (ℝ × ℝ))
    [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ) (n : ℕ) :
    ContDiff ℝ 2 (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) := by
  let g : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ) := (Real.sqrt n)⁻¹ • ContinuousLinearMap.id ℝ (ℝ × ℝ)
  have he : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun ξ => characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ n := by
    funext ξ
    exact characteristic_scaledSum μ n ξ
  rw [he]
  exact ((contDiff_characteristic2 μ (k := 2) hmom).comp g.contDiff).pow n

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The first sum derivative retains `n - 1` characteristic factors.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, derivative order one.
The normalization is the literal `1 / sqrt n` frequency scale. -/
theorem norm_fderiv_characteristic_scaledSum_le (μ : Measure (ℝ × ℝ))
    [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ) (n : ℕ) (ξ : ℝ × ℝ) :
    ‖fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖ ≤
      (n : ℝ) * (Real.sqrt n)⁻¹ * (2 * ∫ z, ‖z‖ ∂μ) *
        ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ (n - 1) := by
  let η : ℝ × ℝ := (Real.sqrt n)⁻¹ • ξ
  have hA : 0 ≤ 2 * ∫ z, ‖z‖ ∂μ := by
    have hM : 0 ≤ ∫ z, ‖z‖ ∂μ := integral_nonneg fun z => norm_nonneg z
    positivity
  have hc : 0 ≤ (Real.sqrt (n : ℝ))⁻¹ := inv_nonneg.mpr (Real.sqrt_nonneg _)
  have hD := norm_fderiv_characteristic2_le μ (hmom.mono_exponent (by norm_num)) η
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  intro v
  rw [fderiv_characteristic_scaledSum_apply μ hmom n ξ v]
  simp only [norm_mul, Complex.norm_natCast, Complex.norm_real, Real.norm_eq_abs,
    abs_of_nonneg hc, norm_pow]
  have hDv : ‖fderiv ℝ (characteristic2 μ) η v‖ ≤ (2 * ∫ z, ‖z‖ ∂μ) * ‖v‖ :=
    ((fderiv ℝ (characteristic2 μ) η).le_opNorm v).trans
      (mul_le_mul_of_nonneg_right hD (norm_nonneg v))
  calc
    _ ≤ (n : ℝ) * (Real.sqrt n)⁻¹ * ‖characteristic2 μ η‖ ^ (n - 1) *
        ((2 * ∫ z, ‖z‖ ∂μ) * ‖v‖) := by gcongr
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The second sum derivative retains `n - 2` characteristic factors.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, derivative order two.
Both product-rule terms are bounded by explicit norm-moment coefficients. -/
theorem norm_iteratedFDeriv_two_characteristic_scaledSum_le (μ : Measure (ℝ × ℝ))
    [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ) {n : ℕ} (hn : 2 ≤ n) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2
        (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ‖ ≤
      (n : ℝ) * ((2 * ∫ z, ‖z‖ ∂μ) ^ 2 + 4 * ∫ z, ‖z‖ ^ 2 ∂μ) *
        ‖characteristic2 μ ((Real.sqrt n)⁻¹ • ξ)‖ ^ (n - 2) := by
  let η : ℝ × ℝ := (Real.sqrt n)⁻¹ • ξ
  let A : ℝ := 2 * ∫ z, ‖z‖ ∂μ
  let B : ℝ := 4 * ∫ z, ‖z‖ ^ 2 ∂μ
  let t : ℝ := ‖characteristic2 μ η‖
  have hM : 0 ≤ ∫ z, ‖z‖ ∂μ := integral_nonneg fun z => norm_nonneg z
  have hM2 : 0 ≤ ∫ z, ‖z‖ ^ 2 ∂μ := integral_nonneg fun z => sq_nonneg _
  have hA : 0 ≤ A := by dsimp [A]; positivity
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have ht : 0 ≤ t := norm_nonneg _
  have ht1 : t ≤ 1 := norm_characteristic2_le_one μ η
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt hn0.le
  have hcoef : (n : ℝ) * (Real.sqrt n)⁻¹ ^ 2 = 1 := by
    calc
      _ = (Real.sqrt n) ^ 2 * (Real.sqrt n)⁻¹ ^ 2 := by rw [hsq]
      _ = _ := by field_simp
  have hD := norm_fderiv_characteristic2_le μ (hmom.mono_exponent (by norm_num)) η
  have hD2 := norm_iteratedFDeriv_characteristic2_le μ (k := 2) hmom η
  have hD2B : ‖iteratedFDeriv ℝ 2 (characteristic2 μ) η‖ ≤ B := by
    norm_num only at hD2
    exact hD2
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  rw [Fin.prod_univ_two]
  have hDv (w : ℝ × ℝ) : ‖fderiv ℝ (characteristic2 μ) η w‖ ≤ A * ‖w‖ :=
    ((fderiv ℝ (characteristic2 μ) η).le_opNorm w).trans
      (mul_le_mul_of_nonneg_right hD (norm_nonneg w))
  have hD2v : ‖iteratedFDeriv ℝ 2 (characteristic2 μ) η v‖ ≤ B * (‖v 0‖ * ‖v 1‖) := by
    have h := (iteratedFDeriv ℝ 2 (characteristic2 μ) η).le_opNorm v
    rw [Fin.prod_univ_two] at h
    exact h.trans (mul_le_mul_of_nonneg_right hD2B (by positivity))
  rw [iteratedFDeriv_two_characteristic_scaledSum_apply μ hmom n ξ v]
  calc
    _ ≤ ‖(n : ℂ) * (n - 1 : ℕ) * (((Real.sqrt n)⁻¹ : ℝ) : ℂ) ^ 2 *
        characteristic2 μ η ^ (n - 2) * fderiv ℝ (characteristic2 μ) η (v 0) *
        fderiv ℝ (characteristic2 μ) η (v 1)‖ +
      ‖(n : ℂ) * (((Real.sqrt n)⁻¹ : ℝ) : ℂ) ^ 2 * characteristic2 μ η ^ (n - 1) *
        iteratedFDeriv ℝ 2 (characteristic2 μ) η v‖ := norm_add_le _ _
    _ ≤ (n : ℝ) * (n - 1 : ℕ) * (Real.sqrt n)⁻¹ ^ 2 * t ^ (n - 2) * (A * ‖v 0‖) * (A * ‖v 1‖) +
      (n : ℝ) * (Real.sqrt n)⁻¹ ^ 2 * t ^ (n - 1) * (B * (‖v 0‖ * ‖v 1‖)) := by
      simp only [norm_mul, norm_pow, Complex.norm_natCast, Complex.norm_real, Real.norm_eq_abs,
        abs_of_pos (inv_pos.mpr hs)]
      gcongr
      all_goals exact hDv _
    _ = (((n - 1 : ℕ) : ℝ) * t ^ (n - 2) * A ^ 2 + t ^ (n - 1) * B) * (‖v 0‖ * ‖v 1‖) := by
      calc
        _ = (((n : ℝ) * (Real.sqrt n)⁻¹ ^ 2) *
          (((n - 1 : ℕ) : ℝ) * t ^ (n - 2) * A ^ 2 + t ^ (n - 1) * B)) * (‖v 0‖ * ‖v 1‖) := by ring
        _ = _ := by rw [hcoef, one_mul]
    _ ≤ ((n : ℝ) * t ^ (n - 2) * A ^ 2 + (n : ℝ) * t ^ (n - 2) * B) * (‖v 0‖ * ‖v 1‖) := by
      gcongr
      · exact_mod_cast (show n - 1 ≤ n by omega)
      · calc
          _ ≤ t ^ (n - 2) := pow_le_pow_of_le_one ht ht1 (by omega)
          _ ≤ (n : ℝ) * t ^ (n - 2) := by
            have hnR : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
            nlinarith [mul_nonneg (sub_nonneg.mpr hnR) (pow_nonneg ht (n - 2))]
    _ = _ := by dsimp [A, B, t, η]; ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 ∧ 2 ≤ (3 : ℕ) :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), by omega⟩

end Transformer.Modes
