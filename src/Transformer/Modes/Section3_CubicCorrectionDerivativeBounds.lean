module

public import Transformer.Modes.Section3_CubicHigherDerivatives
public import Transformer.Modes.Section3_GaussianThirdDerivativeBounds
public import Mathlib.Analysis.Calculus.ContDiff.Bounds

/-!
# Bounds and integrability for derivatives of the Gaussian-damped cubic correction

The additional Fourier term in arXiv:2412.09080v3, §5.4 `eq:big-z-poly`,
is the actual Gaussian characteristic function times the literal cubic
coefficient of §3.1 `eq:psi`. All derivatives of the cubic coefficient
through order three have a common polynomial bound controlled by the
actual third norm moment. The Leibniz bound combines this with the
proved Gaussian decay of every Gaussian derivative through order three.

The resulting envelope is `C (1 + ‖ξ‖³) exp(-|ξ|²/8)`. Every term
in the product rule is present, including differentiation of the
Gaussian damping. Its finite Gaussian moment integrals give whole-space
integrability of the actual correction derivatives. These conclusions
need only a finite third moment; exponential moments are used separately
to identify the source's logarithmic cumulants with the cubic coefficient.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Every actual cubic coefficient derivative through order three has one common polynomial envelope.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`, the
polynomial factor before multiplying by the Gaussian characteristic function. -/
theorem norm_iteratedFDeriv_cubicCharacteristicCorrection_polynomial_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (k : ℕ) (hk : k ≤ 3) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ k (cubicCharacteristicCorrection μ) ξ‖ ≤
      8 * (∫ z, ‖z‖ ^ 3 ∂μ) * (1 + ‖ξ‖ ^ 3) := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hp (j : ℕ) (hj : j ≤ 3) : ‖ξ‖ ^ j ≤ 1 + ‖ξ‖ ^ 3 := by
    rcases le_or_gt ‖ξ‖ 1 with hξ | hξ
    · have h := pow_le_one₀ (norm_nonneg ξ) hξ (n := j)
      have h3 : 0 ≤ ‖ξ‖ ^ 3 := by positivity
      linarith
    · calc
        _ ≤ ‖ξ‖ ^ 3 := pow_le_pow_right₀ hξ.le hj
        _ ≤ _ := by linarith
  interval_cases k
  · rw [norm_iteratedFDeriv_zero]
    calc
      _ ≤ (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 3 :=
        norm_cubicCharacteristicCorrection_le μ hmom ξ
      _ ≤ _ := by gcongr <;> first | exact hp 3 le_rfl | norm_num
  · rw [norm_iteratedFDeriv_one]
    calc
      _ ≤ 4 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 2 :=
        norm_fderiv_cubicCharacteristicCorrection_le μ hmom ξ
      _ ≤ _ := by gcongr <;> first | exact hp 2 (by omega) | norm_num
  · calc
      _ ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ :=
        norm_iteratedFDeriv_two_cubicCharacteristicCorrection_le μ hmom ξ
      _ ≤ _ := mul_le_mul_of_nonneg_left (by simpa using hp 1 (by omega)) (by positivity)
  · calc
      _ ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ) :=
        norm_iteratedFDeriv_three_cubicCharacteristicCorrection_le μ hmom ξ
      _ ≤ _ := by nlinarith [mul_nonneg hM (pow_nonneg (norm_nonneg ξ) 3)]

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (2 : ℕ) ≤ 3 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), by omega⟩

/-- All derivatives through order three of the literal Gaussian-damped correction have polynomial Gaussian decay.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`, including
every Leibniz term and the actual Gaussian damping derivative. -/
theorem exists_norm_iteratedFDeriv_gaussian_cubicCorrection_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ k : ℕ, k ≤ 3 → ∀ ξ : ℝ × ℝ,
      ‖iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 η *
        cubicCharacteristicCorrection μ η) ξ‖ ≤
          C * (1 + ‖ξ‖ ^ 3) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  obtain ⟨G, hG0, hG⟩ := exists_norm_iteratedFDeriv_characteristic2_stdGauss2_gaussian_bound
  let B : ℝ := 8 * ∫ z, ‖z‖ ^ 3 ∂μ
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hB0 : 0 ≤ B := by dsimp [B]; positivity
  have hGC : ContDiff ℝ 3 (characteristic2 stdGauss2) :=
    contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))
  have hBC : ContDiff ℝ 3 (cubicCharacteristicCorrection μ) :=
    (contDiff_cubicCharacteristicCorrection_of_memLp μ hmom).of_le (by simp)
  refine ⟨8 * G * B, by positivity, ?_⟩
  intro k hk ξ
  have h := norm_iteratedFDeriv_mul_le hGC hBC ξ (n := k) (by exact_mod_cast hk)
  have hs : (∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ)) ≤ 8 := by
    interval_cases k <;> norm_num [Finset.sum_range_succ, Nat.choose]
  calc
    _ ≤ ∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ) *
        ‖iteratedFDeriv ℝ i (characteristic2 stdGauss2) ξ‖ *
        ‖iteratedFDeriv ℝ (k - i) (cubicCharacteristicCorrection μ) ξ‖ := h
    _ ≤ ∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ) *
        (G * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)) * (B * (1 + ‖ξ‖ ^ 3)) := by
      apply Finset.sum_le_sum
      intro i hi
      have hi3 : i ≤ 3 := by have := Finset.mem_range.mp hi; omega
      gcongr
      · exact hG i hi3 ξ
      · exact norm_iteratedFDeriv_cubicCharacteristicCorrection_polynomial_le μ hmom
          (k - i) (by omega) ξ
    _ = (∑ i ∈ Finset.range (k + 1), (k.choose i : ℝ)) *
        (G * B * (1 + ‖ξ‖ ^ 3) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)) := by
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i hi
      ring
    _ ≤ 8 * (G * B * (1 + ‖ξ‖ ^ 3) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)) :=
      mul_le_mul_of_nonneg_right hs (by positivity)
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Every polynomial norm weight of the actual correction derivatives through order three is integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`, the finite
Gaussian moments used to estimate the differentiated correction's exterior integral. -/
theorem integrable_norm_pow_iteratedFDeriv_gaussian_cubicCorrection
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (l k : ℕ) (hk : k ≤ 3) :
    Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ ^ l *
      ‖iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 η *
        cubicCharacteristicCorrection μ η) ξ‖) := by
  obtain ⟨C, _, hbound⟩ := exists_norm_iteratedFDeriv_gaussian_cubicCorrection_bound μ hmom
  have he (ξ : ℝ × ℝ) : Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) =
      Real.exp (-(1 / 8 : ℝ) * (ξ.1 ^ 2 + ξ.2 ^ 2)) := by congr 1; ring
  have hI : Integrable (fun ξ : ℝ × ℝ =>
      C * ‖ξ‖ ^ l * (1 + ‖ξ‖ ^ 3) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)) := by
    convert ((integrable_norm_pow_gaussian_frequency l (by norm_num : (0 : ℝ) < 1 / 8)).add
      (integrable_norm_pow_gaussian_frequency (l + 3) (by norm_num : (0 : ℝ) < 1 / 8))).const_mul C using 1
    funext ξ
    simp only [Pi.add_apply]
    rw [← he ξ, pow_add]
    ring
  have hC : ContDiff ℝ 3 (fun η => characteristic2 stdGauss2 η *
      cubicCharacteristicCorrection μ η) :=
    (contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))).mul
      ((contDiff_cubicCharacteristicCorrection_of_memLp μ hmom).of_le (by simp))
  have hcont := ContDiff.continuous_iteratedFDeriv (by exact_mod_cast hk) hC
  apply hI.mono' ((continuous_norm.pow l).mul hcont.norm).aestronglyMeasurable
  exact ae_of_all _ fun ξ => by
    change ‖‖ξ‖ ^ l * ‖iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 η *
      cubicCharacteristicCorrection μ η) ξ‖‖ ≤ _
    rw [Real.norm_of_nonneg (by positivity : 0 ≤ ‖ξ‖ ^ l *
      ‖iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 η *
        cubicCharacteristicCorrection μ η) ξ‖)]
    calc
      _ ≤ ‖ξ‖ ^ l * (C * (1 + ‖ξ‖ ^ 3) * Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8)) :=
        mul_le_mul_of_nonneg_left (hbound k hk ξ) (by positivity)
      _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (2 : ℕ) ≤ 3 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), by omega⟩

/-- The actual Gaussian-damped correction has integrable multilinear derivatives through order three.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal` and
`eq:big-z-poly`, the actual derivatives required for Fourier inversion. -/
theorem integrable_iteratedFDeriv_gaussian_cubicCorrection
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (k : ℕ) (hk : k ≤ 3) :
    Integrable (iteratedFDeriv ℝ k (fun η => characteristic2 stdGauss2 η *
      cubicCharacteristicCorrection μ η)) := by
  have hC : ContDiff ℝ 3 (fun η => characteristic2 stdGauss2 η *
      cubicCharacteristicCorrection μ η) :=
    (contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))).mul
      ((contDiff_cubicCharacteristicCorrection_of_memLp μ hmom).of_le (by simp))
  apply (integrable_norm_iff (ContDiff.continuous_iteratedFDeriv
    (by exact_mod_cast hk) hC).aestronglyMeasurable).mp
  simpa only [pow_zero, one_mul] using
    integrable_norm_pow_iteratedFDeriv_gaussian_cubicCorrection μ hmom 0 k hk

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (2 : ℕ) ≤ 3 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), by omega⟩

end Transformer.Modes
