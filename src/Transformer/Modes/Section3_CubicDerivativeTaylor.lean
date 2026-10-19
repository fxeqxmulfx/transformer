module

public import Transformer.Modes.Section3_CubicDerivatives
public import Transformer.Modes.Section3_WeightedDotMoments
public import Transformer.Modes.Section3_GaussianDerivativeComparison
public import Transformer.Modes.Section3_CubicGaussianComparison

/-!
# Fourth-moment Taylor error for the first characteristic derivative

The `s = 3` comparison in arXiv:2412.09080v3, §5.4
`eq:br-9.10`, subtracts the derivative of the literal cubic correction,
not just the Gaussian derivative. Expanding the weighted unit phase
through degree two gives that exact correction and leaves a cubic
frequency remainder controlled by the fourth norm moment.

Standardization computes the constant and linear weighted moments.
The quadratic weighted moment is the already proved actual derivative
of `B(ξ) = -i E⟨ξ,X⟩³ / 6`, with coefficient `-i/2`.
The resulting Gaussian comparison keeps this derivative separate;
differentiating the Gaussian-damped correction adds further terms.
Every estimate below is for one summand, before frequency scaling
and the normalized-sum product rule.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The first characteristic derivative has cubic error after retaining the actual correction.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:br-9.10`, the unscaled first-derivative expansion. The nonsharp
factor eight is `2^4 / 2!` in the product sup norm. -/
theorem norm_fderiv_characteristic2_add_dot_sub_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ v : ℝ × ℝ) :
    ‖fderiv ℝ (characteristic2 μ) ξ v + ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ) -
      fderiv ℝ (cubicCharacteristicCorrection μ) ξ v‖ ≤
      8 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 3 * ‖v‖ := by
  have h := norm_iteratedFDeriv_characteristic2_sub_phase_taylor μ 1 2
    (by simpa using hmom) ξ (fun _ => v)
  simp only [iteratedFDeriv_one_apply, Fin.prod_univ_one, pow_one] at h
  have hmean : ∫ z, z.1 * v.1 + z.2 * v.2 ∂μ = 0 := by
    convert integral_dot_of_standardized μ hμ v using 1
    congr 1
    funext z
    ring
  have hcov : ∫ z, (z.1 * v.1 + z.2 * v.2) * (z.1 * ξ.1 + z.2 * ξ.2) ∂μ =
      ξ.1 * v.1 + ξ.2 * v.2 := by
    convert integral_dot_mul_of_standardized μ hμ ξ v using 1
    congr 1
    funext z
    ring
  have hB : fderiv ℝ (cubicCharacteristicCorrection μ) ξ v =
      -Complex.I * ((∫ z, (z.1 * v.1 + z.2 * v.2) *
        (z.1 * ξ.1 + z.2 * ξ.2) ^ 2 ∂μ : ℝ) : ℂ) / 2 := by
    rw [fderiv_cubicCharacteristicCorrection_apply μ
      (hmom.mono_exponent (by norm_num))]
    congr 2
    congr 1
    apply integral_congr_ae
    exact ae_of_all _ fun z => by ring
  norm_num [Finset.sum_range_succ, Complex.I_sq, hmean, hcov] at h
  rw [hB]
  have halg (a b c : ℂ) : a + b - (-Complex.I * c / 2) =
      a - Complex.I * (b * Complex.I + -c / 2) := by
    ring_nf
    rw [Complex.I_sq]
    ring
  convert h using 1
  · congr 1
    push_cast
    exact halg _ _ _
  · ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual Gaussian characteristic derivative is its Gaussian times the negative direction dot.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, differentiating
the identity-covariance Gaussian characteristic function. -/
theorem fderiv_characteristic2_stdGauss2_apply (ξ v : ℝ × ℝ) :
    fderiv ℝ (characteristic2 stdGauss2) ξ v =
      -((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ) * characteristic2 stdGauss2 ξ := by
  have hfst : HasFDerivAt (fun η : ℝ × ℝ => η.1)
      (ContinuousLinearMap.fst ℝ ℝ ℝ) ξ := hasFDerivAt_fst
  have hsnd : HasFDerivAt (fun η : ℝ × ℝ => η.2)
      (ContinuousLinearMap.snd ℝ ℝ ℝ) ξ := hasFDerivAt_snd
  have hp := ((hfst.pow 2).add (hsnd.pow 2)).neg.const_smul ((2 : ℝ)⁻¹)
  have hd := Complex.ofRealCLM.hasFDerivAt.comp ξ hp.exp
  change HasFDerivAt (fun η : ℝ × ℝ =>
    ((Real.exp ((2 : ℝ)⁻¹ • -(η.1 ^ 2 + η.2 ^ 2)) : ℝ) : ℂ)) _ ξ at hd
  have he : characteristic2 stdGauss2 = fun η : ℝ × ℝ =>
      ((Real.exp ((2 : ℝ)⁻¹ • -(η.1 ^ 2 + η.2 ^ 2)) : ℝ) : ℂ) := by
    funext η
    rw [characteristic2_stdGauss2]
    congr 2
    simp only [smul_eq_mul]
    ring
  rw [he, hd.fderiv]
  simp [ContinuousLinearMap.comp_apply, smul_eq_mul]
  ring

/-- The actual Gaussian derivative differs from its linear coefficient cubically on the unit ball.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the first
Gaussian derivative in the cubic-corrected expansion. -/
theorem norm_fderiv_characteristic2_stdGauss2_add_dot_cubic_le
    (ξ v : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖fderiv ℝ (characteristic2 stdGauss2) ξ v +
      ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ)‖ ≤ 4 * ‖ξ‖ ^ 3 * ‖v‖ := by
  rw [fderiv_characteristic2_stdGauss2_apply]
  have he : -((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ) * characteristic2 stdGauss2 ξ +
      ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ) =
      ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ) * (1 - characteristic2 stdGauss2 ξ) := by ring
  rw [he, norm_mul, Complex.norm_real, Real.norm_eq_abs]
  have hG : ‖1 - characteristic2 stdGauss2 ξ‖ ≤ 2 * ‖ξ‖ ^ 2 := by
    rw [norm_sub_rev]
    exact norm_characteristic2_stdGauss2_sub_one ξ hξ
  calc
    _ ≤ (2 * ‖ξ‖ * ‖v‖) * (2 * ‖ξ‖ ^ 2) :=
      mul_le_mul (abs_dot_le_two_norm_mul ξ v) hG (norm_nonneg _) (by positivity)
    _ = _ := by ring

example : ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 := by norm_num

/-- Subtracting the literal cubic derivative leaves cubic Gaussian comparison error.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:br-9.10`, for one summand on the unit frequency ball. -/
theorem norm_fderiv_characteristic2_sub_gaussian_sub_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ v : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖fderiv ℝ (characteristic2 μ) ξ v - fderiv ℝ (characteristic2 stdGauss2) ξ v -
      fderiv ℝ (cubicCharacteristicCorrection μ) ξ v‖ ≤
      (8 * (∫ z, ‖z‖ ^ 4 ∂μ) + 4) * ‖ξ‖ ^ 3 * ‖v‖ := by
  let D : ℂ := ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ)
  have he : fderiv ℝ (characteristic2 μ) ξ v - fderiv ℝ (characteristic2 stdGauss2) ξ v -
      fderiv ℝ (cubicCharacteristicCorrection μ) ξ v =
      (fderiv ℝ (characteristic2 μ) ξ v + D - fderiv ℝ (cubicCharacteristicCorrection μ) ξ v) -
        (fderiv ℝ (characteristic2 stdGauss2) ξ v + D) := by ring
  rw [he]
  calc
    _ ≤ ‖fderiv ℝ (characteristic2 μ) ξ v + D - fderiv ℝ (cubicCharacteristicCorrection μ) ξ v‖ +
      ‖fderiv ℝ (characteristic2 stdGauss2) ξ v + D‖ := norm_sub_le _ _
    _ ≤ 8 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 3 * ‖v‖ + 4 * ‖ξ‖ ^ 3 * ‖v‖ :=
      add_le_add (norm_fderiv_characteristic2_add_dot_sub_cubic_le μ hμ hmom ξ v)
        (norm_fderiv_characteristic2_stdGauss2_add_dot_cubic_le ξ v hξ)
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), by norm_num⟩

/-- The full first-derivative error after the literal cubic subtraction is cubically bounded.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the operator
estimate used before the normalized-sum first-derivative product rule. -/
theorem norm_fderiv_characteristic2_sub_gaussian_sub_cubic
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖fderiv ℝ (characteristic2 μ) ξ - fderiv ℝ (characteristic2 stdGauss2) ξ -
      fderiv ℝ (cubicCharacteristicCorrection μ) ξ‖ ≤
      (8 * (∫ z, ‖z‖ ^ 4 ∂μ) + 4) * ‖ξ‖ ^ 3 := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 4 ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  exact fun v => norm_fderiv_characteristic2_sub_gaussian_sub_cubic_le μ hμ hmom ξ v hξ

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), by norm_num⟩

end Transformer.Modes
