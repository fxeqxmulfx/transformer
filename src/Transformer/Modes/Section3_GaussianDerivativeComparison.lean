import Transformer.Modes.Section3_DerivativeTaylor
import Transformer.Modes.Section3_SmallFrequency
/-!
# Gaussian comparison of the first two characteristic derivatives

The finite-third-moment estimates for one standardized summand are the
inputs to the product rule in arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
for the `s = 2` case of §3 `thm:br`. Both characteristic functions have
the same linear first-derivative term and constant second-derivative
term. Subtracting these terms gives quadratic and linear errors,
respectively, without an exponential-moment hypothesis.

These are estimates before taking the normalized-sum power. Operator
norm versions retain all directions, and the Gaussian first derivative
is small near zero. The second derivative of every standardized law
has a uniform bound using only its covariance and finite second moment.
The constants account for the product sup norm and are not sharp.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The first characteristic derivatives differ quadratically in frequency.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4 `eq:br-9.10`,
the first-derivative single-summand comparison. Both linear terms
cancel because the covariance matrices are the identity. -/
theorem norm_fderiv_characteristic2_sub_gaussian_apply_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ v : ℝ × ℝ) :
    ‖fderiv ℝ (characteristic2 μ) ξ v - fderiv ℝ (characteristic2 stdGauss2) ξ v‖ ≤
      8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2) * ‖ξ‖ ^ 2 * ‖v‖ := by
  let D : ℂ := ((ξ.1 * v.1 + ξ.2 * v.2 : ℝ) : ℂ)
  have hR := norm_fderiv_characteristic2_add_dot_le μ hμ hmom ξ v
  have hG := norm_fderiv_characteristic2_add_dot_le stdGauss2 isStandardized_stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) ξ v
  have he : fderiv ℝ (characteristic2 μ) ξ v - fderiv ℝ (characteristic2 stdGauss2) ξ v =
      (fderiv ℝ (characteristic2 μ) ξ v + D) -
        (fderiv ℝ (characteristic2 stdGauss2) ξ v + D) := by ring
  rw [he]
  calc
    _ ≤ ‖fderiv ℝ (characteristic2 μ) ξ v + D‖ +
        ‖fderiv ℝ (characteristic2 stdGauss2) ξ v + D‖ := norm_sub_le _ _
    _ ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 2 * ‖v‖ +
        8 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2) * ‖ξ‖ ^ 2 * ‖v‖ := add_le_add hR hG
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Operator-norm first-derivative comparison under a finite third moment.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the single-summand
first-derivative error used in the normalized-sum product rule.
The estimate holds in every direction simultaneously. -/
theorem norm_fderiv_characteristic2_sub_gaussian_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    ‖fderiv ℝ (characteristic2 μ) ξ - fderiv ℝ (characteristic2 stdGauss2) ξ‖ ≤
      8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2) * ‖ξ‖ ^ 2 := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hG : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  exact fun v => norm_fderiv_characteristic2_sub_gaussian_apply_le μ hμ hmom ξ v

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The second characteristic derivatives differ linearly in frequency.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, and §5.4 `eq:br-9.10`,
the second-derivative single-summand comparison. The constant covariance
terms cancel, leaving a remainder controlled by the third moments. -/
theorem norm_iteratedFDeriv_two_characteristic2_sub_gaussian_apply_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ v‖ ≤
        8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2) * ‖ξ‖ * ‖v 0‖ * ‖v 1‖ := by
  let D : ℂ := (((v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 : ℝ) : ℂ)
  have hR := norm_iteratedFDeriv_two_characteristic2_add_dot_le μ hμ hmom ξ v
  have hG := norm_iteratedFDeriv_two_characteristic2_add_dot_le stdGauss2 isStandardized_stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) ξ v
  have he : iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ v =
      (iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v + D) -
        (iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ v + D) := by ring
  rw [he]
  calc
    _ ≤ ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v + D‖ +
        ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ v + D‖ := norm_sub_le _ _
    _ ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ * ‖v 0‖ * ‖v 1‖ +
        8 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2) * ‖ξ‖ * ‖v 0‖ * ‖v 1‖ := add_le_add hR hG
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Operator-norm second-derivative comparison under a finite third moment.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the single-summand
second-derivative error used in the normalized-sum product rule.
The estimate holds for the full continuous multilinear derivative. -/
theorem norm_iteratedFDeriv_two_characteristic2_sub_gaussian_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ - iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ‖ ≤
      8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2) * ‖ξ‖ := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hG : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  rw [Fin.prod_univ_two]
  have h := norm_iteratedFDeriv_two_characteristic2_sub_gaussian_apply_le μ hμ hmom ξ v
  convert h using 1; ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The Gaussian first derivative is bounded by the frequency norm
on the unit ball. Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
the Gaussian derivative factor in the first-derivative power comparison.
The third Gaussian moment gives an explicit nonsharp constant. -/
theorem norm_fderiv_characteristic2_stdGauss2_small (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖fderiv ℝ (characteristic2 stdGauss2) ξ‖ ≤
      (2 + 8 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2)) * ‖ξ‖ := by
  exact norm_fderiv_characteristic2_small stdGauss2 isStandardized_stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) ξ hξ

example : ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 := by norm_num

/-- Identity covariance bounds the second characteristic derivative
uniformly in frequency. Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`,
the moment bound for second-derivative factors. The product sup-norm
second moment is at most the sum of the two variances, which is two. -/
theorem norm_iteratedFDeriv_two_characteristic2_of_standardized
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ‖ ≤ 8 := by
  have hI := hμ.memLp.integrable_norm_pow'
  have hI1 := hμ.memLp.fst.integrable_sq
  have hI2 := hμ.memLp.snd.integrable_sq
  have hM : (∫ z, ‖z‖ ^ 2 ∂μ) ≤ 2 := by
    have h := integral_mono_ae hI (hI1.add hI2) (ae_of_all _ norm_sq_le_dot_self)
    have he : (∫ z : ℝ × ℝ, z.1 ^ 2 + z.2 ^ 2 ∂μ) = 2 := by
      have ha := integral_add hI1 hI2
      change (∫ z : ℝ × ℝ, z.1 ^ 2 + z.2 ^ 2 ∂μ) =
        (∫ z : ℝ × ℝ, z.1 ^ 2 ∂μ) + ∫ z : ℝ × ℝ, z.2 ^ 2 ∂μ at ha
      rw [ha, hμ.var_fst, hμ.var_snd]
      norm_num
    exact h.trans_eq he
  have h := norm_iteratedFDeriv_characteristic2_le μ hμ.memLp ξ
  change ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ‖ ≤
    2 ^ 2 * (∫ z, ‖z‖ ^ 2 ∂μ) at h
  nlinarith [h]

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2⟩

end Transformer.Modes
