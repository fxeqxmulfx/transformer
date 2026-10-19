module

public import Transformer.Modes.Section3_CubicHigherDerivatives
public import Transformer.Modes.Section3_WeightedDotMoments

/-!
# Fourth-moment errors for the second and third characteristic derivatives

The `s = 3` expansion in arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
retains the derivatives of the literal cubic correction. At derivative
order two, the weighted phase is expanded through degree one. The
constant moment is the identity covariance, and the linear moment is
exactly the second correction derivative. The remaining error is
quadratic in frequency and uses the fourth norm moment.

At derivative order three, subtracting the constant third derivative
of the correction leaves the change of the characteristic derivative
from zero. One further moment bounds this change linearly in frequency.
All estimates keep every direction norm, followed by full multilinear
operator bounds. They apply before Gaussian comparison or sum scaling.

The positive-sign phase has the coefficients `i² = -1` and `i³ = -i`.
Accordingly, if `d(v,z) = z₁ v₁ + z₂ v₂`, the second derivative
expansion is `-E[d(v₀,X)d(v₁,X)] - i E[d(v₀,X)d(v₁,X)d(ξ,X)]`
plus its remainder. The first expectation is the direction dot
product; the second term is the actual second derivative of `B`.

The phase remainder uses total moment order `k + m + 1 = 4` in
both cases: `(k,m) = (2,1)` and `(3,0)`. No fifth moment or higher
characteristic derivative enters either estimate. The operator bounds
follow from the applied estimates without restricting the directions.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The second derivative of the literal correction is its three-dot moment with coefficient `-i`.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the second differentiated positive-sign characteristic correction. -/
theorem iteratedFDeriv_two_cubicCharacteristicCorrection_moment
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ v =
      -Complex.I * ((∫ z, (z.1 * (v 0).1 + z.2 * (v 0).2) *
        (z.1 * (v 1).1 + z.2 * (v 1).2) * (z.1 * ξ.1 + z.2 * ξ.2) ∂μ : ℝ) : ℂ) := by
  rw [iteratedFDeriv_two_cubicCharacteristicCorrection_apply μ hmom,
    iteratedFDeriv_characteristic2 μ hmom]
  simp only [Prod.fst_zero, Prod.snd_zero, mul_zero, add_zero, Complex.ofReal_zero,
    zero_mul, Complex.exp_zero, mul_one, integral_complex_ofReal,
    Fin.prod_univ_three]
  norm_num [pow_succ, Complex.I_sq]

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The second characteristic derivative has quadratic error after retaining covariance and the cubic term.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:br-9.10`. The nonsharp constant sixteen is `2^4 / 1!`
for the literal weighted phase remainder in the product sup norm. -/
theorem norm_iteratedFDeriv_two_characteristic2_add_dot_sub_cubic_apply_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v +
      (((v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 : ℝ) : ℂ) -
        iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ v‖ ≤
      16 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 2 * (∏ i, ‖v i‖) := by
  have h := norm_iteratedFDeriv_characteristic2_sub_phase_taylor μ 2 1
    (by simpa using hmom) ξ v
  simp only [Fin.prod_univ_two] at h
  have hcov : ∫ z, (z.1 * (v 0).1 + z.2 * (v 0).2) *
      (z.1 * (v 1).1 + z.2 * (v 1).2) ∂μ =
        (v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 := by
    convert integral_dot_mul_of_standardized μ hμ (v 0) (v 1) using 1
    congr 1
    funext z
    ring
  norm_num [Finset.sum_range_succ, Complex.I_sq, hcov] at h
  rw [iteratedFDeriv_two_cubicCharacteristicCorrection_moment μ
    (hmom.mono_exponent (by norm_num))]
  convert h using 1
  · congr 1
    push_cast
    ring
  · simp only [Fin.prod_univ_two]
    ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The third characteristic derivative has linear fourth-moment error after the actual cubic subtraction.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:br-9.10`. The subtracted term is the actual third derivative
at zero, and the remainder retains all three direction norms. -/
theorem norm_iteratedFDeriv_three_characteristic2_sub_cubic_apply_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 4 μ)
    (ξ : ℝ × ℝ) (v : Fin 3 → ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 3 (characteristic2 μ) ξ v -
      iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ v‖ ≤
      16 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ * (∏ i, ‖v i‖) := by
  rw [iteratedFDeriv_three_cubicCharacteristicCorrection_apply μ
    (hmom.mono_exponent (by norm_num))]
  have h := norm_iteratedFDeriv_characteristic2_sub_zero_apply_le μ 3
    (by simpa using hmom) ξ v
  calc
    _ ≤ 2 ^ (3 + 1) * (∏ i, ‖v i‖) * ‖ξ‖ * ∫ z, ‖z‖ ^ (3 + 1) ∂μ := h
    _ = _ := by norm_num; ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The second characteristic Taylor error has the same quadratic bound in full multilinear norm.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the operator
form of the second-derivative unscaled remainder. At zero the
standardized characteristic derivative is exactly the covariance term. -/
theorem norm_iteratedFDeriv_two_characteristic2_sub_zero_sub_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ -
      iteratedFDeriv ℝ 2 (characteristic2 μ) 0 -
        iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ‖ ≤
      16 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 2 := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 4 ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  simp only [sub_apply]
  rw [iteratedFDeriv_two_characteristic2_zero μ hμ]
  simpa only [sub_neg_eq_add] using
    norm_iteratedFDeriv_two_characteristic2_add_dot_sub_cubic_apply_le μ hμ hmom ξ v

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The third characteristic Taylor error has a linear frequency bound in full multilinear norm.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the operator
form of the third-derivative unscaled remainder, controlled by
the fourth norm moment without an exponential-moment assumption. -/
theorem norm_iteratedFDeriv_three_characteristic2_sub_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 4 μ)
    (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 3 (characteristic2 μ) ξ -
      iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ‖ ≤
      16 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 4 ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  exact fun v => norm_iteratedFDeriv_three_characteristic2_sub_cubic_apply_le μ hmom ξ v

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
