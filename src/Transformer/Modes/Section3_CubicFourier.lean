import Transformer.Modes.Section3_HermiteFourier
import Transformer.Modes.Section3_CubicCorrection
import Transformer.Modes.Section3_GaussianDensity

/-!
# Fourier identity for the literal third-cumulant density correction

Section 3.1 of arXiv:2412.09080v3, `eq:psi`, defines `ψ` by four
Gaussian Hermite basis functions with the actual mixed logarithmic
cumulants. Its coefficients are constants in space, so the literal
function is smooth and integrable for every law. Its total integral is
zero, as appropriate for the first correction to a probability density.

Under the source's standardization and exponential-moment hypotheses,
the proved third-cumulant identity identifies its positive-sign phase
integral with `exp(-|ξ|²/2) B(ξ)`, where `B = -i E⟨ξ,X⟩³/6`.
The finite sum is passed through the actual integral only after proving
each term integrable. This is an identity for a signed correction;
no nonnegativity or probability-density assertion about `ψ` is assumed.

This repairs the §5.4 display before `eq:higher-error-goal`: its Hermite
polynomial in frequency must be the cubic monomial multiplier. The
spatial definition and all cumulant coefficients in `eq:psi` are retained.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- The literal correction is the finite sum of its weighted Gaussian Hermite basis.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem psiOf_eq_sum_weighted_hermite (μ : Measure (ℝ × ℝ)) (x : ℝ × ℝ) :
    psiOf μ x = ∑ k ∈ Finset.range 4, cumulantOf μ k (3 - k) /
      ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) * (phi2 x * hermite3 k x) := by
  unfold psiOf
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k hk
  ring

/-- Each recorded third-order Hermite polynomial is smooth.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem contDiff_hermite3 (k : ℕ) (hk : k ≤ 3) : ContDiff ℝ ⊤ (hermite3 k) := by
  change ContDiff ℝ ⊤ (fun x : ℝ × ℝ => hermite1 k x.1 * hermite1 (3 - k) x.2)
  interval_cases k <;> simp only [hermite1] <;> fun_prop

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The actual cumulant-weighted Gaussian correction is smooth.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem contDiff_psiOf (μ : Measure (ℝ × ℝ)) : ContDiff ℝ ⊤ (psiOf μ) := by
  unfold psiOf
  apply ContDiff.mul (by unfold phi2; fun_prop)
  apply ContDiff.sum
  intro k hk
  exact contDiff_const.mul
    (contDiff_hermite3 k (by have h := Finset.mem_range.mp hk; omega))

/-- The actual correction, included in the complex numbers, is integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_psiOf_complex (μ : Measure (ℝ × ℝ)) :
    Integrable (fun x : ℝ × ℝ => ((psiOf μ x : ℝ) : ℂ)) := by
  have hI (k : ℕ) (hk : k ∈ Finset.range 4) :=
    (integrable_hermite3_mul_phi2 k (by have h := Finset.mem_range.mp hk; omega)).const_mul
      ((cumulantOf μ k (3 - k) / ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) : ℝ) : ℂ)
  convert integrable_finsetSum (Finset.range 4) hI using 1
  funext x
  rw [psiOf_eq_sum_weighted_hermite]
  simp only [Complex.ofReal_sum, Complex.ofReal_mul]

/-- The actual correction remains integrable after multiplication by the characteristic phase.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_psiOf_characteristic (μ : Measure (ℝ × ℝ)) (ξ : ℝ × ℝ) :
    Integrable (fun x : ℝ × ℝ => Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ)) *
      ((psiOf μ x : ℝ) : ℂ)) := by
  have hc : Continuous (fun x : ℝ × ℝ =>
      Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ))) := by fun_prop
  exact (integrable_psiOf_complex μ).bdd_mul hc.aestronglyMeasurable
    (ae_of_all _ fun x => by rw [mul_comm Complex.I, Complex.norm_exp_ofReal_mul_I])

/-- The actual complex-valued correction has zero total integral.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integral_psiOf_complex_zero (μ : Measure (ℝ × ℝ)) :
    (∫ x : ℝ × ℝ, ((psiOf μ x : ℝ) : ℂ)) = 0 := by
  have he (x : ℝ × ℝ) : ((psiOf μ x : ℝ) : ℂ) =
      ∑ k ∈ Finset.range 4, ((cumulantOf μ k (3 - k) /
        ((k.factorial : ℝ) * ((3 - k).factorial : ℝ)) : ℝ) : ℂ) *
          ((phi2 x * hermite3 k x : ℝ) : ℂ) := by
    rw [psiOf_eq_sum_weighted_hermite, Complex.ofReal_sum]
    apply Finset.sum_congr rfl
    intro k hk
    exact Complex.ofReal_mul _ _
  simp_rw [he]
  rw [integral_finsetSum]
  · apply Finset.sum_eq_zero
    intro k hk
    rw [integral_const_mul, integral_hermite3_mul_phi2_zero k (by have h := Finset.mem_range.mp hk; omega), mul_zero]
  · intro k hk
    exact (integrable_hermite3_mul_phi2 k (by have h := Finset.mem_range.mp hk; omega)).const_mul _

/-- The actual positive-sign transform of `ψ` equals the Gaussian times the cubic phase coefficient.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integral_psiOf_characteristic_eq_gaussian_cubic
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hexp : HasExpMoments μ) (ξ : ℝ × ℝ) :
    (∫ x : ℝ × ℝ, Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ)) *
      ((psiOf μ x : ℝ) : ℂ)) = characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ := by
  let a (k : ℕ) : ℝ := cumulantOf μ k (3 - k) /
    ((k.factorial : ℝ) * ((3 - k).factorial : ℝ))
  have he (x : ℝ × ℝ) : Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ)) *
      ((psiOf μ x : ℝ) : ℂ) = ∑ k ∈ Finset.range 4, (a k : ℂ) *
        (Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ)) *
          ((phi2 x * hermite3 k x : ℝ) : ℂ)) := by
    rw [psiOf_eq_sum_weighted_hermite]
    simp only [Complex.ofReal_sum, Finset.mul_sum, Complex.ofReal_mul]
    apply Finset.sum_congr rfl
    intro k hk
    dsimp only [a]
    ring
  simp_rw [he]
  rw [integral_finsetSum (Finset.range 4) (fun k hk =>
    (integrable_hermite3_phi2_characteristic k (by have h := Finset.mem_range.mp hk; omega) ξ).const_mul (a k : ℂ))]
  simp only [integral_const_mul]
  have hT (k : ℕ) (hk : k ∈ Finset.range 4) := integral_hermite3_phi2_characteristic k (by have h := Finset.mem_range.mp hk; omega) ξ
  simp only [Complex.ofReal_mul] at hT
  rw [characteristic2_stdGauss2, cubicCharacteristicCorrection_eq_cumulant_polynomial μ hμ hexp]
  simp only [Complex.ofReal_sum, Complex.ofReal_mul, Complex.ofReal_pow, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro k hk
  rw [hT k hk]
  dsimp only [a]
  ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, hasExpMoments_stdGauss2⟩

/-- The actual real-valued correction is integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_psiOf (μ : Measure (ℝ × ℝ)) : Integrable (psiOf μ) := by
  simpa only [RCLike.re_to_complex, Complex.ofReal_re] using (integrable_psiOf_complex μ).re

/-- The actual real-valued correction has zero total integral.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integral_psiOf_zero (μ : Measure (ℝ × ℝ)) : (∫ x : ℝ × ℝ, psiOf μ x) = 0 := by
  have h := integral_psiOf_complex_zero μ
  have hR := congrArg (fun z : ℂ => RCLike.re z) h
  simpa only [RCLike.re_to_complex, Complex.ofReal_re, Complex.zero_re] using
    (integral_re (integrable_psiOf_complex μ)).trans hR

end Transformer.Modes
