import Transformer.Modes.Section3_CharacteristicDerivatives
/-!
# The first two characteristic derivatives and standardization

For the finite-moment Fourier argument in arXiv:2412.09080v3, §3
`thm:br` and §5.4 `eq:br-9.10`, the derivative expansion is anchored
by mean zero and identity covariance. This module specializes the
proved phase-integral formula to the first two derivatives and
computes their values at zero under exactly those hypotheses.

Polarization of the proved quadratic linear-form moment supplies
the mixed covariance identity. Thus no vanishing moment or covariance
property of the actual law is assumed beyond `IsStandardized`.
The resulting derivatives agree with those of `exp(-|ξ|²/2)` at zero;
quantitative derivative remainders require further moment estimates.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The first derivative of the actual characteristic function is
`I` times its linear-weighted phase integral. Source:
arXiv:2412.09080v3, §5.4, the Fourier derivative step in `eq:br-9.10`.
Only a finite first moment is needed for this formula. -/
theorem fderiv_characteristic2_apply
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 1 μ)
    (ξ v : ℝ × ℝ) : fderiv ℝ (characteristic2 μ) ξ v =
      Complex.I * ∫ z, ((z.1 * v.1 + z.2 * v.2 : ℝ) : ℂ) *
        Complex.exp (((z.1 * ξ.1 + z.2 * ξ.2 : ℝ) : ℂ) * Complex.I) ∂μ := by
  have h := iteratedFDeriv_characteristic2 μ (k := 1) (by simpa using hmom) ξ (fun _ : Fin 1 => v)
  simpa only [iteratedFDeriv_one_apply, Fin.prod_univ_one, pow_one] using h

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 1 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The second derivative is the negative phase integral with two
linear weights, with the sign from `I² = -1`. Source:
arXiv:2412.09080v3, §5.4, the Fourier derivative step in `eq:br-9.10`.
Only a finite second moment is needed for this formula. -/
theorem iteratedFDeriv_two_characteristic2_apply
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ)
    (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 (characteristic2 μ) ξ v =
      -∫ z, (((z.1 * (v 0).1 + z.2 * (v 0).2) *
        (z.1 * (v 1).1 + z.2 * (v 1).2) : ℝ) : ℂ) *
        Complex.exp (((z.1 * ξ.1 + z.2 * ξ.2 : ℝ) : ℂ) * Complex.I) ∂μ := by
  have h := iteratedFDeriv_characteristic2 μ hmom ξ v
  simpa only [Fin.prod_univ_two, Complex.I_sq, neg_one_mul] using h

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Identity covariance gives the mixed moment of any two linear
forms by polarization. Source: arXiv:2412.09080v3, §3 `thm:br`,
the identity covariance hypothesis, and §5.4 `eq:br-9.10`. -/
theorem integral_dot_mul_of_standardized
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (v w : ℝ × ℝ) :
    ∫ z, (v.1 * z.1 + v.2 * z.2) * (w.1 * z.1 + w.2 * z.2) ∂μ =
      v.1 * w.1 + v.2 * w.2 := by
  have hV := memLp_dot μ hμ.memLp v
  have hW := memLp_dot μ hμ.memLp w
  have hVW := hV.integrable_mul hW
  have h := integral_dot_sq_of_standardized μ hμ (v + w)
  have he : (fun z : ℝ × ℝ => ((v + w).1 * z.1 + (v + w).2 * z.2) ^ 2) =
      fun z => (v.1 * z.1 + v.2 * z.2) ^ 2 +
        2 * ((v.1 * z.1 + v.2 * z.2) * (w.1 * z.1 + w.2 * z.2)) +
        (w.1 * z.1 + w.2 * z.2) ^ 2 := by
    funext z
    simp only [Prod.fst_add, Prod.snd_add]
    ring
  rw [he] at h
  have ha := integral_add (hV.integrable_sq.add (hVW.const_mul 2)) hW.integrable_sq
  have hb := integral_add hV.integrable_sq (hVW.const_mul 2)
  have hc : (∫ z : ℝ × ℝ, (v.1 * z.1 + v.2 * z.2) ^ 2 +
      2 * ((v.1 * z.1 + v.2 * z.2) * (w.1 * z.1 + w.2 * z.2)) +
      (w.1 * z.1 + w.2 * z.2) ^ 2 ∂μ) =
      (∫ z, (v.1 * z.1 + v.2 * z.2) ^ 2 ∂μ) +
        2 * (∫ z, (v.1 * z.1 + v.2 * z.2) * (w.1 * z.1 + w.2 * z.2) ∂μ) +
        (∫ z, (w.1 * z.1 + w.2 * z.2) ^ 2 ∂μ) := by
    exact ha.trans (congrArg (fun u => u + ∫ z, (w.1 * z.1 + w.2 * z.2) ^ 2 ∂μ)
      (hb.trans (congrArg (fun u => (∫ z, (v.1 * z.1 + v.2 * z.2) ^ 2 ∂μ) + u)
        (integral_const_mul (2 : ℝ) (fun z : ℝ × ℝ =>
          (v.1 * z.1 + v.2 * z.2) * (w.1 * z.1 + w.2 * z.2))))))
  rw [hc, integral_dot_sq_of_standardized μ hμ v, integral_dot_sq_of_standardized μ hμ w] at h
  simp only [Prod.fst_add, Prod.snd_add] at h
  nlinarith [h]

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2⟩

/-- Mean zero makes the actual characteristic function's first
Fréchet derivative vanish at zero. Source: arXiv:2412.09080v3,
§3 `thm:br`, and §5.4, the first Gaussian Taylor coefficient. -/
theorem fderiv_characteristic2_zero
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ) :
    fderiv ℝ (characteristic2 μ) 0 = 0 := by
  apply ContinuousLinearMap.ext
  intro v
  change fderiv ℝ (characteristic2 μ) 0 v = 0
  rw [fderiv_characteristic2_apply μ (hμ.memLp.mono_exponent (by norm_num)) 0 v]
  simp only [Prod.fst_zero, Prod.snd_zero, mul_zero, add_zero, Complex.ofReal_zero, zero_mul,
    Complex.exp_zero, mul_one]
  rw [integral_complex_ofReal]
  have he : (fun z : ℝ × ℝ => z.1 * v.1 + z.2 * v.2) =
      fun z => v.1 * z.1 + v.2 * z.2 := by funext z; ring
  rw [he, integral_dot_of_standardized μ hμ v]
  simp

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2⟩

/-- Identity covariance makes the second actual characteristic
derivative at zero the negative Euclidean dot product of its two
directions. Source: arXiv:2412.09080v3, §3 `thm:br`, and §5.4,
the second Gaussian Taylor coefficient. -/
theorem iteratedFDeriv_two_characteristic2_zero
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (v : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 (characteristic2 μ) 0 v =
      -(((v 0).1 * (v 1).1 + (v 0).2 * (v 1).2 : ℝ) : ℂ) := by
  rw [iteratedFDeriv_two_characteristic2_apply μ hμ.memLp 0 v]
  simp only [Prod.fst_zero, Prod.snd_zero, mul_zero, add_zero, Complex.ofReal_zero, zero_mul,
    Complex.exp_zero, mul_one]
  rw [integral_complex_ofReal]
  have he : (fun z : ℝ × ℝ => (z.1 * (v 0).1 + z.2 * (v 0).2) *
      (z.1 * (v 1).1 + z.2 * (v 1).2)) =
      fun z => ((v 0).1 * z.1 + (v 0).2 * z.2) * ((v 1).1 * z.1 + (v 1).2 * z.2) := by
    funext z
    ring
  rw [he, integral_dot_mul_of_standardized μ hμ]

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2⟩

/-- The ordinary first derivative has the uniform finite-moment bound
needed by the normalized-sum product rule. Source: arXiv:2412.09080v3,
§5.4 `eq:big-z-exp`, its first-derivative moment factor. This specializes
the proved multilinear operator bound to the ordinary derivative. -/
theorem norm_fderiv_characteristic2_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 1 μ) (ξ : ℝ × ℝ) :
    ‖fderiv ℝ (characteristic2 μ) ξ‖ ≤ 2 * ∫ z, ‖z‖ ∂μ := by
  have h := norm_iteratedFDeriv_characteristic2_le μ (k := 1) (by simpa using hmom) ξ
  simpa only [norm_iteratedFDeriv_one, pow_one] using h

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 1 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
