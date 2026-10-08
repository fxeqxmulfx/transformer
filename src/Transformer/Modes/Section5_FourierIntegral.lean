import Transformer.Modes.Section5_PhaseSummation

/-!
# The Fourier decay of the Gaussian curve law

This module proves the corrected fixed-`t` version of `eq:uniform-decay`
in arXiv:2412.09080v3, §5.5, for `0 < β < 2`. The Fourier transform is
the original expectation under the standard Gaussian, with phase
`ξ₁ G(t) + ξ₂ G'(t)`; its definition is unchanged.

The change of variables `u = t - x` makes the phase a linear combination
of `g(u)` and `g'(u)`, while retaining the shifted Gaussian amplitude
`exp (-(u - t)²/2)`. The exact integral identity includes the normalization
`1 / sqrt (2π)`. It holds for every real bandwidth and frequency, without
any decay assumption.

For a nonzero frequency vector, normalize by its product norm. The
whole-line estimate from `Section5_PhaseSummation` then gives a uniform
inverse square root bound. At frequencies of norm at most one, use the
unit-norm bound for the original Gaussian expectation. Combining both
regimes proves the stated bound with `sqrt (1 + ‖ξ‖)` in the denominator.

The source prints the estimate for every positive `β` with a constant
depending only on `β`. The constant here also depends on the fixed `t`,
and `β < 2` is required when summing the Gaussian envelopes. Those source
errors have proved counterexamples in `Section5_PtBddFourier` and
`Section5_PtBddDecayFalse`; the borderline `β = 2` is not asserted.
Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`.
-/

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Modes

/-- `𝓕ν_t(ξ) = 𝔼 e^{-i(ξ₁ G(t) + ξ₂ G'(t))}`, the Fourier transform of the law
`ν_t` of `(G(t), G'(t))`. Source: arXiv:2412.09080v3, §5.5. -/
noncomputable def fourierNu (β t : ℝ) (ξ : ℝ × ℝ) : ℂ :=
  ∫ x, Complex.exp (-(Complex.I * ((ξ.1 * bigG β t x + ξ.2 * bigG' β t x : ℝ) : ℂ)))
    ∂gaussianReal 0 1

/-- The Gaussian expectation of a unit-modulus exponential is bounded by
one, including at zero frequency. Source: arXiv:2412.09080v3, §5.5. -/
theorem norm_fourierNu_le_one (β t : ℝ) (ξ : ℝ × ℝ) : ‖fourierNu β t ξ‖ ≤ 1 := by
  have h := norm_integral_le_of_norm_le_const (μ := gaussianReal 0 1) (C := (1 : ℝ))
    (f := fun x => Complex.exp (-(Complex.I * ((ξ.1 * bigG β t x + ξ.2 * bigG' β t x : ℝ) : ℂ))))
    (ae_of_all _ fun x => by simp [Complex.norm_exp])
  simpa only [fourierNu, probReal_univ, mul_one] using h

/-- The exact Fourier integral after `u = t - x`, with the shifted weight
and Gaussian normalization retained. The source's reduction to `t = 0`
omits this shift. Source: arXiv:2412.09080v3, §5.5, the Fourier display. -/
theorem fourierNu_eq_gaussian_phase_integral (β ρ t : ℝ) (θ : ℝ × ℝ) :
    fourierNu β t (ρ * θ.1, ρ * θ.2) =
      (Real.sqrt (2 * Real.pi))⁻¹ • ∫ u : ℝ, gaussianPhaseIntegrand β ρ θ t u := by
  unfold fourierNu
  rw [integral_gaussianReal_eq_integral_smul one_ne_zero]
  have he (x : ℝ) : gaussianPDFReal 0 1 x •
      Complex.exp (-(Complex.I * (((ρ * θ.1) * bigG β t x + (ρ * θ.2) * bigG' β t x : ℝ) : ℂ))) =
      (Real.sqrt (2 * Real.pi))⁻¹ • gaussianPhaseIntegrand β ρ θ t (t - x) := by
    have hpdf : gaussianPDFReal 0 1 x =
        (Real.sqrt (2 * Real.pi))⁻¹ * gaussianPhaseAmplitude t (t - x) := by
      simp only [gaussianPDFReal_def, NNReal.coe_one, mul_one, sub_zero, gaussianPhaseAmplitude]
      rw [show -(x ^ 2) / 2 = -(1 / 2 : ℝ) * (t - x - t) ^ 2 by ring]
    have hphase : (ρ * θ.1) * bigG β t x + (ρ * θ.2) * bigG' β t x =
        ρ * (θ.1 * bigG β (t - x) 0 + θ.2 * bigG' β (t - x) 0) := by
      simp only [bigG, bigG', sub_zero]
      ring
    rw [hpdf, hphase]
    simp only [gaussianPhaseIntegrand, oscillatoryKernel, Complex.real_smul,
      Complex.ofReal_mul, neg_mul]
    ring
  calc _ = ∫ x : ℝ, (Real.sqrt (2 * Real.pi))⁻¹ • gaussianPhaseIntegrand β ρ θ t (t - x) :=
        integral_congr_ae (ae_of_all _ he)
    _ = (Real.sqrt (2 * Real.pi))⁻¹ • ∫ x : ℝ, gaussianPhaseIntegrand β ρ θ t (t - x) :=
        integral_smul _ _
    _ = _ := by rw [integral_sub_left_eq_self (gaussianPhaseIntegrand β ρ θ t) volume t]

/-- Dividing a nonzero frequency vector by its norm gives a direction of
max norm one. Source: arXiv:2412.09080v3, §5.5, the factorization `ξ = ρω`. -/
theorem normalized_fourier_direction {ξ : ℝ × ℝ} (hξ : 0 < ‖ξ‖) :
    max |ξ.1 / ‖ξ‖| |ξ.2 / ‖ξ‖| = 1 := by
  rw [abs_div, abs_div, abs_of_pos hξ, max_div_div_right hξ.le,
    ← Real.norm_eq_abs, ← Real.norm_eq_abs, ← Prod.norm_def, div_self hξ.ne']

example : (0 : ℝ) < ‖((1, 0) : ℝ × ℝ)‖ := by
  norm_num [Prod.norm_def, Real.norm_eq_abs]

/-- For a fixed `t` and `0 < β < 2`, the whole-line phase estimate gives
Fourier decay at every nonzero frequency.
Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`. -/
theorem fourierNu_decay_nonzero {β : ℝ} (hβ : 0 < β) (hβ2 : β < 2) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ξ : ℝ × ℝ, 0 < ‖ξ‖ →
      ‖fourierNu β t ξ‖ ≤ C / Real.sqrt ‖ξ‖ := by
  obtain ⟨C, hC, hbound⟩ := gaussian_phase_integral_bound hβ hβ2 t
  let A := (Real.sqrt (2 * Real.pi))⁻¹
  have hA : 0 < A := by dsimp [A]; positivity
  refine ⟨A * C, mul_pos hA hC, ?_⟩
  intro ξ hξ
  let θ : ℝ × ℝ := (ξ.1 / ‖ξ‖, ξ.2 / ‖ξ‖)
  have hθ : max |θ.1| |θ.2| = 1 := normalized_fourier_direction hξ
  have hrepr : (‖ξ‖ * θ.1, ‖ξ‖ * θ.2) = ξ := by
    apply Prod.ext
    · exact mul_div_cancel₀ _ hξ.ne'
    · exact mul_div_cancel₀ _ hξ.ne'
  have heq := fourierNu_eq_gaussian_phase_integral β ‖ξ‖ t θ
  rw [hrepr] at heq
  rw [heq, norm_smul, Real.norm_eq_abs, abs_of_pos hA]
  calc _ ≤ A * (C / Real.sqrt ‖ξ‖) :=
        mul_le_mul_of_nonneg_left (by simpa only [abs_of_pos hξ] using hbound ‖ξ‖ hξ.ne' θ hθ) hA.le
    _ = _ := by ring

example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧ (0 : ℝ) < ‖((1, 0) : ℝ × ℝ)‖ := by
  norm_num [Prod.norm_def, Real.norm_eq_abs]

/-- **Equation (eq:uniform-decay)**, corrected for a fixed `t` and `0 < β < 2`:
`|𝓕ν_t(ξ)| ≲ (1 + ‖ξ‖)^{-1/2}` for every frequency vector.

The source states this for every positive `β`, with a constant depending
only on `β`. The constant must also depend on `t` (`not_uniform_decay`),
and the estimate fails for every `β > 2` (`not_uniform_decay_of_two_lt`).
The hypothesis `β < 2` is the correction; the borderline is not claimed.
The proof retains the shifted Gaussian weight and sums every unit interval.
Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`. -/
theorem uniform_decay {β : ℝ} (hβ : 0 < β) (hβ2 : β < 2) (t : ℝ) :
    ∃ C : ℝ, ∀ ξ : ℝ × ℝ, ‖fourierNu β t ξ‖ ≤ C / Real.sqrt (1 + ‖ξ‖) := by
  obtain ⟨C, hC, hdecay⟩ := fourierNu_decay_nonzero hβ hβ2 t
  refine ⟨Real.sqrt 2 * (C + 1), ?_⟩
  intro ξ
  have hroot : 0 < Real.sqrt (1 + ‖ξ‖) := Real.sqrt_pos.mpr (by positivity)
  by_cases hsmall : ‖ξ‖ ≤ 1
  · apply (norm_fourierNu_le_one β t ξ).trans
    apply (le_div_iff₀ hroot).mpr
    calc 1 * Real.sqrt (1 + ‖ξ‖) ≤ Real.sqrt 2 := by
          simpa only [one_mul] using Real.sqrt_le_sqrt (by linarith : 1 + ‖ξ‖ ≤ 2)
      _ ≤ Real.sqrt 2 * (C + 1) := by
        have h := mul_nonneg (Real.sqrt_nonneg 2) hC.le
        nlinarith
  · have hlarge : 1 ≤ ‖ξ‖ := (not_le.mp hsmall).le
    have hξ : 0 < ‖ξ‖ := zero_lt_one.trans_le hlarge
    have hs : Real.sqrt (1 + ‖ξ‖) ≤ Real.sqrt 2 * Real.sqrt ‖ξ‖ := by
      rw [← Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2)]
      apply Real.sqrt_le_sqrt
      linarith
    apply (hdecay ξ hξ).trans
    apply (div_le_div_iff₀ (Real.sqrt_pos.mpr hξ) hroot).mpr
    calc _ ≤ C * (Real.sqrt 2 * Real.sqrt ‖ξ‖) := mul_le_mul_of_nonneg_left hs hC.le
      _ ≤ (C + 1) * (Real.sqrt 2 * Real.sqrt ‖ξ‖) :=
        mul_le_mul_of_nonneg_right (by linarith) (by positivity)
      _ = _ := by ring

/-- The corrected decay hypotheses are satisfiable at `β = 1`. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 := ⟨one_pos, one_lt_two⟩

end Transformer.Modes
