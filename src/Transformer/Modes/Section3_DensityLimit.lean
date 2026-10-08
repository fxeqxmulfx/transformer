import Transformer.Modes.Section3_ComplexCharacteristic
import Transformer.Modes.Section3_FourierRegularization

/-!
# The actual inverse integral as a limit of Gaussian smoothings

Section 5.4–§5.5 of arXiv:2412.09080v3 appeals to Fourier inversion to
obtain a continuous density. Here the input is the integrability of the
actual law's characteristic function. Gaussian smoothing gives genuine
nonnegative integrable approximations of mass one without a density
assumption on that law.

Dominated convergence removes the Fourier Gaussian multiplier. The limit
is the literal inverse characteristic-function integral. Since the
approximations are real and nonnegative, the inverse integral has zero
imaginary part and nonnegative real part. Domination also gives continuity.
Fatou's lemma then gives mass at most one and integrability of this limit.
The next module invokes inversion to prove the actual measure identity
and total mass one; that identity is not a hypothesis here.

All the characteristic-function hypotheses below have concrete examples
from the actual normalized sum of seventeen `lawY 3 0` summands.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped FourierTransform Topology ENNReal

namespace Transformer.Modes

/-- The actual Gaussian smoothing converges to the inverse integral.
The integrable characteristic function dominates the Fourier integrands,
whose Gaussian multipliers tend pointwise to one. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem tendsto_gaussianRegularization2_fourier (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) (w : ℂ) :
    Tendsto (fun n : ℕ => (gaussianRegularization2 μ ((n : ℝ) + 1) w : ℂ)) atTop
      (𝓝 (𝓕 (normalizedCharacteristic μ) w)) := by
  let P : ℂ → ℂ := fun ξ =>
    Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I)
  let F : ℕ → ℂ → ℂ := fun n ξ =>
    P ξ * (gaussianDamping2 ((n : ℝ) + 1) ξ * normalizedCharacteristic μ ξ)
  have hmeas (n : ℕ) : AEStronglyMeasurable (F n) volume := by
    have hc : Continuous (F n) := by
      unfold F P gaussianDamping2 normalizedCharacteristic
      fun_prop
    exact hc.aestronglyMeasurable
  have hb (n : ℕ) : ∀ᵐ ξ, ‖F n ξ‖ ≤ ‖normalizedCharacteristic μ ξ‖ := by
    exact ae_of_all _ fun ξ => by
      dsimp only [F, P]
      rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul, norm_mul]
      exact mul_le_of_le_one_left (norm_nonneg _) (norm_gaussianDamping2_le_one (by positivity) ξ)
  have ht : Tendsto (fun n : ℕ => (n : ℝ) + 1) atTop atTop := by
    simpa only [add_comm] using tendsto_atTop_add_const_left atTop (1 : ℝ)
      tendsto_natCast_atTop_atTop
  have hlim : ∀ᵐ ξ, Tendsto (fun n => F n ξ) atTop
      (𝓝 (P ξ * normalizedCharacteristic μ ξ)) := by
    exact ae_of_all _ fun ξ => by
      have hd := (tendsto_gaussianDamping2 ξ).comp ht
      simpa only [F, one_mul, Function.comp_def] using
        (tendsto_const_nhds.mul (hd.mul_const (normalizedCharacteristic μ ξ)))
  have hD := tendsto_integral_of_dominated_convergence
    (fun ξ => ‖normalizedCharacteristic μ ξ‖) hmeas
    (integrable_normalizedCharacteristic μ hI).norm hb hlim
  have he (n : ℕ) : (gaussianRegularization2 μ ((n : ℝ) + 1) w : ℂ) = ∫ ξ, F n ξ := by
    rw [gaussianRegularization2_fourier μ (by positivity) w, Real.fourier_eq']
    rfl
  have he' : (∫ ξ, P ξ * normalizedCharacteristic μ ξ) =
      𝓕 (normalizedCharacteristic μ) w := by
    rw [Real.fourier_eq']
    rfl
  rw [he'] at hD
  exact hD.congr' (Eventually.of_forall fun n => (he n).symm)

example := tendsto_gaussianRegularization2_fourier _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)) 0

/-- Taking real parts transfers the actual smoothing limit.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem tendsto_gaussianRegularization2 (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) (w : ℂ) :
    Tendsto (fun n : ℕ => gaussianRegularization2 μ ((n : ℝ) + 1) w) atTop
      (𝓝 (inverseCharacteristic μ w)) := by
  have h := Complex.continuous_re.continuousAt.tendsto.comp
    (tendsto_gaussianRegularization2_fourier μ hI w)
  simpa only [Function.comp_def, Complex.ofReal_re, inverseCharacteristic] using h

example := tendsto_gaussianRegularization2 _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)) 0

/-- Nonnegativity follows from the nonnegative Gaussian smoothings
and their proved pointwise limit. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem inverseCharacteristic_nonneg (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) (w : ℂ) : 0 ≤ inverseCharacteristic μ w := by
  exact ge_of_tendsto' (tendsto_gaussianRegularization2 μ hI w)
    (fun n => gaussianRegularization2_nonneg μ (by positivity) w)

example := inverseCharacteristic_nonneg _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)) 0

/-- The inverse integral is real: its imaginary part is the limit
of the identically zero imaginary parts of the real smoothings.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem ofReal_inverseCharacteristic (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) (w : ℂ) :
    (inverseCharacteristic μ w : ℂ) = 𝓕 (normalizedCharacteristic μ) w := by
  apply Complex.ext
  · rfl
  · have h := Complex.continuous_im.continuousAt.tendsto.comp
      (tendsto_gaussianRegularization2_fourier μ hI w)
    have hz : (𝓕 (normalizedCharacteristic μ) w).im = 0 := by
      apply tendsto_nhds_unique h
      simpa only [Function.comp_def, Complex.ofReal_im] using
        (tendsto_const_nhds : Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (𝓝 0))
    simpa only [Complex.ofReal_im] using hz.symm

example := ofReal_inverseCharacteristic _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)) 0

/-- Fatou's lemma bounds the limit's mass by the mass-one Gaussian
smoothings. This is a bound on the actual nonnegative Lebesgue integral.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem lintegral_inverseCharacteristic_le_one (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) : ∫⁻ w, ENNReal.ofReal (inverseCharacteristic μ w) ≤ 1 := by
  let G : ℕ → ℂ → ℝ≥0∞ := fun n w =>
    ENNReal.ofReal (gaussianRegularization2 μ ((n : ℝ) + 1) w)
  have hm (n : ℕ) : Measurable (G n) :=
    ENNReal.continuous_ofReal.measurable.comp
      (continuous_gaussianRegularization2 μ (by positivity)).measurable
  have hl (w : ℂ) : liminf (fun n => G n w) atTop =
      ENNReal.ofReal (inverseCharacteristic μ w) := by
    apply Tendsto.liminf_eq
    exact ENNReal.continuous_ofReal.continuousAt.tendsto.comp
      (tendsto_gaussianRegularization2 μ hI w)
  have hi (n : ℕ) : (∫⁻ w, G n w) = 1 := by
    rw [← ofReal_integral_eq_lintegral_ofReal (integrable_gaussianRegularization2 μ (by positivity))
      (ae_of_all _ fun w => gaussianRegularization2_nonneg μ (by positivity) w),
      integral_gaussianRegularization2 μ (by positivity), ENNReal.ofReal_one]
  calc
    (∫⁻ w, ENNReal.ofReal (inverseCharacteristic μ w)) =
        ∫⁻ w, liminf (fun n => G n w) atTop := by
      exact lintegral_congr fun w => (hl w).symm
    _ ≤ liminf (fun n => ∫⁻ w, G n w) atTop := lintegral_liminf_le hm
    _ = 1 := by simp only [hi, liminf_const]

example := lintegral_inverseCharacteristic_le_one _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

/-- Continuity and the proved finite nonnegative integral give
integrability of the actual inverse function. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem integrable_inverseCharacteristic (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) : Integrable (inverseCharacteristic μ) := by
  refine ⟨(continuous_inverseCharacteristic μ hI).aestronglyMeasurable, ?_⟩
  apply (hasFiniteIntegral_iff_ofReal (ae_of_all _ (inverseCharacteristic_nonneg μ hI))).mpr
  exact (lintegral_inverseCharacteristic_le_one μ hI).trans_lt (by simp)

example := integrable_inverseCharacteristic _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

end Transformer.Modes
