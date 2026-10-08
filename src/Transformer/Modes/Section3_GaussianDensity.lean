import Transformer.Modes.Section3_DensityComparison
import Transformer.Modes.Section3_IntegralComparison
/-!
# Uniform Gaussian approximation of the actual sum density

The zero-derivative part of arXiv:2412.09080v3, §3 `thm:br`,
`s = 2`, follows by combining the proved full Fourier error integral
with inversion. Under the original standardization, third-moment,
and integrable-characteristic-power hypotheses, actual continuous
densities exist eventually and satisfy `sup |q_n - φ| ≤ C / sqrt n`.

The Gaussian characteristic function and its actual density are
identified explicitly, including every normalization constant.
The spatially weighted estimate in `thm:br` additionally needs Fourier
derivative bounds; the unweighted conclusions here are separate results.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes
/-- The actual characteristic function of `N(0,I₂)` is
`exp(-|ξ|²/2)`, with positive characteristic-function phase.
Source: arXiv:2412.09080v3, §3 `thm:br`, the term `Q₀ = φ`. -/
theorem characteristic2_stdGauss2 (ξ : ℝ × ℝ) :
    characteristic2 stdGauss2 ξ = ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) := by
  have h := integral_prod_mul (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)
    (fun a => Complex.exp (ξ.1 * a * Complex.I)) (fun a => Complex.exp (ξ.2 * a * Complex.I))
  have e : (fun x : ℝ × ℝ => Complex.exp (((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ) * Complex.I)) =
      fun x => Complex.exp (ξ.1 * x.1 * Complex.I) * Complex.exp (ξ.2 * x.2 * Complex.I) := by
    funext x
    rw [← Complex.exp_add]
    push_cast
    ring_nf
  unfold characteristic2
  rw [e, h, ← charFun_apply_real, ← charFun_apply_real, charFun_gaussianReal,
    charFun_gaussianReal, ← Complex.exp_add, Complex.ofReal_exp]
  congr 1
  push_cast
  simp only [mul_zero, one_mul, zero_mul, zero_sub]
  ring
/-- The actual two-dimensional standard Gaussian characteristic
function is integrable. Source: arXiv:2412.09080v3, §5.4,
`eq:higher-error-goal` and `eq:big-phi-poly`. -/
theorem integrable_characteristic2_stdGauss2 : Integrable (characteristic2 stdGauss2) := by
  have hG : Integrable (fun ξ : ℝ × ℝ => Real.exp (-(1 / 2) * (ξ.1 ^ 2 + ξ.2 ^ 2))) := by
    simpa only [pow_zero, one_mul] using
      integrable_norm_pow_gaussian_frequency 0 (b := 1 / 2) (by norm_num)
  have he : characteristic2 stdGauss2 =
      fun ξ : ℝ × ℝ => ((Real.exp (-(1 / 2) * (ξ.1 ^ 2 + ξ.2 ^ 2)) : ℝ) : ℂ) := by
    funext ξ
    rw [characteristic2_stdGauss2]
    congr 1
    congr 1
    ring
  rw [he]
  exact hG.ofReal

/-- The paper's `φ` is the Lebesgue density of the actual product
standard Gaussian law. Source: arXiv:2412.09080v3, §3 `thm:br`, `Q₀ = φ`. -/
theorem stdGauss2_eq_withDensity_phi2 :
    stdGauss2 = volume.withDensity (fun z => ENNReal.ofReal (phi2 z)) := by
  have hS : scaledSum 1 = Function.eval 0 := by funext X; simp [scaledSum]
  have h := isDensityOf_stdGauss2.map_eq
  rw [hS, (measurePreserving_eval (μ := fun _ : Fin 1 => stdGauss2) 0).map_eq] at h
  exact h

/-- The proved inverse characteristic integral agrees pointwise
with the paper's Gaussian density. Source: arXiv:2412.09080v3,
§5.4 `eq:higher-error-goal`, the zero-derivative Gaussian term. -/
theorem inverseCharacteristic_complexLaw_stdGauss2 (x : ℝ × ℝ) :
    inverseCharacteristic (complexLaw stdGauss2) (Complex.measurableEquivRealProd.symm x) =
      phi2 x := by
  have h := continuous_density_eq_inverseCharacteristic stdGauss2 integrable_characteristic2_stdGauss2
    (show Continuous phi2 by unfold phi2; fun_prop) isDensityOf_stdGauss2.nonneg
    stdGauss2_eq_withDensity_phi2
  exact (congrFun h x).symm

/-- Every continuous density of the actual normalized sum has uniform
unweighted Gaussian error `O(1 / sqrt n)`. Source: arXiv:2412.09080v3,
§3 `thm:br`, `s = 2`, and §5.4 `eq:higher-error-goal`, the zero-derivative
consequence. The source's additional spatial weight is not asserted here. -/
theorem eventually_abs_scaledSum_density_sub_phi2
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ,
      Continuous q → IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, |q x - phi2 x| ≤ C * (Real.sqrt n)⁻¹ := by
  obtain ⟨C, hC, herror⟩ := eventually_integral_characteristic_scaledSum_sub_gaussian_rate μ hμ hmom hcf
  refine ⟨((2 * Real.pi) ^ 2)⁻¹ * C, by positivity, ?_⟩
  filter_upwards [herror, eventually_integrable_characteristic_scaledSum μ hcf] with n hn hIn
  intro q hq hqd x
  have h := continuous_density_sub_abs_le _ stdGauss2 hIn integrable_characteristic2_stdGauss2
    hq (show Continuous phi2 by unfold phi2; fun_prop) hqd.nonneg isDensityOf_stdGauss2.nonneg
    hqd.map_eq stdGauss2_eq_withDensity_phi2 x
  have hG : characteristic2 stdGauss2 = fun ξ : ℝ × ℝ =>
      ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) := funext characteristic2_stdGauss2
  rw [hG] at h
  calc
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ * ∫ ξ : ℝ × ℝ,
        ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
          ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ)‖ := h
    _ ≤ ((2 * Real.pi) ^ 2)⁻¹ * (C * (Real.sqrt n)⁻¹) := mul_le_mul_of_nonneg_left hn (by positivity)
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2⟩

/-- Actual continuous densities exist eventually and satisfy the
uniform unweighted Gaussian error bound. Source: arXiv:2412.09080v3,
§3 `thm:br`, `s = 2`, the zero-derivative consequence, using the separately
proved actual density existence rather than assuming such a density. -/
theorem eventually_exists_scaledSum_density_gaussian_rate
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ᶠ n : ℕ in atTop, ∃ q : ℝ × ℝ → ℝ,
      Continuous q ∧ IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q ∧
      ∀ x, |q x - phi2 x| ≤ C * (Real.sqrt n)⁻¹ := by
  obtain ⟨C, hC, herror⟩ := eventually_abs_scaledSum_density_sub_phi2 μ hμ hmom hcf
  refine ⟨C, hC, ?_⟩
  filter_upwards [herror, eventually_exists_continuous_density_scaledSum μ hcf] with n hn hex
  obtain ⟨q, hq, hqd, _⟩ := hex
  exact ⟨q, hq, hqd, hn q hq hqd⟩

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2⟩

/-- The actual continuous sum densities converge uniformly to `φ`.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`, the unweighted
consequence of its rate tending to zero. -/
theorem eventually_abs_scaledSum_density_sub_phi2_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (hcf : HasIntegrableCharFun μ) {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ n : ℕ in atTop, ∀ q : ℝ × ℝ → ℝ,
      Continuous q → IsDensityOf (Measure.pi fun _ : Fin n => μ) (scaledSum n) q →
      ∀ x, |q x - phi2 x| ≤ ε := by
  obtain ⟨C, _, herror⟩ := eventually_abs_scaledSum_density_sub_phi2 μ hμ hmom hcf
  have hlim : Tendsto (fun n : ℕ => C * (Real.sqrt (n : ℝ))⁻¹) atTop (nhds 0) := by
    simpa only [mul_zero, Function.comp_def] using
      (tendsto_inv_atTop_zero.comp (Real.tendsto_sqrt_atTop.comp
        (tendsto_natCast_atTop_atTop : Tendsto (fun n : ℕ => (n : ℝ)) atTop atTop))).const_mul C
  filter_upwards [herror, hlim.eventually (gt_mem_nhds hε)] with n hn hεn
  intro q hq hqd x
  exact (hn q hq hqd x).trans hεn.le

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 3 stdGauss2 ∧ HasIntegrableCharFun stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    hasIntegrableCharFun_stdGauss2, one_pos⟩

end Transformer.Modes
