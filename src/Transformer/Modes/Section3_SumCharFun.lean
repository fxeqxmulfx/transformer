import Transformer.Modes.Section3_YCharFun
import Transformer.Modes.Section3_ScaledSumMGF
import Mathlib.MeasureTheory.Measure.Haar.NormedSpace

/-!
# The characteristic function of the actual normalized sum

Section 3 of arXiv:2412.09080v3 uses the law of
`n^(-1/2) (Y₁ + ⋯ + Yₙ)`. The Fourier argument in §5.4 also decomposes
this sum into independent blocks, but its displayed block normalizations
are inconsistent. Here the frequency normalization is obtained directly
from the definition of the actual `scaledSum`.

The two-dimensional characteristic function is its literal positive-sign
exponential expectation. The linear form of `scaledSum` is a sum of
scaled linear forms. A complex exponential turns that sum into a finite
product, and integration under the finite product measure factors the
expectation. This proves the exact formula
`χ(Sₙ)(ξ) = χ(Y)(ξ / sqrt n)^n`, without assuming a density.

An integrable `n`th norm power remains integrable after the nonzero
frequency scaling for `n ≥ 1`. The exact formula then gives integrability
of the complex characteristic function of the actual normalized sum.
For `lawY β t`, the already proved power estimate applies whenever
`n > 4(β + 1)`, for every positive bandwidth and fixed translation.

For an arbitrary probability law satisfying `HasIntegrableCharFun`, all
sufficiently large powers are also integrable, since `|χ| ≤ 1`.
Consequently the actual normalized sums eventually have integrable
characteristic functions. This is the Fourier input to density inversion;
it does not assume the continuous densities requested elsewhere in §3.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4 `eq:big-z-exp`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology

namespace Transformer.Modes

/-- The actual positive-sign characteristic function of a law on `ℝ²`.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4, Fourier display. -/
noncomputable def characteristic2 (μ : Measure (ℝ × ℝ)) (ξ : ℝ × ℝ) : ℂ :=
  ∫ z, Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) ∂μ

/-- The unit-modulus integrand gives continuity by dominated convergence.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4. -/
theorem continuous_characteristic2 (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] :
    Continuous (characteristic2 μ) := by
  unfold characteristic2
  apply continuous_of_dominated (bound := fun _ => (1 : ℝ))
  · intro ξ
    have hc : Continuous fun z : ℝ × ℝ =>
        Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) := by fun_prop
    exact hc.aestronglyMeasurable
  · exact fun ξ => ae_of_all _ fun z => by simp [Complex.norm_exp]
  · exact integrable_const _
  · exact ae_of_all _ fun z => by fun_prop

/-- A probability law's characteristic function has norm at most one.
Source: arXiv:2412.09080v3, §5.4, the characteristic-function bound. -/
theorem norm_characteristic2_le_one (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ]
    (ξ : ℝ × ℝ) : ‖characteristic2 μ ξ‖ ≤ 1 := by
  have h := norm_integral_le_of_norm_le_const (μ := μ) (C := (1 : ℝ))
    (f := fun z : ℝ × ℝ => Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I))
    (ae_of_all _ fun z => by simp [Complex.norm_exp])
  simpa only [characteristic2, probReal_univ, mul_one] using h

example : IsProbabilityMeasure stdGauss2 := inferInstance

/-- The exact normalized-sum formula, retaining the frequency factor
`1 / sqrt n`. The source's independent-block argument has inconsistent
normalizing factors; this formula follows from `scaledSum` itself.
Source: arXiv:2412.09080v3, §3 and §5.4 `eq:big-z-exp`. -/
theorem characteristic_scaledSum (μ : Measure (ℝ × ℝ)) [SigmaFinite μ]
    (n : ℕ) (ξ : ℝ × ℝ) :
    characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ =
      characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ n := by
  have hc : Continuous fun z : ℝ × ℝ =>
      Complex.exp (((ξ.1 * z.1 + ξ.2 * z.2 : ℝ) : ℂ) * Complex.I) := by fun_prop
  unfold characteristic2
  rw [integral_map (measurable_scaledSum n).aemeasurable hc.aestronglyMeasurable]
  have he (X : Fin n → ℝ × ℝ) :
      Complex.exp (((ξ.1 * (scaledSum n X).1 + ξ.2 * (scaledSum n X).2 : ℝ) : ℂ) * Complex.I) =
        ∏ i, Complex.exp (((((Real.sqrt n)⁻¹ • ξ).1 * (X i).1 +
          ((Real.sqrt n)⁻¹ • ξ).2 * (X i).2 : ℝ) : ℂ) * Complex.I) := by
    rw [scaledSum_dot, Complex.ofReal_sum, Finset.sum_mul, Complex.exp_sum]
    apply Finset.prod_congr rfl
    intro i hi
    apply congrArg Complex.exp
    simp only [Prod.smul_fst, Prod.smul_snd, smul_eq_mul, Complex.ofReal_add, Complex.ofReal_mul]
    ring
  rw [integral_congr_ae (ae_of_all _ he)]
  simpa only [Fintype.card_fin] using integral_fintype_prod_eq_pow (ι := Fin n) (μ := μ)
    (fun z : ℝ × ℝ => Complex.exp (((((Real.sqrt n)⁻¹ • ξ).1 * z.1 +
      ((Real.sqrt n)⁻¹ • ξ).2 * z.2 : ℝ) : ℂ) * Complex.I))

example : SigmaFinite stdGauss2 := inferInstance

/-- An integrable norm power gives an integrable characteristic function
of the actual normalized sum, after its invertible frequency scaling.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4. -/
theorem integrable_characteristic_scaledSum_of_power
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {n : ℕ} (hn : 1 ≤ n)
    (hpow : Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ n)) :
    Integrable (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hρ : (Real.sqrt (n : ℝ))⁻¹ ≠ 0 := inv_ne_zero (Real.sqrt_pos.mpr hn0).ne'
  have hint := hpow.comp_smul hρ
  have hc : Continuous (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) := by
    have heq : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
        fun ξ => characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ n := by
      funext ξ
      exact characteristic_scaledSum μ n ξ
    rw [heq]
    exact ((continuous_characteristic2 μ).comp (continuous_const_smul (Real.sqrt n)⁻¹)).pow n
  apply hint.mono' hc.aestronglyMeasurable
  exact ae_of_all _ fun ξ => by rw [characteristic_scaledSum, norm_pow]

example : IsProbabilityMeasure (lawY 3 0) ∧ 1 ≤ (17 : ℕ) ∧
    Integrable (fun ξ => ‖characteristic2 (lawY 3 0) ξ‖ ^ 17) := by
  refine ⟨inferInstance, by norm_num, ?_⟩
  exact integrable_characteristic_lawY_pow (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)

/-- The standardized normalized sum has an integrable characteristic
function for `n > 4(β + 1)`. The source uses a fixed five summands;
this corrected bandwidth-dependent threshold is derived from the actual
Fourier estimate. Source: arXiv:2412.09080v3, §5.4–§5.5. -/
theorem integrable_characteristic_scaledSum_lawY {β : ℝ} (hβ : 0 < β) (t : ℝ)
    {n : ℕ} (hn : 4 * (β + 1) < (n : ℝ)) :
    Integrable (characteristic2 ((Measure.pi fun _ : Fin n => lawY β t).map (scaledSum n))) := by
  have hnpos : 0 < n := by
    have : (0 : ℝ) < (n : ℝ) := by linarith
    exact_mod_cast this
  exact integrable_characteristic_scaledSum_of_power (lawY β t) hnpos
    (integrable_characteristic_lawY_pow hβ t hn)

example : (0 : ℝ) < 3 ∧ 4 * (3 + 1) < ((17 : ℕ) : ℝ) := by norm_num

/-- Every probability law with some integrable characteristic norm power
has integrable characteristic functions for all sufficiently large
normalized sums. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4. -/
theorem eventually_integrable_characteristic_scaledSum
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop,
      Integrable (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) := by
  obtain ⟨ν, hν, hI⟩ := hcf
  change Integrable (fun ξ => ‖characteristic2 μ ξ‖ ^ ν) at hI
  filter_upwards [eventually_ge_atTop ν] with n hn
  apply integrable_characteristic_scaledSum_of_power μ (hν.trans hn)
  apply hI.mono' ((continuous_characteristic2 μ).norm.pow n).aestronglyMeasurable
  exact ae_of_all _ fun ξ => by
    change ‖‖characteristic2 μ ξ‖ ^ n‖ ≤ _
    rw [Real.norm_eq_abs, abs_of_nonneg (pow_nonneg (norm_nonneg _) n)]
    exact pow_le_pow_of_le_one (norm_nonneg _) (norm_characteristic2_le_one μ ξ) hn

example : IsProbabilityMeasure stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
