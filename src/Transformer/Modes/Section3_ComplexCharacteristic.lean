import Transformer.Modes.Section3_SumCharFun
import Mathlib.Analysis.Fourier.Inversion
import Mathlib.MeasureTheory.Measure.CharacteristicFunction.TaylorExpansion
import Mathlib.MeasureTheory.Measure.Lebesgue.Complex

/-!
# Complex coordinates for the actual two-dimensional law

Section 3 of arXiv:2412.09080v3 works on `ℝ × ℝ`. Mathlib's Fourier
inversion theorem uses a finite-dimensional real inner product space.
The product norm on `ℝ × ℝ` is the maximum norm, so we explicitly use
`ℂ` with its two-dimensional Euclidean norm and transport the actual law.

The standard measurable coordinate equivalence preserves Lebesgue volume.
Its inner product is exactly the real dot product in the paper. Thus the
literal characteristic function, its integrability, and the resulting
density all transfer without any rescaling of the measure or Jacobian.
The only frequency rescaling is `2π`, required by the differing Fourier
convention. These facts are proved for actual measures and densities.

In particular the already proved bound for the actual normalized sum of
`lawY β t` transfers for `n > 4(β + 1)`. No continuous density is assumed
in this coordinate change. The literal inverse characteristic integral
is also defined here, and continuity and its global norm bound are proved.
Its reality, positivity, integrability, and measure identity follow in the
next modules from actual Gaussian smoothing and Fourier inversion.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped FourierTransform Topology ENNReal

namespace Transformer.Modes

/-- The actual law in complex coordinates, via the measurable equivalence
with `ℝ²`. Source: arXiv:2412.09080v3, §3 and §5.4–§5.5, Fourier inversion. -/
noncomputable def complexLaw (μ : Measure (ℝ × ℝ)) : Measure ℂ :=
  μ.map Complex.measurableEquivRealProd.symm

/-- Changing coordinates preserves a probability law.
Source: arXiv:2412.09080v3, §3, the law of the normalized sum. -/
instance complexLaw_isProbabilityMeasure (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] :
    IsProbabilityMeasure (complexLaw μ) := by
  unfold complexLaw
  infer_instance

example : IsProbabilityMeasure (complexLaw stdGauss2) := inferInstance

/-- Complex-coordinate inner products give the literal two-dimensional
characteristic function. The sign is positive in both expressions.
Source: arXiv:2412.09080v3, §3 and §5.4–§5.5. -/
theorem charFun_complexLaw (μ : Measure (ℝ × ℝ)) (ξ : ℂ) :
    charFun (complexLaw μ) ξ = characteristic2 μ (ξ.re, ξ.im) := by
  rw [charFun_apply, complexLaw, integral_map
    Complex.measurableEquivRealProd.symm.measurable.aemeasurable]
  · apply integral_congr_ae
    exact ae_of_all _ fun z => by
      simp [Complex.measurableEquivRealProd, Complex.inner, Complex.mul_re]
  · exact (by fun_prop : Continuous fun z : ℂ =>
      Complex.exp (((inner ℝ z ξ : ℝ) : ℂ) * Complex.I)).aestronglyMeasurable

/-- The volume-preserving coordinate equivalence retains characteristic
function integrability. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem integrable_charFun_complexLaw (μ : Measure (ℝ × ℝ))
    (hI : Integrable (characteristic2 μ)) : Integrable (charFun (complexLaw μ)) := by
  have h := Complex.volume_preserving_equiv_real_prod.integrable_comp_of_integrable hI
  have he : charFun (complexLaw μ) = characteristic2 μ ∘ Complex.measurableEquivRealProd := by
    funext ξ
    exact charFun_complexLaw μ ξ
  rw [he]
  exact h

example : Integrable (characteristic2
    ((Measure.pi fun _ : Fin 17 => lawY 3 0).map (scaledSum 17))) :=
  integrable_characteristic_scaledSum_lawY (by norm_num : (0 : ℝ) < 3) 0 (by norm_num)

/-- The positive-sign characteristic function at `2π ξ`. This frequency
factor matches Mathlib's `exp(-2π i ⟨ξ,x⟩)` Fourier convention.
The source uses `exp(-i ⟨ξ,x⟩)`. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
noncomputable def normalizedCharacteristic (μ : Measure ℂ) (ξ : ℂ) : ℂ :=
  charFun μ ((2 * Real.pi) • ξ)

/-- The nonzero `2π` frequency scaling preserves integrability.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem integrable_normalizedCharacteristic (μ : Measure ℂ)
    (hI : Integrable (charFun μ)) : Integrable (normalizedCharacteristic μ) := by
  exact hI.comp_smul (by positivity : (2 * Real.pi : ℝ) ≠ 0)

example : Integrable (charFun (complexLaw
    ((Measure.pi fun _ : Fin 17 => lawY 3 0).map (scaledSum 17)))) :=
  integrable_charFun_complexLaw _
    (integrable_characteristic_scaledSum_lawY (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

/-- Mapping the actual law back recovers the original measure.
Source: arXiv:2412.09080v3, §3 and §5.4–§5.5. -/
theorem complexLaw_map_realProd (μ : Measure (ℝ × ℝ)) :
    (complexLaw μ).map Complex.measurableEquivRealProd = μ := by
  unfold complexLaw
  rw [Measure.map_map Complex.measurableEquivRealProd.measurable
    Complex.measurableEquivRealProd.symm.measurable]
  have he : Complex.measurableEquivRealProd ∘ Complex.measurableEquivRealProd.symm = id := by
    funext z
    exact Complex.measurableEquivRealProd.apply_symm_apply z
  rw [he, Measure.map_id]

/-- A density in complex coordinates is the density in `ℝ²`, composed
with the inverse coordinate equivalence. The Jacobian factor is exactly one.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem map_realProd_withDensity (q : ℂ → ℝ) (hq : Measurable q) :
    (volume.withDensity (fun z => ENNReal.ofReal (q z))).map Complex.measurableEquivRealProd =
      volume.withDensity (fun z => ENNReal.ofReal (q (Complex.measurableEquivRealProd.symm z))) := by
  apply Measure.ext
  intro s hs
  rw [Measure.map_apply Complex.measurableEquivRealProd.measurable hs,
    withDensity_apply _ (Complex.measurableEquivRealProd.measurable hs), withDensity_apply _ hs]
  have he := Complex.volume_preserving_equiv_real_prod.setLIntegral_comp_preimage
    hs (f := fun z => ENNReal.ofReal (q (Complex.measurableEquivRealProd.symm z)))
    (ENNReal.continuous_ofReal.measurable.comp
      (hq.comp Complex.measurableEquivRealProd.symm.measurable))
  simpa only [Complex.measurableEquivRealProd.symm_apply_apply] using he

example : Measurable (fun z : ℂ => Real.exp (-‖z‖ ^ 2)) := by fun_prop

/-- The actual standardized normalized sum has an integrable complex
characteristic function when `n > 4(β + 1)`. This retains the proved
bandwidth-dependent threshold, replacing the source's fixed five summands.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem integrable_charFun_complexLaw_scaledSum_lawY {β : ℝ} (hβ : 0 < β) (t : ℝ)
    {n : ℕ} (hn : 4 * (β + 1) < (n : ℝ)) :
    Integrable (charFun (complexLaw ((Measure.pi fun _ : Fin n => lawY β t).map (scaledSum n)))) := by
  exact integrable_charFun_complexLaw _ (integrable_characteristic_scaledSum_lawY hβ t hn)

example : (0 : ℝ) < 3 ∧ 4 * (3 + 1) < ((17 : ℕ) : ℝ) := by norm_num

/-- The real part of the actual inverse characteristic-function integral.
Reality, nonnegativity, and the measure identity are proved separately;
none of these properties is built into this definition. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
noncomputable def inverseCharacteristic (μ : Measure ℂ) (w : ℂ) : ℝ :=
  (𝓕 (normalizedCharacteristic μ) w).re

/-- Domination by the integrable characteristic function makes the
inverse integral continuous at every point. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem continuous_inverseCharacteristic (μ : Measure ℂ) [IsProbabilityMeasure μ]
    (hI : Integrable (charFun μ)) : Continuous (inverseCharacteristic μ) := by
  have he : 𝓕 (normalizedCharacteristic μ) = fun w => ∫ ξ,
      Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I) *
        normalizedCharacteristic μ ξ := by
    funext w
    exact Real.fourier_eq' _ w
  have hc : Continuous (𝓕 (normalizedCharacteristic μ)) := by
    rw [he]
    apply continuous_of_dominated (bound := fun ξ => ‖normalizedCharacteristic μ ξ‖)
    · intro w
      have h : Continuous fun ξ : ℂ =>
          Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I) *
            normalizedCharacteristic μ ξ := by
        unfold normalizedCharacteristic
        fun_prop
      exact h.aestronglyMeasurable
    · exact fun w => ae_of_all _ fun ξ => by
        rw [norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]
    · exact (integrable_normalizedCharacteristic μ hI).norm
    · exact ae_of_all _ fun ξ => by fun_prop
  exact Complex.continuous_re.comp hc

example := continuous_inverseCharacteristic _
  (integrable_charFun_complexLaw_scaledSum_lawY (n := 17)
    (by norm_num : (0 : ℝ) < 3) 0 (by norm_num))

/-- The inverse integral is bounded by the integral of the normalized
characteristic norm. The statement holds with Bochner's integral convention;
the density uses below have proved integrability. Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4–§5.5, Fourier inversion. -/
theorem inverseCharacteristic_abs_le (μ : Measure ℂ) (w : ℂ) :
    |inverseCharacteristic μ w| ≤ ∫ ξ, ‖normalizedCharacteristic μ ξ‖ := by
  apply (Complex.abs_re_le_norm _).trans
  rw [Real.fourier_eq']
  apply (norm_integral_le_integral_norm _).trans
  have he (ξ : ℂ) :
      ‖Complex.exp (((-2 * Real.pi * inner ℝ ξ w : ℝ) : ℂ) * Complex.I) •
        normalizedCharacteristic μ ξ‖ = ‖normalizedCharacteristic μ ξ‖ := by
    rw [smul_eq_mul, norm_mul, Complex.norm_exp_ofReal_mul_I, one_mul]
  simp only [he]
  exact le_rfl

end Transformer.Modes
