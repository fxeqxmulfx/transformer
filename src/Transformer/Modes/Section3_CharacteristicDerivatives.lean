import Transformer.Modes.Section3_ComplexCharacteristic
import Transformer.Modes.Section3_CharacteristicTaylor
import Mathlib.MeasureTheory.Function.LpSpace.Basic
import Mathlib.Analysis.Calculus.ContDiff.Basic
/-!
# Differentiating the literal characteristic function

The derivative step in arXiv:2412.09080v3, §5.4, expresses Fourier
derivatives as characteristic phase integrals with polynomial weights.
A finite moment of order `k` justifies `k` derivatives of the actual
characteristic function. The formula and operator norm bound are proved
here with the project's product norm on `ℝ²`.

Transport to the Euclidean complex model supplies the library's
characteristic derivative formula. The real continuous linear coordinate
equivalence preserves finite moments even though it is not an isometry
for the product norm. Transporting the derivative and integral back
recovers the paper's explicit dot products and positive phase.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual law's finite moments transfer to complex coordinates.
Source: arXiv:2412.09080v3, §3 `thm:br`, the finite moment condition,
and §5.4, differentiating the characteristic function. -/
theorem memLp_id_complexLaw (μ : Measure (ℝ × ℝ)) {p : ℝ≥0∞} (hmom : MemLp id p μ) :
    MemLp id p (complexLaw μ) := by
  unfold complexLaw
  rw [Complex.measurableEquivRealProd.symm.memLp_map_measure_iff]
  exact hmom.continuousLinearMap_comp Complex.equivRealProdCLM.symm.toContinuousLinearMap

example : MemLp id 3 stdGauss2 := IsGaussian.memLp_id _ _ (by simp)

/-- A finite moment of order `k` gives `k` continuous derivatives
of the literal characteristic function on the product norm space.
Source: arXiv:2412.09080v3, §5.4, the Fourier derivative step. -/
theorem contDiff_characteristic2 (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {k : ℕ}
    (hmom : MemLp id (k : ℝ≥0∞) μ) : ContDiff ℝ k (characteristic2 μ) := by
  have hC := contDiff_charFun (memLp_id_complexLaw μ hmom)
  have he : characteristic2 μ = charFun (complexLaw μ) ∘ Complex.equivRealProdCLM.symm := by
    funext ξ
    rw [Function.comp_apply, charFun_complexLaw]
    rfl
  rw [he]
  exact hC.comp Complex.equivRealProdCLM.symm.contDiff

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual `k`th derivative is the phase integral times `I^k`
and the product of the `k` direction dot products. Source:
arXiv:2412.09080v3, §5.4, the derivative formula used in
`eq:br-9.10` and `eq:big-z-exp`. -/
theorem iteratedFDeriv_characteristic2
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {k : ℕ}
    (hmom : MemLp id (k : ℝ≥0∞) μ) (ξ : ℝ × ℝ) (v : Fin k → ℝ × ℝ) :
    iteratedFDeriv ℝ k (characteristic2 μ) ξ v = Complex.I ^ k *
      ∫ z, ((∏ i, (z.1 * (v i).1 + z.2 * (v i).2)) : ℝ) *
        Complex.exp (((z.1 * ξ.1 + z.2 * ξ.2 : ℝ) : ℂ) * Complex.I) ∂μ := by
  let g : (ℝ × ℝ) →L[ℝ] ℂ := Complex.equivRealProdCLM.symm.toContinuousLinearMap
  have hC := memLp_id_complexLaw μ hmom
  have he : characteristic2 μ = charFun (complexLaw μ) ∘ g := by
    funext η
    rw [Function.comp_apply, charFun_complexLaw]
    rfl
  rw [he, g.iteratedFDeriv_comp_right (contDiff_charFun hC) ξ le_rfl,
    ContinuousMultilinearMap.compContinuousLinearMap_apply,
    iteratedFDeriv_charFun hC]
  unfold complexLaw
  rw [integral_map Complex.measurableEquivRealProd.symm.measurable.aemeasurable
    (by fun_prop : Continuous (fun y : ℂ =>
      ((∏ i, inner ℝ y (g (v i))) : ℝ) *
        Complex.exp (((inner ℝ y (g ξ) : ℝ) : ℂ) * Complex.I))).aestronglyMeasurable]
  congr 1
  apply integral_congr_ae
  exact ae_of_all _ fun z => by
    dsimp [g]
    simp only [Complex.mul_re, Complex.conj_re, Complex.conj_im,
      Complex.equivRealProdCLM_symm_apply_re, Complex.equivRealProdCLM_symm_apply_im]
    congr 2
    · apply Finset.prod_congr rfl
      intro i _
      ring
    · congr 1
      push_cast
      ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Finite moments control the characteristic derivative applied to
arbitrary directions, uniformly in frequency. The nonsharp factor
`2^k` accounts for the two coordinates of the product sup norm.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, its moment factors. -/
theorem norm_iteratedFDeriv_characteristic2_apply_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {k : ℕ}
    (hmom : MemLp id (k : ℝ≥0∞) μ) (ξ : ℝ × ℝ) (v : Fin k → ℝ × ℝ) :
    ‖iteratedFDeriv ℝ k (characteristic2 μ) ξ v‖ ≤
      (2 : ℝ) ^ k * (∏ i, ‖v i‖) * ∫ z, ‖z‖ ^ k ∂μ := by
  have hp (z : ℝ × ℝ) :
      ‖(((∏ i : Fin k, (z.1 * (v i).1 + z.2 * (v i).2)) : ℝ) : ℂ) *
        Complex.exp (((z.1 * ξ.1 + z.2 * ξ.2 : ℝ) : ℂ) * Complex.I)‖ ≤
          ((2 : ℝ) ^ k * (∏ i : Fin k, ‖v i‖)) * ‖z‖ ^ k := by
    rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, Complex.norm_exp_ofReal_mul_I,
      mul_one, Finset.abs_prod]
    calc
      _ ≤ ∏ i : Fin k, (2 * ‖z‖) * ‖v i‖ := by
        apply Finset.prod_le_prod₀ (fun i _ => abs_nonneg _)
        intro i _
        exact abs_dot_le_two_norm_mul z (v i)
      _ = _ := by
        rw [Finset.prod_mul_distrib, Finset.prod_const]
        simp only [Finset.card_univ, Fintype.card_fin, mul_pow]
        ring
  rw [iteratedFDeriv_characteristic2 μ hmom ξ v, norm_mul, norm_pow, Complex.norm_I,
    one_pow, one_mul]
  calc
    _ ≤ ∫ z, ‖(((∏ i : Fin k, (z.1 * (v i).1 + z.2 * (v i).2)) : ℝ) : ℂ) *
        Complex.exp (((z.1 * ξ.1 + z.2 * ξ.2 : ℝ) : ℂ) * Complex.I)‖ ∂μ :=
      norm_integral_le_integral_norm _
    _ ≤ ∫ z, ((2 : ℝ) ^ k * (∏ i : Fin k, ‖v i‖)) * ‖z‖ ^ k ∂μ :=
      integral_mono_of_nonneg (ae_of_all _ fun z => norm_nonneg _)
        (hmom.integrable_norm_pow'.const_mul _) (ae_of_all _ hp)
    _ = _ := integral_const_mul _ _

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The full operator norm of the actual characteristic derivative
is bounded by the finite moment, uniformly in frequency. Source:
arXiv:2412.09080v3, §5.4 `eq:big-z-exp`; this is the single-summand
moment bound before the additional power damping for normalized sums. -/
theorem norm_iteratedFDeriv_characteristic2_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {k : ℕ}
    (hmom : MemLp id (k : ℝ≥0∞) μ) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ k (characteristic2 μ) ξ‖ ≤ (2 : ℝ) ^ k * ∫ z, ‖z‖ ^ k ∂μ := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ k ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  calc
    _ ≤ (2 : ℝ) ^ k * (∏ i, ‖v i‖) * ∫ z, ‖z‖ ^ k ∂μ :=
      norm_iteratedFDeriv_characteristic2_apply_le μ hmom ξ v
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
