import Transformer.Modes.Section3_CharacteristicDerivatives
import Mathlib.Analysis.Calculus.FDeriv.Pow
import Mathlib.Analysis.Calculus.FDeriv.Equiv
import Mathlib.Analysis.Calculus.ContDiff.Basic
/-!
# Exact derivatives of normalized-sum characteristic functions

The product rule used in arXiv:2412.09080v3, §5.4 `eq:br-9.10`
and `eq:big-z-exp`, is made explicit for derivative orders one and two.
The source's block normalization is inconsistent; these formulas retain
the literal frequency scale `1 / sqrt n` from `scaledSum`.

For a complex-valued function, the second derivative of its `n`th power
has one term with two first derivatives and another with a single
second derivative. Real frequency scaling contributes its square to
both terms. Finite second moment suffices for the characteristic
function to be twice continuously differentiable. The identities also
hold for `n = 0` and `n = 1`, with natural-number subtraction as in Lean.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Directional first derivative of a complex power over real frequency.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
the first-order product rule applied to the characteristic-function power.
This identity keeps the exponent and coefficient explicit. -/
theorem fderiv_complex_power_apply (f : ℝ × ℝ → ℂ) (n : ℕ) (ξ v : ℝ × ℝ)
    (hf : DifferentiableAt ℝ f ξ) :
    fderiv ℝ (fun η => f η ^ n) ξ v =
      (n : ℂ) * f ξ ^ (n - 1) * fderiv ℝ f ξ v := by
  rw [fderiv_fun_pow n hf]
  simp only [nsmul_eq_mul, smul_apply, smul_eq_mul]

example : DifferentiableAt ℝ (characteristic2 stdGauss2) (1, 0) :=
  (contDiff_characteristic2 stdGauss2 (k := 2)
    (IsGaussian.memLp_id _ _ (by simp))).differentiable (by norm_num) (1, 0)

/-- Second derivative of a complex power, with both product-rule terms.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
the second-order product rule for the characteristic-function power.
All differentiability hypotheses are on the actual function. -/
theorem iteratedFDeriv_two_complex_power_apply (f : ℝ × ℝ → ℂ) (n : ℕ)
    (hf : ContDiff ℝ 2 f) (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 (fun η => f η ^ n) ξ v =
      (n : ℂ) * (n - 1 : ℕ) * f ξ ^ (n - 2) * fderiv ℝ f ξ (v 0) * fderiv ℝ f ξ (v 1) +
      (n : ℂ) * f ξ ^ (n - 1) * iteratedFDeriv ℝ 2 f ξ v := by
  have hdf := hf.differentiable (by norm_num)
  have hD : Differentiable ℝ (fderiv ℝ f) :=
    ((contDiff_succ_iff_fderiv (n := 1)).1 hf).2.2.differentiable (by norm_num)
  have he : fderiv ℝ (fun η => f η ^ n) =
      fun η => ((n : ℂ) * f η ^ (n - 1)) • fderiv ℝ f η := by
    funext η
    rw [fderiv_fun_pow n (hdf η)]
    simp only [nsmul_eq_mul]
  rw [iteratedFDeriv_two_apply, he]
  rw [fderiv_fun_smul (c := fun η => (n : ℂ) * f η ^ (n - 1))
    ((hdf ξ).pow (n - 1) |>.const_mul (n : ℂ)) (hD ξ)]
  rw [fderiv_const_mul (a := fun η => f η ^ (n - 1)) ((hdf ξ).pow (n - 1)), fderiv_fun_pow (n - 1) (hdf ξ)]
  rw [iteratedFDeriv_two_apply]
  have hexp : n - 1 - 1 = n - 2 := by omega
  simp only [add_apply, smul_apply,
    ContinuousLinearMap.smulRight_apply, nsmul_eq_mul, smul_eq_mul, hexp]
  ring

example : ContDiff ℝ 2 (characteristic2 stdGauss2) :=
  contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp))

/-- Real frequency scaling multiplies the second derivative by its square.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-exp`, the derivative scale
of the normalized sum. The formula holds for the actual multilinear
derivative on the product norm space. -/
theorem iteratedFDeriv_two_complex_comp_smul (f : ℝ × ℝ → ℂ) (c : ℝ)
    (hf : ContDiff ℝ 2 f) (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 (fun η => f (c • η)) ξ v =
      (c : ℂ) ^ 2 * iteratedFDeriv ℝ 2 f (c • ξ) v := by
  let g : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ) := c • ContinuousLinearMap.id ℝ (ℝ × ℝ)
  change iteratedFDeriv ℝ 2 (f ∘ g) ξ v = _
  rw [g.iteratedFDeriv_comp_right hf ξ (by norm_num),
    ContinuousMultilinearMap.compContinuousLinearMap_apply]
  change iteratedFDeriv ℝ 2 f (c • ξ) (fun i => c • v i) = _
  rw [ContinuousMultilinearMap.map_smul_univ, Fin.prod_univ_two]
  change ((c * c : ℝ) : ℂ) * iteratedFDeriv ℝ 2 f (c • ξ) v = _
  push_cast
  ring

example : ContDiff ℝ 2 (characteristic2 stdGauss2) :=
  contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp))

/-- Exact first derivative of the actual normalized-sum characteristic function.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`.
The source's block normalization is replaced by the literal `scaledSum`
frequency scale; no exponential moment or integrable power is required. -/
theorem fderiv_characteristic_scaledSum_apply (μ : Measure (ℝ × ℝ))
    [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ) (n : ℕ) (ξ v : ℝ × ℝ) :
    fderiv ℝ (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ v =
      (n : ℂ) * ((Real.sqrt n)⁻¹ : ℝ) *
        characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ (n - 1) *
          fderiv ℝ (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) v := by
  have hf := contDiff_characteristic2 μ (k := 2) hmom
  let g : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ) := (Real.sqrt n)⁻¹ • ContinuousLinearMap.id ℝ (ℝ × ℝ)
  have hg : Differentiable ℝ (fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η)) :=
    (hf.comp g.contDiff).differentiable (by norm_num)
  have he : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η) ^ n := by
    funext η
    exact characteristic_scaledSum μ n η
  rw [he, fderiv_complex_power_apply _ n ξ v (hg ξ), fderiv_comp_smul]
  simp only [smul_apply]
  change (n : ℂ) * characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ (n - 1) *
    ((((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
      fderiv ℝ (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) v) = _
  ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Exact second derivative of the actual normalized-sum characteristic function.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`.
The source's block normalization is replaced by the literal `scaledSum`
frequency scale in both terms; finite second moment is sufficient. -/
theorem iteratedFDeriv_two_characteristic_scaledSum_apply (μ : Measure (ℝ × ℝ))
    [IsProbabilityMeasure μ] (hmom : MemLp id 2 μ) (n : ℕ) (ξ : ℝ × ℝ)
    (v : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ v =
      (n : ℂ) * (n - 1 : ℕ) * (((Real.sqrt n)⁻¹ : ℝ) : ℂ) ^ 2 *
        characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ (n - 2) *
          fderiv ℝ (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) (v 0) *
          fderiv ℝ (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) (v 1) +
      (n : ℂ) * (((Real.sqrt n)⁻¹ : ℝ) : ℂ) ^ 2 *
        characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ (n - 1) *
          iteratedFDeriv ℝ 2 (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) v := by
  have hf := contDiff_characteristic2 μ (k := 2) hmom
  let g : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ) := (Real.sqrt n)⁻¹ • ContinuousLinearMap.id ℝ (ℝ × ℝ)
  have he : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun η => (fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η)) η ^ n := by
    funext η
    exact characteristic_scaledSum μ n η
  have hg : ContDiff ℝ 2 (fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η)) :=
    hf.comp g.contDiff
  rw [he, iteratedFDeriv_two_complex_power_apply
    (fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η)) n hg ξ v]
  rw [iteratedFDeriv_two_complex_comp_smul _ _ hf]
  simp only [fderiv_comp_smul, smul_apply]
  change (n : ℂ) * (n - 1 : ℕ) * characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ (n - 2) *
      ((((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
        fderiv ℝ (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) (v 0)) *
      ((((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
        fderiv ℝ (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) (v 1)) +
    (n : ℂ) * characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ (n - 1) *
      ((((Real.sqrt n)⁻¹ : ℝ) : ℂ) ^ 2 *
        iteratedFDeriv ℝ 2 (characteristic2 μ) ((Real.sqrt n)⁻¹ • ξ) v) = _
  ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 2 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
