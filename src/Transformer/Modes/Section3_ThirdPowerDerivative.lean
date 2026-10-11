import Transformer.Modes.Section3_PowerDerivatives
import Mathlib.Analysis.Calculus.ContDiff.Comp

/-!
# Third derivatives of characteristic powers and frequency scaling

The product rule in arXiv:2412.09080v3, §5.4 `eq:br-9.10` and
`eq:big-z-exp`, has three kinds of terms at derivative order three:
three first derivatives, one second and one first derivative in each
of three arrangements, and one third derivative. Their falling-factorial
coefficients and remaining undifferentiated powers are retained below.

Real frequency scaling contributes one factor of the scale for every
derivative. The formula at every order follows from the actual
continuous multilinear derivative, rather than a redefined scaled
derivative. Finite moment of that order also gives the corresponding
continuous derivatives of the actual normalized-sum characteristic
function. These facts are independent of Fourier integrability.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The third derivative of a complex power contains all five product-rule terms.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
derivative order three. The falling-factorial coefficients keep the
formula valid even for `n = 0, 1, 2`, with natural-number subtraction. -/
theorem iteratedFDeriv_three_complex_power_apply (f : ℝ × ℝ → ℂ) (n : ℕ)
    (hf : ContDiff ℝ 3 f) (ξ : ℝ × ℝ) (v : Fin 3 → ℝ × ℝ) :
    iteratedFDeriv ℝ 3 (fun η => f η ^ n) ξ v =
      (n : ℂ) * (n - 1 : ℕ) * (n - 2 : ℕ) * f ξ ^ (n - 3) *
        fderiv ℝ f ξ (v 0) * fderiv ℝ f ξ (v 1) * fderiv ℝ f ξ (v 2) +
      (n : ℂ) * (n - 1 : ℕ) * f ξ ^ (n - 2) *
        (iteratedFDeriv ℝ 2 f ξ ![v 0, v 1] * fderiv ℝ f ξ (v 2) +
          iteratedFDeriv ℝ 2 f ξ ![v 0, v 2] * fderiv ℝ f ξ (v 1) +
          iteratedFDeriv ℝ 2 f ξ ![v 1, v 2] * fderiv ℝ f ξ (v 0)) +
      (n : ℂ) * f ξ ^ (n - 1) * iteratedFDeriv ℝ 3 f ξ v := by
  have hdf := hf.differentiable (by norm_num)
  have hD := (hf.fderiv_right (m := 2) (by norm_num)).differentiable (by norm_num)
  have hD2 : Differentiable ℝ (iteratedFDeriv ℝ 2 f) :=
    ContDiff.differentiable_iteratedFDeriv (by norm_num) hf
  let A : ℝ × ℝ → ℂ := fun η => fderiv ℝ f η (v 1)
  let B : ℝ × ℝ → ℂ := fun η => fderiv ℝ f η (v 2)
  let Q : ℝ × ℝ → ℂ := fun η => iteratedFDeriv ℝ 2 f η (Fin.tail v)
  have hA : Differentiable ℝ A := hD.clm_apply (differentiable_const (v 1))
  have hB : Differentiable ℝ B := hD.clm_apply (differentiable_const (v 2))
  have hQ : Differentiable ℝ Q := hD2.continuousMultilinear_apply_const (Fin.tail v)
  have he : (fun η => iteratedFDeriv ℝ 2 (fun ζ => f ζ ^ n) η (Fin.tail v)) =
      fun η => ((n : ℂ) * (n - 1 : ℕ)) * (f η ^ (n - 2) * A η * B η) +
        (n : ℂ) * (f η ^ (n - 1) * Q η) := by
    funext η
    rw [iteratedFDeriv_two_complex_power_apply f n (hf.of_le (by norm_num))]
    dsimp [A, B, Q, Fin.tail]
    ring
  have hd := (((hdf ξ).hasFDerivAt.pow (n - 2)).mul (hA ξ).hasFDerivAt
    |>.mul (hB ξ).hasFDerivAt |>.const_mul ((n : ℂ) * (n - 1 : ℕ))).add
    ((((hdf ξ).hasFDerivAt.pow (n - 1)).mul (hQ ξ).hasFDerivAt).const_mul (n : ℂ))
  change HasFDerivAt (fun η => ((n : ℂ) * (n - 1 : ℕ)) *
    (f η ^ (n - 2) * A η * B η) + (n : ℂ) * (f η ^ (n - 1) * Q η)) _ ξ at hd
  have hPowD2 : Differentiable ℝ (iteratedFDeriv ℝ 2 (fun η => f η ^ n)) :=
    ContDiff.differentiable_iteratedFDeriv (by norm_num) (hf.pow n)
  have hAder : fderiv ℝ A ξ (v 0) = iteratedFDeriv ℝ 2 f ξ ![v 0, v 1] := by
    have h := iteratedFDeriv_clm_apply_const_apply
      (hf.fderiv_right (m := 1) (by norm_num)) (i := 1) (by norm_num)
      (x := ξ) (u := v 1) (m := fun _ => v 0)
    simpa [A, iteratedFDeriv_one_apply, iteratedFDeriv_two_apply] using h
  have hBder : fderiv ℝ B ξ (v 0) = iteratedFDeriv ℝ 2 f ξ ![v 0, v 2] := by
    have h := iteratedFDeriv_clm_apply_const_apply
      (hf.fderiv_right (m := 1) (by norm_num)) (i := 1) (by norm_num)
      (x := ξ) (u := v 2) (m := fun _ => v 0)
    simpa [B, iteratedFDeriv_one_apply, iteratedFDeriv_two_apply] using h
  have hQder : fderiv ℝ Q ξ (v 0) = iteratedFDeriv ℝ 3 f ξ v := by
    rw [iteratedFDeriv_succ_apply_left]
    exact fderiv_continuousMultilinear_apply_const_apply (hD2 ξ) (Fin.tail v) (v 0)
  have htail : Fin.tail v = ![v 1, v 2] := by
    funext i
    fin_cases i <;> rfl
  rw [iteratedFDeriv_succ_apply_left,
    ← fderiv_continuousMultilinear_apply_const_apply (hPowD2 ξ) (Fin.tail v) (v 0), he,
    hd.fderiv]
  simp only [add_apply, smul_apply, nsmul_eq_mul, smul_eq_mul, Pi.mul_apply,
    hAder, hBder, hQder,
    show n - 2 - 1 = n - 3 by omega, show n - 1 - 1 = n - 2 by omega]
  dsimp only [A, B, Q]
  rw [htail]
  ring

example : ContDiff ℝ 3 (characteristic2 stdGauss2) :=
  contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))

/-- Real frequency scaling supplies exactly one scale factor in every derivative slot.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
the literal normalized-sum frequency substitution. The identity is
for the actual iterated Fréchet derivative at every finite order. -/
theorem iteratedFDeriv_complex_comp_smul (f : ℝ × ℝ → ℂ) (c : ℝ) (k : ℕ)
    (hf : ContDiff ℝ k f) (ξ : ℝ × ℝ) (v : Fin k → ℝ × ℝ) :
    iteratedFDeriv ℝ k (fun η => f (c • η)) ξ v =
      (c : ℂ) ^ k * iteratedFDeriv ℝ k f (c • ξ) v := by
  let g : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ) := c • ContinuousLinearMap.id ℝ (ℝ × ℝ)
  change iteratedFDeriv ℝ k (f ∘ g) ξ v = _
  rw [g.iteratedFDeriv_comp_right hf ξ le_rfl,
    ContinuousMultilinearMap.compContinuousLinearMap_apply]
  change iteratedFDeriv ℝ k f (c • ξ) (fun i => c • v i) = _
  rw [ContinuousMultilinearMap.map_smul_univ]
  simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin,
    Complex.real_smul, Complex.ofReal_pow]

example (k : ℕ) : ContDiff ℝ k (characteristic2 stdGauss2) :=
  contDiff_characteristic2 stdGauss2 (k := k) (IsGaussian.memLp_id _ _ (by simp))

/-- Finite moment of order `k` gives `k` continuous derivatives of the actual normalized sum.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
the regularity used before Fourier differentiation. This conclusion
does not assume an integrable characteristic power or a density. -/
theorem contDiff_characteristic_scaledSum_of_memLp (μ : Measure (ℝ × ℝ))
    [IsProbabilityMeasure μ] (k : ℕ) (hmom : MemLp id (k : ℝ≥0∞) μ) (n : ℕ) :
    ContDiff ℝ k (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) := by
  let g : (ℝ × ℝ) →L[ℝ] (ℝ × ℝ) :=
    (Real.sqrt n)⁻¹ • ContinuousLinearMap.id ℝ (ℝ × ℝ)
  have he : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun ξ => characteristic2 μ ((Real.sqrt n)⁻¹ • ξ) ^ n := by
    funext ξ
    exact characteristic_scaledSum μ n ξ
  rw [he]
  exact ((contDiff_characteristic2 μ hmom).comp g.contDiff).pow n

example (k : ℕ) : IsProbabilityMeasure stdGauss2 ∧ MemLp id (k : ℝ≥0∞) stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual normalized sum's third derivative is the literal power derivative times `n^(-3/2)`.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-exp`,
with the actual `scaledSum` frequency substitution replacing the
source's inconsistent block normalization. All five terms of the
preceding power formula receive this same cubed scale factor. -/
theorem iteratedFDeriv_three_characteristic_scaledSum_apply
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (n : ℕ) (ξ : ℝ × ℝ) (v : Fin 3 → ℝ × ℝ) :
    iteratedFDeriv ℝ 3
      (characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n))) ξ v =
      (((Real.sqrt n)⁻¹ : ℝ) : ℂ) ^ 3 *
        iteratedFDeriv ℝ 3 (fun η => characteristic2 μ η ^ n) ((Real.sqrt n)⁻¹ • ξ) v := by
  have he : characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) =
      fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η) ^ n := by
    funext η
    exact characteristic_scaledSum μ n η
  rw [he]
  exact iteratedFDeriv_complex_comp_smul (fun η => characteristic2 μ η ^ n)
    (Real.sqrt n)⁻¹ 3 ((contDiff_characteristic2 μ (k := 3) hmom).pow n) ξ v

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
