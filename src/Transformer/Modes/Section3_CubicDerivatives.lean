module

public import Transformer.Modes.Section3_CubicCorrection
public import Transformer.Modes.Section3_DerivativeMoments
public import Mathlib.Analysis.Calculus.FDeriv.Analytic
public import Mathlib.Analysis.Calculus.ContDiff.CPolynomial

/-!
# The actual derivative of the cubic characteristic correction

The positive-sign cubic coefficient in arXiv:2412.09080v3, §3.1
`eq:psi` and §5.4 `eq:br-9.10`, is the literal moment expression
`B(ξ) = -i E⟨ξ,X⟩³ / 6`. The actual third characteristic derivative at
zero is a continuous multilinear map. Its diagonal equals `6 B`, so
finite third moment alone makes this correction a smooth polynomial.

Differentiating that diagonal gives three equal directional terms.
Their equality follows from the actual phase-integral formula, rather
than an assumed symmetry or a new definition of the correction. The
first derivative is exactly `-i E[⟨ξ,X⟩² ⟨v,X⟩] / 2`, including its
sign and factorial. Its norm is bounded by four times the actual third
norm moment times `‖ξ‖² ‖v‖`, in the project's product sup norm.

These identities supply the differentiated correction in the `s = 3`
normalized-sum comparison. Exponential moments are still used separately
to identify the source's logarithmic cumulants with these moments.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The literal cubic correction is the diagonal of the actual third characteristic derivative.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`.
Finite third moment justifies the derivative, with the exact factor `1/6`. -/
theorem cubicCharacteristicCorrection_eq_third_derivative
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    cubicCharacteristicCorrection μ ξ = (6 : ℝ)⁻¹ •
      iteratedFDeriv ℝ 3 (characteristic2 μ) 0 (fun _ : Fin 3 => ξ) := by
  rw [iteratedFDeriv_characteristic2 μ hmom]
  simp only [Fin.prod_univ_three, Prod.fst_zero, Prod.snd_zero, mul_zero, add_zero,
    Complex.ofReal_zero, zero_mul, Complex.exp_zero, mul_one, integral_complex_ofReal]
  have he : (fun z : ℝ × ℝ => (z.1 * ξ.1 + z.2 * ξ.2) *
      (z.1 * ξ.1 + z.2 * ξ.2) * (z.1 * ξ.1 + z.2 * ξ.2)) =
      fun z => (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 := by funext z; ring
  rw [he]
  unfold cubicCharacteristicCorrection
  simp only [Complex.real_smul, Complex.ofReal_inv]
  norm_num [pow_succ, Complex.I_sq]
  ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Finite third moment makes the actual cubic coefficient smooth at every order.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, differentiating the
literal correction. No exponential moment is needed for this polynomial regularity. -/
theorem contDiff_cubicCharacteristicCorrection_of_memLp
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) :
    ContDiff ℝ ⊤ (cubicCharacteristicCorrection μ) := by
  let T := iteratedFDeriv ℝ 3 (characteristic2 μ) 0
  let D : (ℝ × ℝ) →L[ℝ] (Fin 3 → ℝ × ℝ) :=
    ContinuousLinearMap.pi fun _ => ContinuousLinearMap.id ℝ (ℝ × ℝ)
  have he : cubicCharacteristicCorrection μ = fun ξ => (6 : ℝ)⁻¹ • T (D ξ) :=
    funext (cubicCharacteristicCorrection_eq_third_derivative μ hmom)
  rw [he]
  exact (T.contDiff.comp D.contDiff).const_smul _

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The first derivative of the literal correction is half its third-derivative trilinear form.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the first differentiated cubic term, with all three diagonal contributions retained. -/
theorem fderiv_cubicCharacteristicCorrection_eq_third_derivative
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ v : ℝ × ℝ) : fderiv ℝ (cubicCharacteristicCorrection μ) ξ v =
      (1 / 2 : ℝ) • iteratedFDeriv ℝ 3 (characteristic2 μ) 0 ![v, ξ, ξ] := by
  let T := iteratedFDeriv ℝ 3 (characteristic2 μ) 0
  let D : (ℝ × ℝ) →L[ℝ] (Fin 3 → ℝ × ℝ) :=
    ContinuousLinearMap.pi fun _ => ContinuousLinearMap.id ℝ (ℝ × ℝ)
  have he : cubicCharacteristicCorrection μ = fun η => (6 : ℝ)⁻¹ • T (D η) :=
    funext (cubicCharacteristicCorrection_eq_third_derivative μ hmom)
  have hd := ((T.hasFDerivAt (D ξ)).comp ξ D.hasFDerivAt).const_smul ((6 : ℝ)⁻¹)
  have hd' : HasFDerivAt (fun η => (6 : ℝ)⁻¹ • T (D η))
      ((6 : ℝ)⁻¹ • (T.linearDeriv (D ξ)).comp D) ξ := by
    simpa only [Function.comp_def, Pi.smul_def] using hd
  rw [he, hd'.fderiv]
  simp only [smul_apply, ContinuousLinearMap.comp_apply,
    ContinuousMultilinearMap.linearDeriv_apply]
  have hT (i : Fin 3) : T (Function.update (D ξ) i (D v i)) = T ![v, ξ, ξ] := by
    dsimp only [T]
    rw [iteratedFDeriv_characteristic2 μ hmom, iteratedFDeriv_characteristic2 μ hmom]
    congr 1
    apply integral_congr_ae
    exact ae_of_all _ fun z => by
      fin_cases i <;> simp [D, Fin.prod_univ_three, Function.update, mul_comm, mul_left_comm]
  simp_rw [hT]
  rw [Fin.sum_univ_three]
  simp only [Complex.real_smul, Complex.ofReal_inv]
  norm_num
  ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual first correction derivative has the exact weighted moment and factorial.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the first derivative of
the positive-sign coefficient `-i E⟨ξ,X⟩³ / 6`. -/
theorem fderiv_cubicCharacteristicCorrection_apply
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ v : ℝ × ℝ) : fderiv ℝ (cubicCharacteristicCorrection μ) ξ v =
      -Complex.I * ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 2 *
        (v.1 * z.1 + v.2 * z.2) ∂μ : ℝ) : ℂ) / 2 := by
  rw [fderiv_cubicCharacteristicCorrection_eq_third_derivative μ hmom,
    iteratedFDeriv_characteristic2 μ hmom]
  simp only [Prod.fst_zero, Prod.snd_zero, mul_zero, add_zero, Complex.ofReal_zero,
    zero_mul, Complex.exp_zero, mul_one, integral_complex_ofReal]
  have he : (fun z : ℝ × ℝ => ∏ i : Fin 3,
      (z.1 * (![v, ξ, ξ] i).1 + z.2 * (![v, ξ, ξ] i).2)) =
      fun z => (ξ.1 * z.1 + ξ.2 * z.2) ^ 2 * (v.1 * z.1 + v.2 * z.2) := by
    funext z
    simp only [Fin.prod_univ_three]
    dsimp
    ring
  rw [he]
  simp only [Complex.real_smul]
  norm_num [pow_succ, Complex.I_sq]
  ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The first correction derivative has a quadratic frequency bound from the actual third moment.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the differentiated
cubic coefficient. The explicit factor four uses the product sup norm. -/
theorem norm_fderiv_cubicCharacteristicCorrection_apply_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ v : ℝ × ℝ) : ‖fderiv ℝ (cubicCharacteristicCorrection μ) ξ v‖ ≤
      4 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 2 * ‖v‖ := by
  rw [fderiv_cubicCharacteristicCorrection_eq_third_derivative μ hmom, norm_smul]
  have h := norm_iteratedFDeriv_characteristic2_apply_le μ hmom 0 ![v, ξ, ξ]
  simp [Fin.prod_univ_three] at h
  norm_num only [Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 1 / 2)]
  calc
    _ ≤ (1 / 2 : ℝ) * (2 ^ 3 * (‖v‖ * ‖ξ‖ * ‖ξ‖) * ∫ z, ‖z‖ ^ 3 ∂μ) :=
      mul_le_mul_of_nonneg_left h (by norm_num)
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The full operator norm of the actual first correction derivative has the same moment bound.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the cubic derivative
factor used before normalization and Gaussian damping. -/
theorem norm_fderiv_cubicCharacteristicCorrection_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    ‖fderiv ℝ (cubicCharacteristicCorrection μ) ξ‖ ≤
      4 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 2 := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  exact norm_fderiv_cubicCharacteristicCorrection_apply_le μ hmom ξ

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
