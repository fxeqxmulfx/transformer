import Transformer.Modes.Section3_CubicDerivatives
import Mathlib.Analysis.Calculus.ContDiff.Comp

/-!
# The second and third derivatives of the literal cubic correction

The differentiated correction in arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
comes from the actual polynomial `B(ξ) = -i E⟨ξ,X⟩³ / 6` of §3.1
`eq:psi`. Its second derivative is the third characteristic derivative
at zero with one frequency slot remaining. Its third derivative is
exactly that constant trilinear map.

We differentiate the already identified first derivative and use the
literal phase-integral formula to permute its slots. Thus the identities
need only a finite third moment; they do not assume that the summand's
characteristic function is smooth at all orders. The norm bounds retain
the actual third norm moment and the factor eight from the product sup
norm. These are the correction terms needed for the cubic spatial weight.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual second correction derivative is the third characteristic derivative with one slot ξ.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the second derivative of the literal cubic coefficient, with exact factor one. -/
theorem iteratedFDeriv_two_cubicCharacteristicCorrection_apply
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) :
    iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ v =
      iteratedFDeriv ℝ 3 (characteristic2 μ) 0 ![v 0, v 1, ξ] := by
  let T := iteratedFDeriv ℝ 3 (characteristic2 μ) 0
  let D : (ℝ × ℝ) → Fin 3 → ℝ × ℝ := fun η => ![v 1, η, η]
  let L : (ℝ × ℝ) →L[ℝ] (Fin 3 → ℝ × ℝ) :=
    ContinuousLinearMap.pi ![0, ContinuousLinearMap.id ℝ (ℝ × ℝ),
      ContinuousLinearMap.id ℝ (ℝ × ℝ)]
  have hD : HasFDerivAt D L ξ := by
    apply hasFDerivAt_pi.mpr
    intro i
    fin_cases i
    · exact hasFDerivAt_const (v 1) ξ
    · exact hasFDerivAt_id ξ
    · exact hasFDerivAt_id ξ
  have hd := ((T.hasFDerivAt (D ξ)).comp ξ hD).const_smul (1 / 2 : ℝ)
  have hd' : HasFDerivAt (fun η => (1 / 2 : ℝ) • T (D η))
      ((1 / 2 : ℝ) • (T.linearDeriv (D ξ)).comp L) ξ := by
    simpa only [Function.comp_def, Pi.smul_def] using hd
  have he : (fun η => fderiv ℝ (cubicCharacteristicCorrection μ) η (v 1)) =
      fun η => (1 / 2 : ℝ) • T (D η) := by
    funext η
    exact fderiv_cubicCharacteristicCorrection_eq_third_derivative μ hmom η (v 1)
  have hB := (contDiff_cubicCharacteristicCorrection_of_memLp μ hmom).fderiv_right
    (m := 1) (by simp)
  have hev := iteratedFDeriv_clm_apply_const_apply hB (i := 1) (by simp)
    (x := ξ) (u := v 1) (m := fun _ => v 0)
  simp only [iteratedFDeriv_one_apply] at hev
  rw [iteratedFDeriv_two_apply, ← hev, he, hd'.fderiv]
  simp only [smul_apply, ContinuousLinearMap.comp_apply,
    ContinuousMultilinearMap.linearDeriv_apply, Fin.sum_univ_three]
  have h0 : T (Function.update (D ξ) 0 (L (v 0) 0)) = 0 := by
    dsimp only [T]
    rw [iteratedFDeriv_characteristic2 μ hmom]
    simp [D, L, Fin.prod_univ_three, Function.update]
  have hT (i : Fin 3) (hi : i ≠ 0) :
      T (Function.update (D ξ) i (L (v 0) i)) = T ![v 0, v 1, ξ] := by
    dsimp only [T]
    rw [iteratedFDeriv_characteristic2 μ hmom, iteratedFDeriv_characteristic2 μ hmom]
    congr 1
    apply integral_congr_ae
    exact ae_of_all _ fun z => by
      fin_cases i
      · exact (hi rfl).elim
      · simp [D, L, Fin.prod_univ_three, Function.update, mul_comm, mul_left_comm]
      · simp [D, L, Fin.prod_univ_three, Function.update, mul_comm, mul_left_comm]
  rw [h0, hT 1 (by decide), hT 2 (by decide)]
  simp only [zero_add, Complex.real_smul]
  norm_num
  ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The third derivative of the literal correction is exactly the third characteristic derivative at zero.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the constant third derivative of `-i E⟨ξ,X⟩³ / 6`. -/
theorem iteratedFDeriv_three_cubicCharacteristicCorrection_apply
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ : ℝ × ℝ) (v : Fin 3 → ℝ × ℝ) :
    iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ v =
      iteratedFDeriv ℝ 3 (characteristic2 μ) 0 v := by
  let T := iteratedFDeriv ℝ 3 (characteristic2 μ) 0
  let L := T.toContinuousLinearMap ![v 1, v 2, 0] 2
  have he : (fun η => iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) η
      (Fin.tail v)) = L := by
    funext η
    rw [iteratedFDeriv_two_cubicCharacteristicCorrection_apply μ hmom]
    simp only [L, ContinuousMultilinearMap.toContinuousLinearMap_apply]
    congr 1
    funext i
    fin_cases i <;> simp [Fin.tail, Function.update]
  have hB : Differentiable ℝ (iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ)) :=
    ContDiff.differentiable_iteratedFDeriv (by decide)
      (contDiff_cubicCharacteristicCorrection_of_memLp μ hmom)
  rw [iteratedFDeriv_succ_apply_left,
    ← fderiv_continuousMultilinear_apply_const_apply (hB ξ) (Fin.tail v) (v 0),
    he, L.hasFDerivAt.fderiv, ContinuousMultilinearMap.toContinuousLinearMap_apply]
  dsimp only [T]
  rw [iteratedFDeriv_characteristic2 μ hmom, iteratedFDeriv_characteristic2 μ hmom]
  congr 1
  apply integral_congr_ae
  exact ae_of_all _ fun z => by
    simp [Fin.prod_univ_three, Function.update, mul_comm, mul_assoc]

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual second correction derivative has a linear frequency bound in operator norm.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the second
cubic derivative. The factor eight uses the project's product sup norm. -/
theorem norm_iteratedFDeriv_two_cubicCharacteristicCorrection_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ‖ ≤
      8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  rw [iteratedFDeriv_two_cubicCharacteristicCorrection_apply μ hmom]
  have h := norm_iteratedFDeriv_characteristic2_apply_le μ hmom 0 ![v 0, v 1, ξ]
  simp [Fin.prod_univ_three] at h
  calc
    _ ≤ 2 ^ 3 * (‖v 0‖ * ‖v 1‖ * ‖ξ‖) * ∫ z, ‖z‖ ^ 3 ∂μ := h
    _ = _ := by simp only [Fin.prod_univ_two]; ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The full third correction derivative is bounded by eight times the actual third norm moment.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the constant
third cubic derivative in the Fourier estimate for the cubic spatial weight. -/
theorem norm_iteratedFDeriv_three_cubicCharacteristicCorrection_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ‖ ≤
      8 * (∫ z, ‖z‖ ^ 3 ∂μ) := by
  have he : iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ =
      iteratedFDeriv ℝ 3 (characteristic2 μ) 0 := by
    ext v
    exact iteratedFDeriv_three_cubicCharacteristicCorrection_apply μ hmom ξ v
  rw [he]
  convert norm_iteratedFDeriv_characteristic2_le μ hmom 0 using 1
  norm_num

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
