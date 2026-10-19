module

public import Transformer.Modes.Section3_CubicHigherTaylor

/-!
# Gaussian comparison after the second and third cubic derivatives

The `s = 3` Fourier expansion in arXiv:2412.09080v3, §5.4
`eq:br-9.10`, compares the summand with the centered Gaussian and its
literal cubic correction. The Gaussian has zero third moments: its
law is invariant under simultaneous negation, and each three-dot
product changes sign. Thus its third characteristic derivative at
zero, its cubic coefficient, and every derivative of that coefficient
vanish.

Subtracting the two fourth-moment Taylor remainders now bounds the
second corrected derivative quadratically, and the third corrected
derivative linearly, in frequency. These estimates hold for the full
multilinear operator norms. They do not yet differentiate the Gaussian
damping of the correction or take the normalized-sum power.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual centered Gaussian law is invariant under simultaneous coordinate negation.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, the centered Gaussian
reference whose third moments vanish in the cubic expansion. -/
theorem stdGauss2_map_neg : stdGauss2.map (fun z => -z) = stdGauss2 := by
  have h := Measure.map_prod_map (gaussianReal 0 1) (gaussianReal 0 1)
    (f := fun x : ℝ => -x) (g := fun x : ℝ => -x) measurable_neg measurable_neg
  rw [gaussianReal_map_neg, neg_zero] at h
  exact h.symm

/-- The actual Gaussian third characteristic derivative at zero vanishes in every three directions.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the vanishing third Gaussian coefficient. The proof integrates the
odd three-dot product against the actual symmetric Gaussian law. -/
theorem iteratedFDeriv_three_characteristic2_stdGauss2_zero
    (v : Fin 3 → ℝ × ℝ) : iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) 0 v = 0 := by
  let w : ℝ × ℝ → ℝ := fun z => ∏ i, (z.1 * (v i).1 + z.2 * (v i).2)
  have ht := integral_map (μ := stdGauss2) (φ := fun z => -z)
    measurable_neg.aemeasurable (f := w) (by dsimp [w]; fun_prop)
  rw [stdGauss2_map_neg] at ht
  have he : (fun z => w (-z)) = fun z => -w z := by
    funext z
    dsimp only [w]
    simp only [Fin.prod_univ_three, Prod.fst_neg, Prod.snd_neg]
    ring
  rw [he, integral_neg] at ht
  have hw : ∫ z, w z ∂stdGauss2 = 0 := by linarith
  rw [iteratedFDeriv_characteristic2 stdGauss2 (IsGaussian.memLp_id _ _ (by simp))]
  simp only [Prod.fst_zero, Prod.snd_zero, mul_zero, add_zero, Complex.ofReal_zero,
    zero_mul, Complex.exp_zero, mul_one, integral_complex_ofReal]
  change Complex.I ^ 3 * ((∫ z, w z ∂stdGauss2 : ℝ) : ℂ) = 0
  rw [hw]
  simp

/-- The literal cubic correction of the actual centered Gaussian law is zero at every frequency.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, the absence of a
third-order Edgeworth correction for a centered Gaussian summand. -/
theorem cubicCharacteristicCorrection_stdGauss2 (ξ : ℝ × ℝ) :
    cubicCharacteristicCorrection stdGauss2 ξ = 0 := by
  rw [cubicCharacteristicCorrection_eq_third_derivative stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)),
    iteratedFDeriv_three_characteristic2_stdGauss2_zero]
  exact smul_zero _

/-- Every derivative of the actual Gaussian cubic coefficient vanishes, including order zero.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, all
differentiated Gaussian cubic terms in the corrected comparison. -/
theorem iteratedFDeriv_cubicCharacteristicCorrection_stdGauss2 (k : ℕ) (ξ : ℝ × ℝ) :
    iteratedFDeriv ℝ k (cubicCharacteristicCorrection stdGauss2) ξ = 0 := by
  have he : cubicCharacteristicCorrection stdGauss2 = fun _ => 0 :=
    funext cubicCharacteristicCorrection_stdGauss2
  rw [he, iteratedFDeriv_fun_zero]
  exact Pi.zero_apply ξ

/-- The full second characteristic derivative error after the literal cubic subtraction is quadratic.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:br-9.10`. Identity covariance cancels the constant terms;
the fourth moments bound both remaining Taylor errors. -/
theorem norm_iteratedFDeriv_two_characteristic2_sub_gaussian_sub_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
        iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ‖ ≤
      16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2) * ‖ξ‖ ^ 2 := by
  have hR := norm_iteratedFDeriv_two_characteristic2_sub_zero_sub_cubic_le μ hμ hmom ξ
  have hG := norm_iteratedFDeriv_two_characteristic2_sub_zero_sub_cubic_le stdGauss2
    isStandardized_stdGauss2 (IsGaussian.memLp_id _ _ (by simp)) ξ
  rw [iteratedFDeriv_cubicCharacteristicCorrection_stdGauss2, sub_zero] at hG
  have hzero : iteratedFDeriv ℝ 2 (characteristic2 μ) 0 =
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) 0 := by
    ext v
    rw [iteratedFDeriv_two_characteristic2_zero μ hμ,
      iteratedFDeriv_two_characteristic2_zero stdGauss2 isStandardized_stdGauss2]
  rw [hzero] at hR
  have he : iteratedFDeriv ℝ 2 (characteristic2 μ) ξ -
      iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
        iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ =
      (iteratedFDeriv ℝ 2 (characteristic2 μ) ξ -
        iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) 0 -
          iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ) -
      (iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
        iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) 0) := by abel
  rw [he]
  calc
    _ ≤ ‖iteratedFDeriv ℝ 2 (characteristic2 μ) ξ -
        iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) 0 -
          iteratedFDeriv ℝ 2 (cubicCharacteristicCorrection μ) ξ‖ +
        ‖iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) ξ -
          iteratedFDeriv ℝ 2 (characteristic2 stdGauss2) 0‖ := norm_sub_le _ _
    _ ≤ 16 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 2 +
        16 * (∫ z, ‖z‖ ^ 4 ∂stdGauss2) * ‖ξ‖ ^ 2 := add_le_add hR hG
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The full third characteristic derivative error after the actual cubic subtraction is linear.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4
`eq:br-9.10`. The Gaussian third coefficient vanishes, and
the actual fourth moments control both changes from zero. -/
theorem norm_iteratedFDeriv_three_characteristic2_sub_gaussian_sub_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 4 μ)
    (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 3 (characteristic2 μ) ξ -
      iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ -
        iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ‖ ≤
      16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2) * ‖ξ‖ := by
  have hR := norm_iteratedFDeriv_three_characteristic2_sub_cubic_le μ hmom ξ
  have hG := norm_iteratedFDeriv_three_characteristic2_sub_cubic_le stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) ξ
  rw [iteratedFDeriv_cubicCharacteristicCorrection_stdGauss2, sub_zero] at hG
  have he : iteratedFDeriv ℝ 3 (characteristic2 μ) ξ -
      iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ -
        iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ =
      (iteratedFDeriv ℝ 3 (characteristic2 μ) ξ -
        iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ) -
      iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ := by abel
  rw [he]
  calc
    _ ≤ ‖iteratedFDeriv ℝ 3 (characteristic2 μ) ξ -
        iteratedFDeriv ℝ 3 (cubicCharacteristicCorrection μ) ξ‖ +
        ‖iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ‖ := norm_sub_le _ _
    _ ≤ 16 * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ +
        16 * (∫ z, ‖z‖ ^ 4 ∂stdGauss2) * ‖ξ‖ := add_le_add hR hG
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
