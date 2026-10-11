import Transformer.Modes.Section3_CubicHigherGaussianComparison
import Transformer.Modes.Section3_CubicDerivativeTaylor
import Transformer.Modes.Section3_CubicProductDerivativeBounds

/-!
# Common local derivative bounds for the actual one-summand errors

The Taylor inputs of arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
are assembled here for every derivative order through three. The
Gaussian comparison without the cubic coefficient has degree
`3-k`, while retaining that coefficient leaves degree `4-k`.
All bounds refer to actual iterated derivatives of the literal
characteristic functions and moment polynomial.

The common constants depend on the fixed law's third or fourth
norm moment and the corresponding Gaussian norm moment. The local
unit-ball hypotheses belong only to the zero and first Taylor
comparisons. They are retained in the combined statements. The
cubic coefficient's own homogeneous derivative bounds hold globally.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual Gaussian comparison has its correct decreasing Taylor degree through order three.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
third-order factors in the exact cubic power error. -/
theorem norm_iteratedFDeriv_characteristic2_sub_gaussian_local_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ η - characteristic2 stdGauss2 η) ξ‖ ≤
      (1 + 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2)) * ‖ξ‖ ^ (3 - k : ℕ) := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hG : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hf := contDiff_characteristic2 μ (k := 3) hmom
  have hg := contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))
  have hc0 : 4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1 ≤
      1 + 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2) := by linarith
  have hc : 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2) ≤
      1 + 8 * ((∫ z, ‖z‖ ^ 3 ∂μ) + ∫ z, ‖z‖ ^ 3 ∂stdGauss2) := by linarith
  interval_cases k
  · rw [norm_iteratedFDeriv_zero, characteristic2_stdGauss2]
    exact (norm_characteristic2_sub_gaussian μ hμ hmom ξ hξ).trans
      (mul_le_mul_of_nonneg_right hc0 (by positivity))
  · rw [norm_iteratedFDeriv_one, fderiv_fun_sub
      (hf.differentiable (by norm_num) ξ) (hg.differentiable (by norm_num) ξ)]
    exact (norm_fderiv_characteristic2_sub_gaussian_le μ hμ hmom ξ).trans
      (mul_le_mul_of_nonneg_right hc (by positivity))
  · rw [fun_iteratedFDeriv_sub (hf.of_le (by norm_num)) (hg.of_le (by norm_num))]
    simpa only [Nat.reduceSub, pow_one] using (norm_iteratedFDeriv_two_characteristic2_sub_gaussian_le
      μ hμ hmom ξ).trans (mul_le_mul_of_nonneg_right hc (norm_nonneg ξ))
  · rw [fun_iteratedFDeriv_sub hf hg, pow_zero, mul_one]
    calc
      _ ≤ ‖iteratedFDeriv ℝ 3 (characteristic2 μ) ξ‖ +
          ‖iteratedFDeriv ℝ 3 (characteristic2 stdGauss2) ξ‖ := norm_sub_le _ _
      _ ≤ 2 ^ 3 * (∫ z, ‖z‖ ^ 3 ∂μ) + 2 ^ 3 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2) :=
        add_le_add (norm_iteratedFDeriv_characteristic2_le μ hmom ξ)
          (norm_iteratedFDeriv_characteristic2_le stdGauss2 (IsGaussian.memLp_id _ _ (by simp)) ξ)
      _ ≤ _ := by norm_num at hc ⊢; linarith

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    ‖((0, 0) : ℝ × ℝ)‖ ≤ 1 ∧ (3 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), by norm_num, le_rfl⟩

/-- Subtracting the actual cubic coefficient raises every local Taylor error degree by one.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
fourth-order factor in the exact cubic-corrected power error. -/
theorem norm_iteratedFDeriv_characteristic2_sub_gaussian_sub_cubic_local_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ η - characteristic2 stdGauss2 η -
        cubicCharacteristicCorrection μ η) ξ‖ ≤
      (5 + 16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2)) * ‖ξ‖ ^ (4 - k : ℕ) := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 4 ∂μ := integral_nonneg fun z => by positivity
  have hG : 0 ≤ ∫ z, ‖z‖ ^ 4 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hmom3 : MemLp id 3 μ := hmom.mono_exponent (by norm_num)
  have hf := contDiff_characteristic2 μ (k := 3) hmom3
  have hg := contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))
  have hb : ContDiff ℝ 3 (cubicCharacteristicCorrection μ) :=
    (contDiff_cubicCharacteristicCorrection_of_memLp μ hmom3).of_le (by simp)
  have hd : ContDiff ℝ 3 (fun η => characteristic2 μ η - characteristic2 stdGauss2 η) :=
    hf.sub hg
  have hc0 : (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) + 1 ≤
      5 + 16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2) := by linarith
  have hc1 : 8 * (∫ z, ‖z‖ ^ 4 ∂μ) + 4 ≤
      5 + 16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2) := by linarith
  have hc : 16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2) ≤
      5 + 16 * ((∫ z, ‖z‖ ^ 4 ∂μ) + ∫ z, ‖z‖ ^ 4 ∂stdGauss2) := by linarith
  interval_cases k
  · rw [norm_iteratedFDeriv_zero]
    exact (norm_characteristic2_sub_gaussian_sub_cubicCorrection μ hμ hmom ξ hξ).trans
      (mul_le_mul_of_nonneg_right hc0 (by positivity))
  · rw [norm_iteratedFDeriv_one, fderiv_fun_sub
      (hd.differentiable (by norm_num) ξ)
      (hb.differentiable (by norm_num) ξ), fderiv_fun_sub
        (hf.differentiable (by norm_num) ξ) (hg.differentiable (by norm_num) ξ)]
    exact (norm_fderiv_characteristic2_sub_gaussian_sub_cubic μ hμ hmom ξ hξ).trans
      (mul_le_mul_of_nonneg_right hc1 (by positivity))
  · rw [fun_iteratedFDeriv_sub ((hf.sub hg).of_le (by norm_num)) (hb.of_le (by norm_num)),
      fun_iteratedFDeriv_sub (hf.of_le (by norm_num)) (hg.of_le (by norm_num))]
    exact (norm_iteratedFDeriv_two_characteristic2_sub_gaussian_sub_cubic_le μ hμ hmom ξ).trans
      (mul_le_mul_of_nonneg_right hc (by positivity))
  · rw [fun_iteratedFDeriv_sub (hf.sub hg) hb, fun_iteratedFDeriv_sub hf hg]
    simpa only [Nat.reduceSub, pow_one] using
      (norm_iteratedFDeriv_three_characteristic2_sub_gaussian_sub_cubic_le μ hmom ξ).trans
        (mul_le_mul_of_nonneg_right hc (norm_nonneg ξ))

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 ∧
    ‖((0, 0) : ℝ × ℝ)‖ ≤ 1 ∧ (3 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp), by norm_num, le_rfl⟩

/-- The actual cubic coefficient has homogeneous derivative bounds at every frequency.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
literal third-order moment factor before Gaussian multiplication. -/
theorem norm_iteratedFDeriv_cubicCharacteristicCorrection_homogeneous_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (ξ : ℝ × ℝ) (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (cubicCharacteristicCorrection μ) ξ‖ ≤
      8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ (3 - k : ℕ) := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  interval_cases k
  · rw [norm_iteratedFDeriv_zero]
    exact (norm_cubicCharacteristicCorrection_le μ hmom ξ).trans
      (mul_le_mul_of_nonneg_right (by linarith :
        (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) (by positivity))
  · rw [norm_iteratedFDeriv_one]
    exact (norm_fderiv_cubicCharacteristicCorrection_le μ hmom ξ).trans
      (mul_le_mul_of_nonneg_right (by linarith :
        4 * (∫ z, ‖z‖ ^ 3 ∂μ) ≤ 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) (by positivity))
  · simpa only [Nat.reduceSub, pow_one] using norm_iteratedFDeriv_two_cubicCharacteristicCorrection_le μ hmom ξ
  · simpa only [Nat.sub_self, pow_zero, mul_one] using norm_iteratedFDeriv_three_cubicCharacteristicCorrection_le μ hmom ξ

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ (3 : ℕ) ≤ 3 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), le_rfl⟩

/-- Literal frequency scaling converts the decreasing Taylor degrees into one polynomial profile.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
degree-preserving scale of each actual derivative before taking powers. -/
theorem norm_iteratedFDeriv_comp_smul_le_of_homogeneous_bound
    (f : ℝ × ℝ → ℂ) (k p : ℕ) (hf : ContDiff ℝ k f) (hkp : k ≤ p)
    (c : ℝ) (hc : 0 ≤ c) (ξ : ℝ × ℝ) {C : ℝ} (hC : 0 ≤ C)
    (hbound : ‖iteratedFDeriv ℝ k f (c • ξ)‖ ≤ C * ‖c • ξ‖ ^ (p - k)) :
    ‖iteratedFDeriv ℝ k (fun η => f (c • η)) ξ‖ ≤ C * c ^ p * (1 + ‖ξ‖) ^ p := by
  have hnorm : ‖c • ξ‖ = c * ‖ξ‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hc]
  have hp : ‖ξ‖ ^ (p - k) ≤ (1 + ‖ξ‖) ^ p :=
    (pow_le_pow_left₀ (norm_nonneg ξ) (by linarith [norm_nonneg ξ]) _).trans
      (pow_le_pow_right₀ (by linarith [norm_nonneg ξ]) (by omega))
  rw [norm_iteratedFDeriv_complex_comp_smul f c k hf, abs_of_nonneg hc]
  calc
    _ ≤ c ^ k * (C * ‖c • ξ‖ ^ (p - k)) :=
      mul_le_mul_of_nonneg_left hbound (pow_nonneg hc k)
    _ = C * c ^ p * ‖ξ‖ ^ (p - k) := by
      rw [hnorm, mul_pow]
      calc
        _ = C * (c ^ k * c ^ (p - k)) * ‖ξ‖ ^ (p - k) := by ring
        _ = _ := by rw [← pow_add, Nat.add_sub_of_le hkp]
    _ ≤ _ := mul_le_mul_of_nonneg_left hp (by positivity)

example : ContDiff ℝ 0 (fun _ : ℝ × ℝ => (1 : ℂ)) ∧ (0 : ℕ) ≤ 0 ∧
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    ‖iteratedFDeriv ℝ 0 (fun _ : ℝ × ℝ => (1 : ℂ)) ((1 : ℝ) • ((0, 0) : ℝ × ℝ))‖ ≤
      1 * ‖(1 : ℝ) • ((0, 0) : ℝ × ℝ)‖ ^ (0 - 0 : ℕ) := by
  refine ⟨contDiff_const, le_rfl, zero_le_one, zero_le_one, ?_⟩
  simp only [norm_iteratedFDeriv_zero, norm_one, Nat.sub_self, pow_zero, one_mul, le_refl]

end Transformer.Modes
