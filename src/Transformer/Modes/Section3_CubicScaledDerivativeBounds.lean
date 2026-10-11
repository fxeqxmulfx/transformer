import Transformer.Modes.Section3_CubicLocalDerivativeBounds

/-!
# Small derivatives at the literal normalized frequency scale

In arXiv:2412.09080v3, §5.4 `eq:br-9.10`, centering makes
the first characteristic derivative linear near zero, while identity
covariance bounds the second derivative. A finite third moment bounds
the third derivative. After scaling by `c = 1 / sqrt n`, these three
orders share the factor `c²` times a polynomial in the original frequency.

The bounds here retain the actual compositions and assume explicitly
that their scaled frequency belongs to the unit ball. The Gaussian
difference from one also has a common `c²` bound at all four derivative
orders. It controls the additional term produced when the literal
cubic coefficient is multiplied by Gaussian damping.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Every positive derivative through order three of the actual scaled characteristic function is small.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
centered derivative factors in the mixed characteristic powers. -/
theorem norm_iteratedFDeriv_scaled_characteristic2_positive_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (c : ℝ) (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (ξ : ℝ × ℝ) (hξ : ‖c • ξ‖ ≤ 1) (k : ℕ) (hk1 : 1 ≤ k) (hk3 : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ (c • η)) ξ‖ ≤
      (10 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * c ^ 2 * (1 + ‖ξ‖) := by
  let M : ℝ := ∫ z, ‖z‖ ^ 3 ∂μ
  have hM : 0 ≤ M := integral_nonneg fun z => by positivity
  have hf := contDiff_characteristic2 μ (k := 3) hmom
  have hnorm : ‖c • ξ‖ = c * ‖ξ‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hc]
  have hlin : 2 + 8 * M ≤ 10 + 8 * M := by linarith
  have h8 : 8 ≤ 10 + 8 * M := by linarith
  have hthird : 8 * M ≤ 10 + 8 * M := by linarith
  have hx : 1 ≤ 1 + ‖ξ‖ := by linarith [norm_nonneg ξ]
  interval_cases k
  · rw [norm_iteratedFDeriv_complex_comp_smul _ c 1 (hf.of_le (by norm_num)),
      abs_of_nonneg hc, pow_one, norm_iteratedFDeriv_one]
    calc
      _ ≤ c * ((2 + 8 * M) * ‖c • ξ‖) := mul_le_mul_of_nonneg_left
        (norm_fderiv_characteristic2_small μ hμ hmom (c • ξ) hξ) hc
      _ = (2 + 8 * M) * c ^ 2 * ‖ξ‖ := by rw [hnorm]; ring
      _ ≤ _ := by gcongr; linarith [norm_nonneg ξ]
  · rw [norm_iteratedFDeriv_complex_comp_smul _ c 2 (hf.of_le (by norm_num)), abs_of_nonneg hc]
    calc
      _ ≤ c ^ 2 * 8 := mul_le_mul_of_nonneg_left
        (norm_iteratedFDeriv_two_characteristic2_of_standardized μ hμ (c • ξ)) (by positivity)
      _ ≤ c ^ 2 * ((10 + 8 * M) * (1 + ‖ξ‖)) := by
        gcongr
        exact h8.trans (by simpa using mul_le_mul_of_nonneg_left hx (by positivity : 0 ≤ 10 + 8 * M))
      _ = _ := by ring
  · rw [norm_iteratedFDeriv_complex_comp_smul _ c 3 hf, abs_of_nonneg hc]
    calc
      _ ≤ c ^ 3 * (2 ^ 3 * M) := mul_le_mul_of_nonneg_left
        (norm_iteratedFDeriv_characteristic2_le μ hmom (c • ξ)) (by positivity)
      _ = c ^ 2 * (c * (8 * M)) := by ring
      _ ≤ c ^ 2 * (1 * (8 * M)) := by gcongr
      _ ≤ c ^ 2 * ((10 + 8 * M) * (1 + ‖ξ‖)) := by
        apply mul_le_mul_of_nonneg_left _ (sq_nonneg c)
        simpa only [one_mul] using hthird.trans
          (by simpa using mul_le_mul_of_nonneg_left hx (by positivity : 0 ≤ 10 + 8 * M))
      _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ ‖(1 : ℝ) • ((0, 0) : ℝ × ℝ)‖ ≤ 1 ∧
    (1 : ℕ) ≤ 3 ∧ (3 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    zero_le_one, le_rfl, by norm_num, by omega, le_rfl⟩

/-- All four derivative orders of the actual scaled characteristic function have one polynomial envelope.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
Gaussian-damping factors multiplying the actual cubic coefficient. -/
theorem norm_iteratedFDeriv_scaled_characteristic2_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) (c : ℝ) (hc : 0 ≤ c) (hc1 : c ≤ 1)
    (ξ : ℝ × ℝ) (hξ : ‖c • ξ‖ ≤ 1) (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ (c • η)) ξ‖ ≤
      (11 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * (1 + ‖ξ‖) := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hx : 1 ≤ 1 + ‖ξ‖ := by linarith [norm_nonneg ξ]
  by_cases hk0 : k = 0
  · subst k
    rw [norm_iteratedFDeriv_zero]
    calc
      _ ≤ 1 := norm_characteristic2_le_one μ _
      _ ≤ _ := by nlinarith [mul_nonneg hM (norm_nonneg ξ)]
  have hc2 : c ^ 2 ≤ 1 := pow_le_one₀ hc hc1
  calc
    _ ≤ (10 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * c ^ 2 * (1 + ‖ξ‖) :=
      norm_iteratedFDeriv_scaled_characteristic2_positive_le μ hμ hmom c hc hc1 ξ hξ k (by omega) hk
    _ ≤ (11 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * 1 * (1 + ‖ξ‖) := by gcongr; linarith
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧ ‖(1 : ℝ) • ((0, 0) : ℝ × ℝ)‖ ≤ 1 ∧ (3 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    zero_le_one, le_rfl, by norm_num, le_rfl⟩

/-- The actual difference of the scaled Gaussian characteristic function from one is quadratically small at every derivative order.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
additional error when the cubic moment coefficient is multiplied by `G`. -/
theorem norm_iteratedFDeriv_one_sub_scaled_gaussian_le
    (c : ℝ) (hc : 0 ≤ c) (hc1 : c ≤ 1) (ξ : ℝ × ℝ) (hξ : ‖c • ξ‖ ≤ 1)
    (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => 1 - characteristic2 stdGauss2 (c • η)) ξ‖ ≤
      (10 + 8 * (∫ z, ‖z‖ ^ 3 ∂stdGauss2)) * c ^ 2 * (1 + ‖ξ‖) ^ 2 := by
  have hM : 0 ≤ ∫ z, ‖z‖ ^ 3 ∂stdGauss2 := integral_nonneg fun z => by positivity
  have hnorm : ‖c • ξ‖ = c * ‖ξ‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hc]
  have hscale : ContDiff ℝ 3 (fun η : ℝ × ℝ => c • η) := by fun_prop
  have hf : ContDiff ℝ 3 (fun η => characteristic2 stdGauss2 (c • η)) :=
    (contDiff_characteristic2 stdGauss2 (k := 3) (IsGaussian.memLp_id _ _ (by simp))).comp
      hscale
  by_cases hk0 : k = 0
  · subst k
    rw [norm_iteratedFDeriv_zero, norm_sub_rev]
    calc
      _ ≤ 2 * ‖c • ξ‖ ^ 2 := norm_characteristic2_stdGauss2_sub_one _ hξ
      _ = 2 * c ^ 2 * ‖ξ‖ ^ 2 := by rw [hnorm]; ring
      _ ≤ _ := by gcongr <;> linarith [norm_nonneg ξ]
  rw [fun_iteratedFDeriv_sub (contDiff_const : ContDiff ℝ k (fun _ : ℝ × ℝ => (1 : ℂ)))
    (hf.of_le (by exact_mod_cast hk)), iteratedFDeriv_const_of_ne hk0]
  simp only [Pi.zero_apply, zero_sub, norm_neg]
  have hp : 1 + ‖ξ‖ ≤ (1 + ‖ξ‖) ^ 2 := by nlinarith [sq_nonneg ‖ξ‖, norm_nonneg ξ]
  exact (norm_iteratedFDeriv_scaled_characteristic2_positive_le stdGauss2 isStandardized_stdGauss2
    (IsGaussian.memLp_id _ _ (by simp)) c hc hc1 ξ hξ k (by omega) hk).trans
      (mul_le_mul_of_nonneg_left hp (by positivity))

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    ‖(1 : ℝ) • ((0, 0) : ℝ × ℝ)‖ ≤ 1 ∧ (3 : ℕ) ≤ 3 :=
  ⟨zero_le_one, le_rfl, by norm_num, le_rfl⟩

/-- The literal normalized frequency gives an inverse-n factor for every positive base derivative through order three.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
actual `1 / sqrt n` composition used in the normalized-sum power. -/
theorem norm_iteratedFDeriv_normalized_characteristic2_positive_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 3 μ) {n : ℕ} (hn : 1 ≤ n) (ξ : ℝ × ℝ)
    (hξ : ‖ξ‖ ≤ Real.sqrt n) (k : ℕ) (hk1 : 1 ≤ k) (hk3 : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => characteristic2 μ ((Real.sqrt n)⁻¹ • η)) ξ‖ ≤
      (10 + 8 * (∫ z, ‖z‖ ^ 3 ∂μ)) * (n : ℝ)⁻¹ * (1 + ‖ξ‖) := by
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hs1 : 1 ≤ Real.sqrt (n : ℝ) := Real.one_le_sqrt.mpr (by exact_mod_cast hn)
  have hc1 : (Real.sqrt (n : ℝ))⁻¹ ≤ 1 := (inv_le_one₀ hs).mpr hs1
  have hsmall : ‖(Real.sqrt n)⁻¹ • ξ‖ ≤ 1 := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    exact (inv_mul_le_iff₀ hs).mpr (by simpa only [mul_one] using hξ)
  have hc2 : (Real.sqrt (n : ℝ))⁻¹ ^ 2 = (n : ℝ)⁻¹ := by
    rw [inv_pow, Real.sq_sqrt hn0.le]
  simpa only [hc2] using norm_iteratedFDeriv_scaled_characteristic2_positive_le
    μ hμ hmom (Real.sqrt n)⁻¹ (by positivity) hc1 ξ hsmall k hk1 hk3

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    1 ≤ (2 : ℕ) ∧ ‖((0, 0) : ℝ × ℝ)‖ ≤ Real.sqrt 2 ∧ (1 : ℕ) ≤ 3 ∧ (3 : ℕ) ≤ 3 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    by omega, by norm_num, by omega, le_rfl⟩

end Transformer.Modes
