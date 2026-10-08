import Transformer.Modes.Section3_FourthOrderPhase
import Transformer.Modes.Section3_CharacteristicTaylor
/-!
# The actual cubic characteristic expansion under finite fourth moments

For `s = 3` in arXiv:2412.09080v3, §3 `thm:br`, the characteristic
Taylor polynomial includes the actual third moment. Centering and
identity covariance fix the lower coefficients, and the imaginary
phase gives the sign `-i E⟨ξ,X⟩³ / 6` for the positive characteristic
convention. The remainder uses only the finite fourth absolute moment.

The dot-product moment bounds keep the factor `2^k` from the product
sup norm. The resulting fourth-order error constant is `8/3`, from
`2^4 / 3!`; it is nonsharp because the scalar Taylor bound uses `3!`.
The cubic term is separately bounded by the third moment. These are
single-summand estimates; normalized-sum powers and Gaussian damping
are additional steps of §5.4 `eq:br-9.10`.
Every lower polynomial moment is integrable under the same fourth-moment
hypothesis on a probability law, so the polynomial subtraction is an
identity of actual integrals rather than formal coefficients.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Absolute dot moments are integrable and bounded by the actual norm moment.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4 `eq:br-9.10`, the moment
comparison. The factor `2^k` accounts for the product sup norm. -/
theorem integrable_abs_dot_pow_and_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] {k : ℕ}
    (hmom : MemLp id k μ) (ξ : ℝ × ℝ) :
    Integrable (fun z => |ξ.1 * z.1 + ξ.2 * z.2| ^ k) μ ∧
      (∫ z, |ξ.1 * z.1 + ξ.2 * z.2| ^ k ∂μ) ≤
        2 ^ k * ‖ξ‖ ^ k * ∫ z, ‖z‖ ^ k ∂μ := by
  have hM := hmom.integrable_norm_pow'
  have hp (z : ℝ × ℝ) : |ξ.1 * z.1 + ξ.2 * z.2| ^ k ≤
      (2 ^ k * ‖ξ‖ ^ k) * ‖z‖ ^ k := by
    calc
      _ ≤ (2 * ‖ξ‖ * ‖z‖) ^ k := pow_le_pow_left₀ (abs_nonneg _) (abs_dot_le_two_norm_mul ξ z) k
      _ = _ := by rw [mul_pow, mul_pow]
  have hI : Integrable (fun z => |ξ.1 * z.1 + ξ.2 * z.2| ^ k) μ := by
    apply (hM.const_mul (2 ^ k * ‖ξ‖ ^ k)).mono' (by fun_prop)
    exact ae_of_all _ fun z => by
      rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
      exact hp z
  refine ⟨hI, ?_⟩
  calc
    _ ≤ ∫ z, (2 ^ k * ‖ξ‖ ^ k) * ‖z‖ ^ k ∂μ :=
      integral_mono_ae hI (hM.const_mul _) (ae_of_all _ hp)
    _ = _ := integral_const_mul _ _

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual cubic characteristic polynomial has a fourth-moment remainder.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4 `eq:br-9.10`,
the unscaled phase expansion, retaining the actual third moment and its sign. -/
theorem norm_characteristic2_sub_cubic_moment
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) :
    ‖characteristic2 μ ξ - (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2 -
      Complex.I * ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6)‖ ≤
      (∫ z, |ξ.1 * z.1 + ξ.2 * z.2| ^ 4 ∂μ) / 6 := by
  let f : ℝ × ℝ → ℝ := fun z => ξ.1 * z.1 + ξ.2 * z.2
  have hf : Continuous f := by dsimp [f]; fun_prop
  have hF : MemLp f 4 μ := memLp_dot μ hmom ξ
  have hpoly (k : ℕ) (hk : k ∈ Finset.range (3 + 1)) :
      Integrable (fun z => (1 : ℝ) * f z ^ k) μ := by
    have hk4 : k ≤ 4 := by have := Finset.mem_range.mp hk; omega
    have hp : MemLp f (k : ℝ≥0∞) μ := hF.mono_exponent (by exact_mod_cast hk4)
    apply hp.integrable_norm_pow'.mono' ((continuous_const.mul (hf.pow k)).aestronglyMeasurable)
    exact ae_of_all _ fun z => by
      change ‖(1 : ℝ) * f z ^ k‖ ≤ ‖f z‖ ^ k
      rw [norm_mul, norm_one, one_mul, norm_pow]
  have hM : Integrable (fun z => |(1 : ℝ)| * |f z| ^ 4) μ := by
    simpa only [abs_one, one_mul, Real.norm_eq_abs] using hF.integrable_norm_pow'
  have h := norm_integral_real_weighted_phase_sub_taylor μ (fun _ => (1 : ℝ)) f hf 3 hpoly hM
  rw [integral_real_weighted_phase_taylor μ (fun _ => (1 : ℝ)) f 3 hpoly] at h
  have hI : Complex.I ^ 3 = -Complex.I := by
    rw [pow_succ, Complex.I_sq]
    ring
  norm_num [Finset.sum_range_succ, hI, Complex.I_sq] at h
  have hmean : ∫ z, f z ∂μ = 0 := integral_dot_of_standardized μ hμ ξ
  have hvariance : ∫ z, f z ^ 2 ∂μ = ξ.1 ^ 2 + ξ.2 ^ 2 := integral_dot_sq_of_standardized μ hμ ξ
  rw [hmean, hvariance] at h
  norm_num at h
  change ‖(∫ z, Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
    (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2 -
      Complex.I * ((∫ z, f z ^ 3 ∂μ : ℝ) : ℂ) / 6)‖ ≤ (∫ z, |f z| ^ 4 ∂μ) / 6
  convert h using 1
  congr 1
  push_cast
  ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The cubic characteristic remainder is globally bounded by `M₄ |ξ|⁴`.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the unscaled Taylor step
for `s = 3`, before taking normalized-sum powers. -/
theorem norm_characteristic2_sub_cubic
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) :
    ‖characteristic2 μ ξ - (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2 -
      Complex.I * ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6)‖ ≤
      (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 4 := by
  have h := norm_characteristic2_sub_cubic_moment μ hμ hmom ξ
  have hB := (integrable_abs_dot_pow_and_bound μ hmom ξ).2
  norm_num at hB
  linarith

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual third moment of a linear form is bounded by its norm moment.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the cubic characteristic coefficient. -/
theorem abs_integral_dot_cube_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    |∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ| ≤
      8 * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 3 := by
  calc
    _ ≤ ∫ z, |ξ.1 * z.1 + ξ.2 * z.2| ^ 3 ∂μ := by
      have h : ‖∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ‖ ≤
          ∫ z, ‖(ξ.1 * z.1 + ξ.2 * z.2) ^ 3‖ ∂μ :=
        norm_integral_le_integral_norm (μ := μ) (fun z => (ξ.1 * z.1 + ξ.2 * z.2) ^ 3)
      simpa only [norm_pow, Real.norm_eq_abs] using h
    _ ≤ 2 ^ 3 * ‖ξ‖ ^ 3 * ∫ z, ‖z‖ ^ 3 ∂μ := (integrable_abs_dot_pow_and_bound μ hmom ξ).2
    _ = _ := by norm_num; ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The literal imaginary cubic phase term has size at most `4 M₃ |ξ|³ / 3`.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`,
the third-moment coefficient in the positive-sign characteristic convention. -/
theorem norm_cubic_characteristic_phase_term_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) (ξ : ℝ × ℝ) :
    ‖Complex.I * ((∫ z, (ξ.1 * z.1 + ξ.2 * z.2) ^ 3 ∂μ : ℝ) : ℂ) / 6‖ ≤
      (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 3 := by
  rw [norm_div, norm_mul, Complex.norm_I, one_mul, Complex.norm_real, Real.norm_eq_abs, Complex.norm_ofNat]
  have h := abs_integral_dot_cube_le μ hmom ξ
  linarith


example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

end Transformer.Modes
