import Transformer.Modes.Section3_CubicTailRates
import Transformer.Modes.Section3_CubicFrequency
/-!
# Gaussian cubic correction: moments and exterior rates

The extra term for `s = 3` in arXiv:2412.09080v3, §5.4 `eq:big-z-poly`,
is a Gaussian times a cubic polynomial. Here it is the literal spectral
coefficient `G(ξ) B(ξ)`, with `B(ξ) = -i E⟨ξ,X⟩³ / 6`. For a centered,
identity-covariance law, the proved cumulant identity supplies exactly
the coefficients in §3.1 `eq:psi`.

Its polynomially weighted norm is bounded by the actual third norm
moment times a finite Gaussian moment. Additional Gaussian moments
then give every natural inverse-radius tail estimate. At radius
`a sqrt n`, the first inverse-radius power combines with the exact
correction scale `1 / sqrt n` to give `1 / n`.

The polynomial weight may have any natural degree, including zero.
Its exponent adds to the cubic degree before the tail moment is taken.
The inverse-radius power is arbitrary as well; the normalized tail
uses power one and the resulting fourth Gaussian frequency moment.

The bounds retain the actual complex coefficient and Gaussian factor.
They do not require the correction to be nonnegative, which is needed
only for a probability density. The spectral correction is signed and
is compared through its norm. Relating its inverse Fourier integral
to the actual spatial Hermite expression is a further proof step.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual Gaussian cubic correction has a third-moment polynomial majorant.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10` and `eq:big-z-poly`. -/
theorem norm_pow_mul_gaussian_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ) (k : ℕ) (ξ : ℝ × ℝ) :
    ‖ξ‖ ^ k * ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ ≤
      (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * (‖ξ‖ ^ (k + 3) * ‖characteristic2 stdGauss2 ξ‖) := by
  have hB := norm_cubicCharacteristicCorrection_le μ hmom ξ
  rw [norm_mul]
  calc
    _ ≤ ‖ξ‖ ^ k * (‖characteristic2 stdGauss2 ξ‖ * ((4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 3)) := by gcongr
    _ = _ := by rw [pow_add]; ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp)⟩

/-- Every polynomial frequency weight of the actual cubic correction is integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`; the constant retains
its dependence on the third moment through a finite Gaussian integral. -/
theorem integrable_norm_pow_gaussian_cubic_and_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hexp : HasExpMoments μ) (k : ℕ) :
    Integrable (fun ξ : ℝ × ℝ => ‖ξ‖ ^ k * ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) ∧
    (∫ ξ : ℝ × ℝ, ‖ξ‖ ^ k * ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) ≤
      (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) *
        ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ (k + 3) * ‖characteristic2 stdGauss2 ξ‖ := by
  let K : ℝ := (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)
  let F : ℝ × ℝ → ℝ := fun ξ => ‖ξ‖ ^ k * ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖
  have hB := (contDiff_cubicCharacteristicCorrection μ hexp).continuous
  have hG := continuous_characteristic2 stdGauss2
  have hF : Continuous F := by dsimp [F]; fun_prop
  have hF0 (ξ : ℝ × ℝ) : 0 ≤ F ξ := by dsimp [F]; positivity
  have hM := (integrable_norm_pow_characteristic2_stdGauss2 (k + 3)).const_mul K
  have hp (ξ : ℝ × ℝ) : F ξ ≤ K * (‖ξ‖ ^ (k + 3) * ‖characteristic2 stdGauss2 ξ‖) :=
    norm_pow_mul_gaussian_cubic_le μ hmom k ξ
  have hI : Integrable F := hM.mono' hF.aestronglyMeasurable (ae_of_all _ fun ξ => by
    rw [Real.norm_eq_abs, abs_of_nonneg (hF0 ξ)]
    exact hp ξ)
  refine ⟨hI, ?_⟩
  calc
    _ ≤ ∫ ξ : ℝ × ℝ, K * (‖ξ‖ ^ (k + 3) * ‖characteristic2 stdGauss2 ξ‖) :=
      integral_mono_ae hI hM (ae_of_all _ hp)
    _ = _ := integral_const_mul _ _

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧ HasExpMoments stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasExpMoments_stdGauss2⟩

/-- Additional Gaussian moments control the actual cubic correction's exterior integral.
Source: arXiv:2412.09080v3, §5.4 `eq:big-z-poly`. This inverse-radius
bound replaces the source's stronger exponential estimate at fixed law. -/
theorem integral_norm_pow_gaussian_cubic_tail_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hexp : HasExpMoments μ) (k l : ℕ) {r : ℝ} (hr : 0 < r) :
    (∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖}, ‖ξ‖ ^ k * ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) ≤
      (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * (r ^ l)⁻¹ *
        ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ (k + 3 + l) * ‖characteristic2 stdGauss2 ξ‖ := by
  let K : ℝ := (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)
  have hK : 0 ≤ K := by
    have hM : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
    dsimp [K]; positivity
  have hI := (integrable_norm_pow_gaussian_cubic_and_bound μ hmom hexp k).1
  have hM := (integrable_norm_pow_characteristic2_stdGauss2 (k + 3)).const_mul K
  have he (ξ : ℝ × ℝ) : Real.exp (-(1 / 2 : ℝ) * (ξ.1 ^ 2 + ξ.2 ^ 2)) = ‖characteristic2 stdGauss2 ξ‖ := by
    rw [characteristic2_stdGauss2, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    congr 1
    ring
  have hT := integral_norm_pow_gaussian_frequency_tail_le (k + 3) l (b := 1 / 2) (by norm_num) hr
  simp only [he] at hT
  calc
    _ ≤ ∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖}, K * (‖ξ‖ ^ (k + 3) * ‖characteristic2 stdGauss2 ξ‖) := by
      apply setIntegral_mono_on hI.integrableOn hM.integrableOn
        (isClosed_le continuous_const continuous_norm).measurableSet
      intro ξ _
      exact norm_pow_mul_gaussian_cubic_le μ hmom k ξ
    _ = K * ∫ ξ : ℝ × ℝ in {ξ | r ≤ ‖ξ‖}, ‖ξ‖ ^ (k + 3) * ‖characteristic2 stdGauss2 ξ‖ := integral_const_mul _ _
    _ ≤ K * ((r ^ l)⁻¹ * ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ (k + 3 + l) * ‖characteristic2 stdGauss2 ξ‖) :=
      mul_le_mul_of_nonneg_left hT hK
    _ = _ := by dsimp [K]; ring

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ (0 : ℝ) < 1 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasExpMoments_stdGauss2, one_pos⟩

/-- The actual correction scaled by `1 / sqrt n` has an exterior inverse-n rate.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4 `eq:big-z-poly`.
The normalization and growing radius are literal, and the constant is nonnegative. -/
theorem exists_integral_scaled_gaussian_cubic_tail_rate_inv_nat
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hexp : HasExpMoments μ) {a : ℝ} (ha : 0 < a) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ n : ℕ, 1 ≤ n →
      (∫ ξ : ℝ × ℝ in {ξ | a * Real.sqrt n ≤ ‖ξ‖},
        ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖) ≤
          C * (n : ℝ)⁻¹ := by
  let K : ℝ := (4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)
  let J : ℝ := ∫ ξ : ℝ × ℝ, ‖ξ‖ ^ 4 * ‖characteristic2 stdGauss2 ξ‖
  have hM : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hJ : 0 ≤ J := integral_nonneg fun ξ => by positivity
  refine ⟨K * a⁻¹ * J, by dsimp [K]; positivity, ?_⟩
  intro n hn
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n from hn)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hcinv : (Real.sqrt (n : ℝ))⁻¹ ^ 2 = (n : ℝ)⁻¹ := by rw [inv_pow, Real.sq_sqrt hn0.le]
  have he (ξ : ℝ × ℝ) : ‖(((Real.sqrt n)⁻¹ : ℝ) : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ =
      (Real.sqrt n)⁻¹ * ‖characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ := by
    rw [mul_assoc, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
  simp only [he]
  rw [integral_const_mul]
  have hT := integral_norm_pow_gaussian_cubic_tail_le μ hmom hexp 0 1 (mul_pos ha hs)
  simp only [pow_zero, pow_one, one_mul, Nat.reduceAdd] at hT
  calc
    _ ≤ (Real.sqrt n)⁻¹ * (K * (a * Real.sqrt n)⁻¹ * J) := mul_le_mul_of_nonneg_left hT (by positivity)
    _ = K * a⁻¹ * J * (Real.sqrt n)⁻¹ ^ 2 := by rw [mul_inv_rev]; ring
    _ = _ := by rw [hcinv]

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ (0 : ℝ) < 1 ∧ 1 ≤ (1 : ℕ) :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp), hasExpMoments_stdGauss2, one_pos, le_rfl⟩

end Transformer.Modes
