import Transformer.Modes.Section3_CubicCorrection
import Transformer.Modes.Section3_GaussianDensity
/-!
# Gaussian comparison after the actual cubic correction

For the positive-sign characteristic function, the cubic coefficient in
arXiv:2412.09080v3, §5.4 `eq:br-9.10`, is the literal moment expression
`B(ξ) = -i E⟨ξ,X⟩³ / 6`. Here finite fourth moments control the difference
between one summand's characteristic function and the Gaussian function
with this cubic term retained. On the unit ball the remaining error is
bounded by a constant times the fourth power of the frequency norm.

The spectral Edgeworth correction is `G(ξ) B(ξ)`, where
`G(ξ) = exp(-|ξ|₂²/2)`. Replacing `B` by `G B` also preserves fourth-order
control, using the actual third moment. These are bounds for a single
summand; the normalized-sum estimate requires a further power argument.
Lean uses the product sup norm on the frequency space. The Gaussian
exponent retains the Euclidean quadratic form; comparing these norms
accounts for the constants. The characteristic function `G` equals
`2π` times the Gaussian density at the same frequency.
The constants are elementary explicit bounds, rather than the unspecified
constants in the source's `eq:br-9.10`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual Gaussian characteristic function differs from one quadratically.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the Gaussian comparison term. -/
theorem norm_characteristic2_stdGauss2_sub_one (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖characteristic2 stdGauss2 ξ - 1‖ ≤ 2 * ‖ξ‖ ^ 2 := by
  let Q : ℝ := ξ.1 ^ 2 + ξ.2 ^ 2
  have hQ0 : 0 ≤ Q := by dsimp [Q]; positivity
  have hQ2 : Q ≤ 2 * ‖ξ‖ ^ 2 := dot_self_le_two_norm_sq ξ
  have hr : ‖ξ‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg ξ]
  have hn : ‖((-(Q / 2) : ℝ) : ℂ)‖ = Q / 2 := by
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_nonpos (by linarith)]
    ring
  have he : characteristic2 stdGauss2 ξ = Complex.exp (((-(Q / 2) : ℝ) : ℂ)) := by
    rw [characteristic2_stdGauss2, Complex.ofReal_exp]
    congr 1
    dsimp [Q]
    push_cast
    ring
  rw [he]
  have h := Complex.norm_exp_sub_one_le (x := ((-(Q / 2) : ℝ) : ℂ)) (by rw [hn]; linarith)
  rw [hn] at h
  linarith

example : ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 := by norm_num

/-- The actual Gaussian characteristic function has fourth-order quadratic error.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the Gaussian Taylor step. -/
theorem norm_characteristic2_stdGauss2_sub_quadratic (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖characteristic2 stdGauss2 ξ - (1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2)‖ ≤ ‖ξ‖ ^ 4 := by
  let Q : ℝ := ξ.1 ^ 2 + ξ.2 ^ 2
  have hQ0 : 0 ≤ Q := by dsimp [Q]; positivity
  have hQ2 : Q ≤ 2 * ‖ξ‖ ^ 2 := dot_self_le_two_norm_sq ξ
  have hr : ‖ξ‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg ξ]
  have hn : ‖((-(Q / 2) : ℝ) : ℂ)‖ = Q / 2 := by
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_nonpos (by linarith)]
    ring
  have he : characteristic2 stdGauss2 ξ - (1 - (Q : ℂ) / 2) =
      Complex.exp (((-(Q / 2) : ℝ) : ℂ)) - 1 - ((-(Q / 2) : ℝ) : ℂ) := by
    rw [characteristic2_stdGauss2, Complex.ofReal_exp]
    dsimp [Q]
    have hq : -(ξ.1 ^ 2 + ξ.2 ^ 2) / 2 = -((ξ.1 ^ 2 + ξ.2 ^ 2) / 2) := by ring
    rw [hq]
    push_cast
    ring
  change ‖characteristic2 stdGauss2 ξ - (1 - (Q : ℂ) / 2)‖ ≤ _
  rw [he]
  have h := Complex.norm_exp_sub_one_sub_id_le (x := ((-(Q / 2) : ℝ) : ℂ)) (by rw [hn]; linarith)
  rw [hn] at h
  have hsq := pow_le_pow_left₀ hQ0 hQ2 2
  nlinarith

example : ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 := by norm_num

/-- Retaining the actual cubic moment leaves fourth-order Gaussian comparison error.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4 `eq:br-9.10`,
for one summand before taking the normalized-sum power. -/
theorem norm_characteristic2_sub_gaussian_sub_cubicCorrection
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖characteristic2 μ ξ - characteristic2 stdGauss2 ξ - cubicCharacteristicCorrection μ ξ‖ ≤
      ((8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) + 1) * ‖ξ‖ ^ 4 := by
  let P : ℂ := 1 - ((ξ.1 ^ 2 + ξ.2 ^ 2 : ℝ) : ℂ) / 2
  have he : characteristic2 μ ξ - characteristic2 stdGauss2 ξ - cubicCharacteristicCorrection μ ξ =
      (characteristic2 μ ξ - (P + cubicCharacteristicCorrection μ ξ)) + (P - characteristic2 stdGauss2 ξ) := by ring
  rw [he]
  calc
    _ ≤ ‖characteristic2 μ ξ - (P + cubicCharacteristicCorrection μ ξ)‖ +
        ‖P - characteristic2 stdGauss2 ξ‖ := norm_add_le _ _
    _ ≤ (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) * ‖ξ‖ ^ 4 + ‖ξ‖ ^ 4 := by
      apply add_le_add (norm_characteristic2_sub_cubicCorrection μ hμ hmom ξ)
      rw [norm_sub_rev]
      exact norm_characteristic2_stdGauss2_sub_quadratic ξ hξ
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 :=
  ⟨inferInstance, isStandardized_stdGauss2,
    IsGaussian.memLp_id _ _ (by simp), by norm_num⟩

/-- The Gaussian-damped cubic correction also leaves fourth-order error.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:br-9.10`.
The Gaussian factor belongs to the spectral correction; this estimate
controls its difference from the literal cubic Taylor coefficient. -/
theorem norm_characteristic2_sub_gaussian_mul_cubicCorrection
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) :
    ‖characteristic2 μ ξ - characteristic2 stdGauss2 ξ -
      characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ ≤
      ((8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) + 1 + (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)) * ‖ξ‖ ^ 4 := by
  have hM : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hB := norm_cubicCharacteristicCorrection_le μ (hmom.mono_exponent (by norm_num)) ξ
  have hG : ‖1 - characteristic2 stdGauss2 ξ‖ ≤ 2 * ‖ξ‖ ^ 2 := by
    rw [norm_sub_rev]
    exact norm_characteristic2_stdGauss2_sub_one ξ hξ
  have hp : ‖ξ‖ ^ 5 ≤ ‖ξ‖ ^ 4 := by
    have h := mul_le_mul_of_nonneg_right hξ (pow_nonneg (norm_nonneg ξ) 4)
    nlinarith
  have hBG : ‖cubicCharacteristicCorrection μ ξ * (1 - characteristic2 stdGauss2 ξ)‖ ≤
      (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 4 := by
    rw [norm_mul]
    calc
      _ ≤ ((4 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 3) * (2 * ‖ξ‖ ^ 2) := by gcongr
      _ = (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 5 := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left hp (by positivity)
  have he : characteristic2 μ ξ - characteristic2 stdGauss2 ξ -
      characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ =
      (characteristic2 μ ξ - characteristic2 stdGauss2 ξ - cubicCharacteristicCorrection μ ξ) +
        cubicCharacteristicCorrection μ ξ * (1 - characteristic2 stdGauss2 ξ) := by ring
  rw [he]
  calc
    _ ≤ ‖characteristic2 μ ξ - characteristic2 stdGauss2 ξ - cubicCharacteristicCorrection μ ξ‖ +
      ‖cubicCharacteristicCorrection μ ξ * (1 - characteristic2 stdGauss2 ξ)‖ := norm_add_le _ _
    _ ≤ (((8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) + 1) * ‖ξ‖ ^ 4) +
      (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ) * ‖ξ‖ ^ 4 :=
      add_le_add (norm_characteristic2_sub_gaussian_sub_cubicCorrection μ hμ hmom ξ hξ) hBG
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 :=
  ⟨inferInstance, isStandardized_stdGauss2,
    IsGaussian.memLp_id _ _ (by simp), by norm_num⟩

end Transformer.Modes
