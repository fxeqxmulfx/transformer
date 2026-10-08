import Transformer.Modes.Section3_CubicPower
import Transformer.Modes.Section3_FirstDerivativeFrequency
/-!
# Cubic Edgeworth error for the actual normalized-sum characteristic function

Section 5.4 `eq:br-9.10` of arXiv:2412.09080v3 asks for a polynomial
Gaussian bound at rate `1 / n` when `s = 3`. Here that zero-derivative
estimate is proved for the literal normalized sum of a fixed centered,
identity-covariance law with finite fourth norm moment.

The cubic term is the actual third-moment coefficient. Its exact scaling
and the Gaussian characteristic power identity retain precisely
`G(ξ) B(ξ) / sqrt n`. The two power remainders scale as
`n² |ξ/sqrt n|⁶` and `n |ξ/sqrt n|⁴`, both of order `1 / n`.
Their common modulus bound gives Gaussian damping on the growing ball.

The elementary proof gives exponent `1/8`, where the source's display
uses `1/4`; this weaker damping still has every polynomial moment
integrable. The constant here depends on the fixed law's third and
fourth moments. The statement makes this fixed-law dependence explicit
through an existential constant, as required for §3 `thm:br`.

The remaining derivative and exterior estimates are further proof steps
for the spatially weighted density expansion. The present statement
controls the actual characteristic function itself without such inputs.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The actual cubic-corrected characteristic error has a Gaussian `1 / n` majorant.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4 `eq:br-9.10`,
its zero-derivative step. The source's damping exponent is `1/4`; this
fixed-law elementary bound uses `1/8` with explicit polynomial degrees. -/
theorem exists_characteristic_scaledSum_sub_gaussian_cubic_bound
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) :
    ∃ a C : ℝ, 0 < a ∧ 0 ≤ C ∧ ∀ n : ℕ, 4 ≤ n → ∀ ξ : ℝ × ℝ,
      ‖ξ‖ ≤ a * Real.sqrt n →
      ‖characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
          characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ ≤
        C * (n : ℝ)⁻¹ * (‖ξ‖ ^ 4 + ‖ξ‖ ^ 6) *
          Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 8) := by
  obtain ⟨a, ha, hb⟩ := exists_characteristic2_gaussian_bound μ hμ (hmom.mono_exponent (by norm_num))
  let K0 : ℝ := 4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1
  let K4 : ℝ := (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) + 1 + (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)
  have hM3 : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hM4 : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 4 ∂μ := integral_nonneg fun z => by positivity
  have hK0 : 0 ≤ K0 := by dsimp [K0]; positivity
  have hK4 : 0 ≤ K4 := by dsimp [K4]; positivity
  refine ⟨min a 1, K0 ^ 2 + K4, by positivity, by positivity, ?_⟩
  intro n hn ξ hξ
  have hn0 : 0 < (n : ℝ) := by exact_mod_cast (show 0 < n by omega)
  have hs : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr hn0
  have hsq : (Real.sqrt (n : ℝ)) ^ 2 = n := Real.sq_sqrt hn0.le
  let c : ℝ := (Real.sqrt (n : ℝ))⁻¹
  let η : ℝ × ℝ := c • ξ
  let Q : ℝ := ξ.1 ^ 2 + ξ.2 ^ 2
  let q : ℝ := η.1 ^ 2 + η.2 ^ 2
  let r : ℝ := Real.exp (-q / 4)
  let E : ℝ := Real.exp (-Q / 8)
  have hc : 0 < c := by dsimp [c]; positivity
  have hη : ‖η‖ ≤ min a 1 := by
    dsimp [η, c]
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hs)]
    exact (inv_mul_le_iff₀ hs).mpr (by simpa [mul_comm] using hξ)
  have hη1 : ‖η‖ ≤ 1 := (le_min_iff.mp hη).2
  have hχ : ‖characteristic2 μ η‖ ≤ r := hb η ((le_min_iff.mp hη).1)
  have hG : ‖characteristic2 stdGauss2 η‖ ≤ r := by
    rw [characteristic2_stdGauss2, Complex.norm_real, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _)]
    apply Real.exp_le_exp.mpr
    dsimp [r, q]; nlinarith [sq_nonneg η.1, sq_nonneg η.2]
  have hcoef : (n : ℝ) * c ^ 2 = 1 := by
    dsimp [c]
    calc
      _ = (Real.sqrt n) ^ 2 * (Real.sqrt n)⁻¹ ^ 2 := by rw [hsq]
      _ = _ := by field_simp
  have hcinv : c ^ 2 = (n : ℝ)⁻¹ := by dsimp [c]; rw [inv_pow, hsq]
  have hnq : (n : ℝ) * q = Q := by
    dsimp [q, η, Q]
    calc
      _ = ((n : ℝ) * c ^ 2) * (ξ.1 ^ 2 + ξ.2 ^ 2) := by ring
      _ = _ := by rw [hcoef, one_mul]
  have hdecay (j : ℕ) (hj : j ≤ 2) : r ^ (n - j) ≤ E := by
    dsimp [r, E]
    rw [← Real.exp_nat_mul]
    apply Real.exp_le_exp.mpr
    have hn4 : (4 : ℝ) ≤ n := by exact_mod_cast hn
    have hj2 : (j : ℝ) ≤ 2 := by exact_mod_cast hj
    have hcast : ((n - j : ℕ) : ℝ) = (n : ℝ) - j := by
      have h : ((n - j : ℕ) : ℝ) + j = (n : ℝ) := by exact_mod_cast (show n - j + j = n by omega)
      linarith
    rw [hcast]
    have hq0 : 0 ≤ q := by dsimp [q]; positivity
    nlinarith [mul_nonneg (show 0 ≤ (n : ℝ) - 2 * j by linarith) hq0]
  have hnorm : ‖η‖ = c * ‖ξ‖ := by
    rw [show η = c • ξ from rfl, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
  have hscale6 : (n : ℝ) ^ 2 * ‖η‖ ^ 6 = (n : ℝ)⁻¹ * ‖ξ‖ ^ 6 := by
    rw [hnorm]
    calc
      _ = ((n : ℝ) * c ^ 2) ^ 2 * c ^ 2 * ‖ξ‖ ^ 6 := by ring
      _ = _ := by rw [hcoef, hcinv]; ring
  have hscale4 : (n : ℝ) * ‖η‖ ^ 4 = (n : ℝ)⁻¹ * ‖ξ‖ ^ 4 := by
    rw [hnorm]
    calc
      _ = ((n : ℝ) * c ^ 2) * c ^ 2 * ‖ξ‖ ^ 4 := by ring
      _ = _ := by rw [hcoef, hcinv]; ring
  have hcorrection : (n : ℂ) * cubicCharacteristicCorrection μ η * characteristic2 stdGauss2 ξ =
      (c : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ := by
    change (n : ℂ) * cubicCharacteristicCorrection μ ((Real.sqrt n)⁻¹ • ξ) * characteristic2 stdGauss2 ξ = _
    rw [normalized_cubicCharacteristicCorrection μ (by omega : 1 ≤ n)]
    dsimp [c]
    ring
  have hpow := norm_characteristic2_pow_sub_gaussian_cubic_le μ hμ hmom η hη1 hχ hG (by omega : 2 ≤ n)
  rw [characteristic2_stdGauss2_scaled_power n (by omega : 1 ≤ n) ξ, hcorrection] at hpow
  rw [characteristic_scaledSum]
  change ‖characteristic2 μ η ^ n - characteristic2 stdGauss2 ξ -
    (c : ℂ) * characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ ≤ _
  calc
    _ ≤ (n : ℝ) ^ 2 * K0 ^ 2 * ‖η‖ ^ 6 * r ^ (n - 2) +
        (n : ℝ) * K4 * ‖η‖ ^ 4 * r ^ (n - 1) := hpow
    _ ≤ (n : ℝ) ^ 2 * K0 ^ 2 * ‖η‖ ^ 6 * E + (n : ℝ) * K4 * ‖η‖ ^ 4 * E := by
      gcongr
      · exact hdecay 2 (by omega)
      · exact hdecay 1 (by omega)
    _ = (K0 ^ 2 * ((n : ℝ) ^ 2 * ‖η‖ ^ 6) + K4 * ((n : ℝ) * ‖η‖ ^ 4)) * E := by ring
    _ = (K0 ^ 2 * ((n : ℝ)⁻¹ * ‖ξ‖ ^ 6) + K4 * ((n : ℝ)⁻¹ * ‖ξ‖ ^ 4)) * E := by rw [hscale6, hscale4]
    _ ≤ (K0 ^ 2 + K4) * (n : ℝ)⁻¹ * (‖ξ‖ ^ 4 + ‖ξ‖ ^ 6) * E := by
      have hExtra : 0 ≤ K0 ^ 2 * ‖ξ‖ ^ 4 + K4 * ‖ξ‖ ^ 6 := by positivity
      gcongr
      nlinarith

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧ MemLp id 4 stdGauss2 :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp)⟩

/-- The actual cubic-corrected Fourier error is eventually integrable.
Source: arXiv:2412.09080v3, §5.4 `eq:higher-error-goal` and `eq:big-z-poly`.
The integrable characteristic power supplies the sum term, while
Gaussian damping supplies the literal third-moment correction. -/
theorem eventually_integrable_characteristic_scaledSum_cubic_error
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hmom : MemLp id 3 μ)
    (hexp : HasExpMoments μ) (hcf : HasIntegrableCharFun μ) :
    ∀ᶠ n : ℕ in atTop, Integrable (fun ξ : ℝ × ℝ =>
      characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
        characteristic2 stdGauss2 ξ - (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
          characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ) := by
  have hB : Integrable (fun ξ : ℝ × ℝ =>
      characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ) := by
    convert (integrable_cubicCharacteristicCorrection_gaussian_and_bound μ hexp hmom
      (b := 1 / 2) (by norm_num)).1 using 1
    funext ξ
    rw [characteristic2_stdGauss2]
    have he : -(ξ.1 ^ 2 + ξ.2 ^ 2) / 2 = -(1 / 2 : ℝ) * (ξ.1 ^ 2 + ξ.2 ^ 2) := by ring
    rw [he, mul_comm]
  filter_upwards [eventually_integrable_characteristic_scaledSum_gaussian_error μ hcf] with n hI
  have h : Integrable (fun ξ : ℝ × ℝ =>
      characteristic2 ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ξ -
        ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) -
          (((Real.sqrt n)⁻¹ : ℝ) : ℂ) *
            (characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ)) :=
    hI.sub (hB.const_mul (((Real.sqrt n)⁻¹ : ℝ) : ℂ))
  simpa only [characteristic2_stdGauss2, mul_assoc] using h

example : IsProbabilityMeasure stdGauss2 ∧ MemLp id 3 stdGauss2 ∧
    HasExpMoments stdGauss2 ∧ HasIntegrableCharFun stdGauss2 :=
  ⟨inferInstance, IsGaussian.memLp_id _ _ (by simp),
    hasExpMoments_stdGauss2, hasIntegrableCharFun_stdGauss2⟩

end Transformer.Modes
