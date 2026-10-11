import Transformer.Modes.Section3_CubicPowerFactorization
import Mathlib.Analysis.Calculus.ContDiff.Bounds

/-!
# Uniform derivative bounds for mixed characteristic powers

The exact power factorization in arXiv:2412.09080v3, §5.4
`eq:br-9.10`, leaves powers whose exponents vary from zero to `n`.
The first three derivatives of each power retain its undifferentiated
modulus factor. If every positive-order base derivative is at most
`A` and `n A ≤ P`, every such power has a common derivative bound
depending on `P`, rather than on the individual exponent.

Natural subtraction keeps the formulas valid for powers zero, one,
and two. The common retained modulus power is `m-3`; the positive
coefficients that vanish in these small cases are never cancelled.
This supplies the factors used in the actual cubic error, after its
literal frequency normalization makes `n A` polynomial in frequency.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The first derivative of an actual complex power retains its modulus power.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
first-order product term of a mixed-power factor. -/
theorem norm_fderiv_complex_power_le (f : ℝ × ℝ → ℂ) (n : ℕ)
    (ξ : ℝ × ℝ) (hf : DifferentiableAt ℝ f ξ) :
    ‖fderiv ℝ (fun η => f η ^ n) ξ‖ ≤
      (n : ℝ) * ‖f ξ‖ ^ (n - 1) * ‖fderiv ℝ f ξ‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by positivity)
  intro v
  rw [fderiv_complex_power_apply f n ξ v hf]
  simp only [norm_mul, norm_pow, Complex.norm_natCast]
  calc
    _ ≤ (n : ℝ) * ‖f ξ‖ ^ (n - 1) * (‖fderiv ℝ f ξ‖ * ‖v‖) := by
      gcongr
      exact (fderiv ℝ f ξ).le_opNorm v
    _ = _ := by ring

example : DifferentiableAt ℝ (characteristic2 stdGauss2) (0, 0) :=
  (contDiff_characteristic2 stdGauss2 (k := 3)
    (IsGaussian.memLp_id _ _ (by simp))).differentiable (by norm_num) _

/-- Both second derivative terms of an actual power retain their base powers.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
second-order product terms of a mixed-power factor. -/
theorem norm_iteratedFDeriv_two_complex_power_le (f : ℝ × ℝ → ℂ) (n : ℕ)
    (hf : ContDiff ℝ 2 f) (ξ : ℝ × ℝ) :
    ‖iteratedFDeriv ℝ 2 (fun η => f η ^ n) ξ‖ ≤
      (n : ℝ) ^ 2 * ‖f ξ‖ ^ (n - 2) * ‖fderiv ℝ f ξ‖ ^ 2 +
        (n : ℝ) * ‖f ξ‖ ^ (n - 1) * ‖iteratedFDeriv ℝ 2 f ξ‖ := by
  have hcoef : (n : ℝ) * (n - 1 : ℕ) ≤ (n : ℝ) ^ 2 := by
    have h : ((n - 1 : ℕ) : ℝ) ≤ n := by exact_mod_cast (show n - 1 ≤ n by omega)
    nlinarith [mul_nonneg (Nat.cast_nonneg n) (sub_nonneg.mpr h)]
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity)
  intro v
  rw [iteratedFDeriv_two_complex_power_apply f n hf]
  calc
    _ ≤ ‖(n : ℂ) * (n - 1 : ℕ) * f ξ ^ (n - 2) *
        fderiv ℝ f ξ (v 0) * fderiv ℝ f ξ (v 1)‖ +
        ‖(n : ℂ) * f ξ ^ (n - 1) * iteratedFDeriv ℝ 2 f ξ v‖ := norm_add_le _ _
    _ = ((n : ℝ) * (n - 1 : ℕ)) * ‖f ξ‖ ^ (n - 2) *
        (‖fderiv ℝ f ξ (v 0)‖ * ‖fderiv ℝ f ξ (v 1)‖) +
        (n : ℝ) * ‖f ξ‖ ^ (n - 1) * ‖iteratedFDeriv ℝ 2 f ξ v‖ := by
      simp only [norm_mul, norm_pow, Complex.norm_natCast]
      ring
    _ ≤ (n : ℝ) ^ 2 * ‖f ξ‖ ^ (n - 2) *
        ((‖fderiv ℝ f ξ‖ * ‖v 0‖) * (‖fderiv ℝ f ξ‖ * ‖v 1‖)) +
        (n : ℝ) * ‖f ξ‖ ^ (n - 1) *
          (‖iteratedFDeriv ℝ 2 f ξ‖ * ∏ i, ‖v i‖) := by
      gcongr
      · exact (fderiv ℝ f ξ).le_opNorm _
      · exact (fderiv ℝ f ξ).le_opNorm _
      · exact (iteratedFDeriv ℝ 2 f ξ).le_opNorm _
    _ = _ := by rw [Fin.prod_univ_two]; ring

example : ContDiff ℝ 2 (characteristic2 stdGauss2) :=
  contDiff_characteristic2 stdGauss2 (k := 2) (IsGaussian.memLp_id _ _ (by simp))

/-- Every power up to `n` has one bound for all derivative orders through three.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the
mixed-power bounds after the exact cubic-error factorization.
The positive derivative input is on the actual base function. -/
theorem norm_iteratedFDeriv_complex_power_le_of_derivative_bounds
    (f : ℝ × ℝ → ℂ) (hf : ContDiff ℝ 3 f) (ξ : ℝ × ℝ)
    (n : ℕ) {r A P : ℝ} (hr : r ≤ 1) (hmod : ‖f ξ‖ ≤ r)
    (hA : 0 ≤ A) (hP : 0 ≤ P) (hnA : (n : ℝ) * A ≤ P)
    (hD : ∀ j : ℕ, 1 ≤ j → j ≤ 3 → ‖iteratedFDeriv ℝ j f ξ‖ ≤ A)
    (m : ℕ) (hm : m ≤ n) (k : ℕ) (hk : k ≤ 3) :
    ‖iteratedFDeriv ℝ k (fun η => f η ^ m) ξ‖ ≤
      (1 + P + 4 * P ^ 2 + P ^ 3) * r ^ (m - 3) := by
  let K : ℝ := 1 + P + 4 * P ^ 2 + P ^ 3
  have hr0 : 0 ≤ r := (norm_nonneg _).trans hmod
  have hmA : (m : ℝ) * A ≤ P :=
    (mul_le_mul_of_nonneg_right (by exact_mod_cast hm) hA).trans hnA
  have hDf : ‖fderiv ℝ f ξ‖ ≤ A := by
    simpa only [norm_iteratedFDeriv_one] using hD 1 le_rfl (by omega)
  have hp (j : ℕ) (hj : j ≤ 3) : ‖f ξ‖ ^ (m - j) ≤ r ^ (m - 3) :=
    (pow_le_pow_left₀ (norm_nonneg _) hmod _).trans
      (pow_le_pow_of_le_one hr0 hr (by omega))
  have hK0 : 1 ≤ K := by dsimp [K]; nlinarith [sq_nonneg P, pow_nonneg hP 3]
  have hK1 : P ≤ K := by dsimp [K]; nlinarith [sq_nonneg P, pow_nonneg hP 3]
  have hK2 : P ^ 2 + P ≤ K := by dsimp [K]; nlinarith [sq_nonneg P, pow_nonneg hP 3]
  have hK3 : P ^ 3 + 3 * P ^ 2 + P ≤ K := by dsimp [K]; nlinarith [sq_nonneg P]
  interval_cases k
  · rw [norm_iteratedFDeriv_zero, norm_pow]
    have hb : r ^ (m - 3) ≤ K * r ^ (m - 3) := by
      simpa only [one_mul] using
        mul_le_mul_of_nonneg_right hK0 (pow_nonneg hr0 (m - 3))
    exact (hp 0 (by omega)).trans hb
  · rw [norm_iteratedFDeriv_one]
    calc
      _ ≤ (m : ℝ) * ‖f ξ‖ ^ (m - 1) * ‖fderiv ℝ f ξ‖ :=
        norm_fderiv_complex_power_le f m ξ (hf.differentiable (by norm_num) ξ)
      _ ≤ (m : ℝ) * r ^ (m - 3) * A := by gcongr; exact hp 1 (by omega)
      _ = ((m : ℝ) * A) * r ^ (m - 3) := by ring
      _ ≤ K * r ^ (m - 3) := mul_le_mul_of_nonneg_right (hmA.trans hK1) (by positivity)
  · calc
      _ ≤ (m : ℝ) ^ 2 * ‖f ξ‖ ^ (m - 2) * ‖fderiv ℝ f ξ‖ ^ 2 +
          (m : ℝ) * ‖f ξ‖ ^ (m - 1) * ‖iteratedFDeriv ℝ 2 f ξ‖ :=
        norm_iteratedFDeriv_two_complex_power_le f m (hf.of_le (by norm_num)) ξ
      _ ≤ (m : ℝ) ^ 2 * r ^ (m - 3) * A ^ 2 + (m : ℝ) * r ^ (m - 3) * A := by
        gcongr
        · exact hp 2 (by omega)
        · exact hp 1 (by omega)
        · exact hD 2 (by omega) (by omega)
      _ = (((m : ℝ) * A) ^ 2 + (m : ℝ) * A) * r ^ (m - 3) := by ring
      _ ≤ (P ^ 2 + P) * r ^ (m - 3) := by gcongr
      _ ≤ K * r ^ (m - 3) := mul_le_mul_of_nonneg_right hK2 (by positivity)
  · calc
      _ ≤ ((m : ℝ) ^ 3 * ‖fderiv ℝ f ξ‖ ^ 3 + 3 * (m : ℝ) ^ 2 *
          ‖fderiv ℝ f ξ‖ * ‖iteratedFDeriv ℝ 2 f ξ‖ +
          (m : ℝ) * ‖iteratedFDeriv ℝ 3 f ξ‖) * ‖f ξ‖ ^ (m - 3) :=
        norm_iteratedFDeriv_three_complex_power_le_of_norm_le_one f m hf ξ (hmod.trans hr)
      _ ≤ ((m : ℝ) ^ 3 * A ^ 3 + 3 * (m : ℝ) ^ 2 * A * A + (m : ℝ) * A) *
          r ^ (m - 3) := by
        gcongr
        · exact hD 2 (by omega) (by omega)
        · exact hD 3 (by omega) (by omega)
      _ = (((m : ℝ) * A) ^ 3 + 3 * ((m : ℝ) * A) ^ 2 + (m : ℝ) * A) *
          r ^ (m - 3) := by ring
      _ ≤ (P ^ 3 + 3 * P ^ 2 + P) * r ^ (m - 3) := by gcongr
      _ ≤ K * r ^ (m - 3) := mul_le_mul_of_nonneg_right hK3 (by positivity)

example : ContDiff ℝ 3 (fun _ : ℝ × ℝ => (1 : ℂ)) ∧
    ‖(1 : ℂ)‖ ≤ (1 : ℝ) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧
    (2 : ℝ) * 0 ≤ 0 ∧
    (∀ j : ℕ, 1 ≤ j → j ≤ 3 →
      ‖iteratedFDeriv ℝ j (fun _ : ℝ × ℝ => (1 : ℂ)) (0, 0)‖ ≤ 0) ∧
    (2 : ℕ) ≤ 2 ∧ (3 : ℕ) ≤ 3 := by
  refine ⟨contDiff_const, by norm_num, le_rfl, le_rfl, by norm_num, ?_, le_rfl, le_rfl⟩
  intro j hj hj3
  rw [iteratedFDeriv_const_of_ne (by omega : j ≠ 0)]
  simp

end Transformer.Modes
