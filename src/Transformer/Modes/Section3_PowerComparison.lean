import Transformer.Modes.Section3_PowerDerivatives
import Transformer.Modes.Section3_GaussianComparison
/-!
# Comparing the first two derivatives of characteristic powers

The power comparison in arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
uses both the value error and the derivative errors of one summand.
An elementary telescoping bound for complex powers retains a common
modulus bound `r`. Applying the product rule then gives explicit
first- and second-derivative comparisons before frequency scaling.

The second derivative includes the difference of products of first
derivatives as well as the difference of second derivatives. Both
terms retain their powers of `r`, which supply Gaussian damping when
the characteristic functions are bounded on a small frequency ball.
No logarithm or division by a characteristic value is used, so its
possible zeros create no extra hypothesis.

The first comparison starts at `n = 2`, and the second at `n = 3`,
so every telescoping power used below has a positive exponent.
These fixed thresholds are sufficient for the eventual estimates
of the normalized sums in the manuscript.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- Splitting a product difference retains one value error and one factor error.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the algebraic
product comparison underlying the derivative estimates. -/
theorem norm_complex_mul_sub_mul_le (a b x y : ℂ) :
    ‖a * x - b * y‖ ≤ ‖a‖ * ‖x - y‖ + ‖a - b‖ * ‖y‖ := by
  have he : a * x - b * y = a * (x - y) + (a - b) * y := by ring
  rw [he]
  exact (norm_add_le _ _).trans_eq (by rw [norm_mul, norm_mul])

/-- A common modulus bound controls a powered-factor difference.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the characteristic-power
comparison. The telescoping exponent is retained explicitly. -/
theorem norm_complex_pow_mul_sub_mul_le (z w x y : ℂ) {r : ℝ}
    (hz : ‖z‖ ≤ r) (hw : ‖w‖ ≤ r) (m : ℕ) (hm : 1 ≤ m) :
    ‖z ^ m * x - w ^ m * y‖ ≤
      r ^ m * ‖x - y‖ + (m : ℝ) * ‖z - w‖ * r ^ (m - 1) * ‖y‖ := by
  have hmPred : m - 1 + 1 = m := by omega
  have hpow := norm_complex_pow_succ_sub_le z w hz hw (m - 1)
  rw [hmPred] at hpow
  have hcast : ((m - 1 : ℕ) : ℝ) + 1 = (m : ℝ) := by exact_mod_cast hmPred
  rw [hcast] at hpow
  calc
    _ ≤ ‖z ^ m‖ * ‖x - y‖ + ‖z ^ m - w ^ m‖ * ‖y‖ :=
      norm_complex_mul_sub_mul_le _ _ _ _
    _ ≤ r ^ m * ‖x - y‖ + (m : ℝ) * ‖z - w‖ * r ^ (m - 1) * ‖y‖ := by
      rw [norm_pow]
      gcongr

example : ‖(1 : ℂ)‖ ≤ (1 : ℝ) ∧ 1 ≤ (2 : ℕ) := by norm_num

/-- First-derivative comparison for two complex powers.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the first-order
product comparison before normalized-sum frequency scaling. -/
theorem norm_fderiv_complex_power_sub_apply_le (f g : ℝ × ℝ → ℂ)
    (n : ℕ) (hn : 2 ≤ n) (ξ v : ℝ × ℝ)
    (hf : DifferentiableAt ℝ f ξ) (hg : DifferentiableAt ℝ g ξ)
    {r : ℝ} (hfr : ‖f ξ‖ ≤ r) (hgr : ‖g ξ‖ ≤ r) :
    ‖fderiv ℝ (fun η => f η ^ n) ξ v - fderiv ℝ (fun η => g η ^ n) ξ v‖ ≤
      (n : ℝ) * (r ^ (n - 1) * ‖fderiv ℝ f ξ v - fderiv ℝ g ξ v‖ +
        (n - 1 : ℕ) * ‖f ξ - g ξ‖ * r ^ (n - 2) * ‖fderiv ℝ g ξ v‖) := by
  rw [fderiv_complex_power_apply f n ξ v hf, fderiv_complex_power_apply g n ξ v hg]
  have he : (n : ℂ) * f ξ ^ (n - 1) * fderiv ℝ f ξ v -
      (n : ℂ) * g ξ ^ (n - 1) * fderiv ℝ g ξ v =
      (n : ℂ) * (f ξ ^ (n - 1) * fderiv ℝ f ξ v -
        g ξ ^ (n - 1) * fderiv ℝ g ξ v) := by ring
  rw [he, norm_mul, Complex.norm_natCast]
  have hexp : n - 1 - 1 = n - 2 := by omega
  have h := norm_complex_pow_mul_sub_mul_le (f ξ) (g ξ)
    (fderiv ℝ f ξ v) (fderiv ℝ g ξ v) hfr hgr (n - 1) (by omega)
  rw [hexp] at h
  exact mul_le_mul_of_nonneg_left h (by positivity)

example : 2 ≤ (2 : ℕ) ∧ DifferentiableAt ℝ (characteristic2 stdGauss2) 0 ∧
    ‖characteristic2 stdGauss2 0‖ ≤ 1 :=
  ⟨by omega, (contDiff_characteristic2 stdGauss2 (k := 2)
    (IsGaussian.memLp_id _ _ (by simp))).differentiable (by norm_num) 0,
    norm_characteristic2_le_one stdGauss2 0⟩

/-- Second-derivative comparison retaining both product-rule terms.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the second-order
product comparison before normalized-sum frequency scaling. -/
theorem norm_iteratedFDeriv_two_complex_power_sub_apply_le (f g : ℝ × ℝ → ℂ)
    (n : ℕ) (hn : 3 ≤ n) (hf : ContDiff ℝ 2 f) (hg : ContDiff ℝ 2 g)
    (ξ : ℝ × ℝ) (v : Fin 2 → ℝ × ℝ) {r : ℝ} (hfr : ‖f ξ‖ ≤ r) (hgr : ‖g ξ‖ ≤ r) :
    ‖iteratedFDeriv ℝ 2 (fun η => f η ^ n) ξ v -
      iteratedFDeriv ℝ 2 (fun η => g η ^ n) ξ v‖ ≤
    (n : ℝ) * (n - 1 : ℕ) *
      (r ^ (n - 2) *
        (‖fderiv ℝ f ξ (v 0)‖ * ‖fderiv ℝ f ξ (v 1) - fderiv ℝ g ξ (v 1)‖ +
          ‖fderiv ℝ f ξ (v 0) - fderiv ℝ g ξ (v 0)‖ * ‖fderiv ℝ g ξ (v 1)‖) +
        (n - 2 : ℕ) * ‖f ξ - g ξ‖ * r ^ (n - 3) *
          ‖fderiv ℝ g ξ (v 0)‖ * ‖fderiv ℝ g ξ (v 1)‖) +
    (n : ℝ) * (r ^ (n - 1) *
      ‖iteratedFDeriv ℝ 2 f ξ v - iteratedFDeriv ℝ 2 g ξ v‖ +
        (n - 1 : ℕ) * ‖f ξ - g ξ‖ * r ^ (n - 2) * ‖iteratedFDeriv ℝ 2 g ξ v‖) := by
  let F0 := fderiv ℝ f ξ (v 0)
  let F1 := fderiv ℝ f ξ (v 1)
  let G0 := fderiv ℝ g ξ (v 0)
  let G1 := fderiv ℝ g ξ (v 1)
  let F2 := iteratedFDeriv ℝ 2 f ξ v
  let G2 := iteratedFDeriv ℝ 2 g ξ v
  have hprod := norm_complex_mul_sub_mul_le F0 G0 F1 G1
  have hfirst := norm_complex_pow_mul_sub_mul_le (f ξ) (g ξ) (F0 * F1) (G0 * G1)
    hfr hgr (n - 2) (by omega)
  have hsecond := norm_complex_pow_mul_sub_mul_le (f ξ) (g ξ) F2 G2 hfr hgr (n - 1) (by omega)
  have h12 : n - 1 - 1 = n - 2 := by omega
  have h23 : n - 2 - 1 = n - 3 := by omega
  rw [h23, norm_mul] at hfirst
  rw [h12] at hsecond
  have hfirst' : ‖f ξ ^ (n - 2) * (F0 * F1) - g ξ ^ (n - 2) * (G0 * G1)‖ ≤
      r ^ (n - 2) * (‖F0‖ * ‖F1 - G1‖ + ‖F0 - G0‖ * ‖G1‖) +
        (n - 2 : ℕ) * ‖f ξ - g ξ‖ * r ^ (n - 3) * ‖G0‖ * ‖G1‖ := by
    calc
      _ ≤ r ^ (n - 2) * ‖F0 * F1 - G0 * G1‖ +
          (n - 2 : ℕ) * ‖f ξ - g ξ‖ * r ^ (n - 3) * (‖G0‖ * ‖G1‖) := hfirst
      _ ≤ r ^ (n - 2) * (‖F0‖ * ‖F1 - G1‖ + ‖F0 - G0‖ * ‖G1‖) +
          (n - 2 : ℕ) * ‖f ξ - g ξ‖ * r ^ (n - 3) * (‖G0‖ * ‖G1‖) := by
        have hr : 0 ≤ r := (norm_nonneg (f ξ)).trans hfr
        gcongr
      _ = _ := by ring
  rw [iteratedFDeriv_two_complex_power_apply f n hf, iteratedFDeriv_two_complex_power_apply g n hg]
  have he : (n : ℂ) * (n - 1 : ℕ) * f ξ ^ (n - 2) * F0 * F1 + (n : ℂ) * f ξ ^ (n - 1) * F2 -
      ((n : ℂ) * (n - 1 : ℕ) * g ξ ^ (n - 2) * G0 * G1 + (n : ℂ) * g ξ ^ (n - 1) * G2) =
      ((n : ℂ) * (n - 1 : ℕ)) *
        (f ξ ^ (n - 2) * (F0 * F1) - g ξ ^ (n - 2) * (G0 * G1)) +
      (n : ℂ) * (f ξ ^ (n - 1) * F2 - g ξ ^ (n - 1) * G2) := by ring
  rw [he]
  calc
    _ ≤ ‖((n : ℂ) * (n - 1 : ℕ)) *
        (f ξ ^ (n - 2) * (F0 * F1) - g ξ ^ (n - 2) * (G0 * G1))‖ +
        ‖(n : ℂ) * (f ξ ^ (n - 1) * F2 - g ξ ^ (n - 1) * G2)‖ := norm_add_le _ _
    _ ≤ _ := by
      simp only [norm_mul, Complex.norm_natCast]
      exact add_le_add (mul_le_mul_of_nonneg_left hfirst' (by positivity))
        (mul_le_mul_of_nonneg_left hsecond (by positivity))

example : 3 ≤ (3 : ℕ) ∧ ContDiff ℝ 2 (characteristic2 stdGauss2) ∧
    ‖characteristic2 stdGauss2 0‖ ≤ 1 :=
  ⟨by omega, contDiff_characteristic2 stdGauss2 (k := 2)
    (IsGaussian.memLp_id _ _ (by simp)), norm_characteristic2_le_one stdGauss2 0⟩

end Transformer.Modes
