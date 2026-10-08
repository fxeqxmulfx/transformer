import Transformer.Modes.Section3_CubicGaussianComparison
import Transformer.Modes.Section3_GaussianComparison
/-!
# Retaining the cubic correction under characteristic-function powers

The normalized-sum step of arXiv:2412.09080v3, §5.4 `eq:br-9.10`,
requires controlling powers while preserving the first correction.
For two complex numbers bounded in norm by `r`, subtracting the linear
term from `z^n - w^n` leaves a remainder bounded by
`n² |z-w|² r^(n-2)`. An induction proves this with the common damping
factor retained, including the endpoint `n = 2`.

With a proposed relative correction `b`, the error also contains
`n |z-w-w b| r^(n-1)`. Applying both bounds to the actual one-summand
characteristic functions gives fourth- and sixth-degree frequency
terms. Their normalized scales will both be `1 / n`; the frequency
rescaling and Gaussian damping are proved in the subsequent module.

Centering and identity covariance make the first difference cubic
in the frequency. Finite fourth moment makes its correction error
fourth order on the unit ball. Squaring the first difference produces
the sixth-degree term in the power estimate. Exponent monotonicity
supplies the third moment from the fourth moment, so this step uses
exactly the finite-moment input of the local expansion.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- A power's remainder after its actual linear term is quadratically small.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, retaining the cubic term
in the normalized-sum power. The explicit coefficient `n²` is nonsharp. -/
theorem norm_complex_pow_sub_linear_le (z w : ℂ) {r : ℝ} (hz : ‖z‖ ≤ r) (hw : ‖w‖ ≤ r)
    {n : ℕ} (hn : 2 ≤ n) :
    ‖z ^ n - w ^ n - (n : ℂ) * w ^ (n - 1) * (z - w)‖ ≤
      (n : ℝ) ^ 2 * ‖z - w‖ ^ 2 * r ^ (n - 2) := by
  have hr : 0 ≤ r := (norm_nonneg z).trans hz
  induction n, hn using Nat.le_induction with
  | base =>
    have he : z ^ 2 - w ^ 2 - ((2 : ℕ) : ℂ) * w ^ (2 - 1) * (z - w) = (z - w) ^ 2 := by
      norm_num
      ring
    rw [he, norm_pow]
    norm_num
    nlinarith [sq_nonneg ‖z - w‖]
  | succ n hn ih =>
    have hn1 : n - 1 + 1 = n := by omega
    have hn2 : n - 2 + 1 = n - 1 := by omega
    have hwP : w ^ (n - 1) * w = w ^ n := by rw [← pow_succ, hn1]
    have hrP : r ^ (n - 2) * r = r ^ (n - 1) := by rw [← pow_succ, hn2]
    have he : z ^ (n + 1) - w ^ (n + 1) - ((n + 1 : ℕ) : ℂ) * w ^ (n + 1 - 1) * (z - w) =
        w * (z ^ n - w ^ n - (n : ℂ) * w ^ (n - 1) * (z - w)) + (z - w) * (z ^ n - w ^ n) := by
      rw [show n + 1 - 1 = n by omega, pow_succ z n, pow_succ w n]
      push_cast
      rw [← hwP]
      ring
    have hpow := norm_complex_pow_succ_sub_le z w hz hw (n - 1)
    rw [hn1] at hpow
    have hnCast : ((n - 1 : ℕ) : ℝ) + 1 = (n : ℝ) := by exact_mod_cast hn1
    rw [hnCast] at hpow
    rw [he]
    calc
      _ ≤ ‖w * (z ^ n - w ^ n - (n : ℂ) * w ^ (n - 1) * (z - w))‖ +
          ‖(z - w) * (z ^ n - w ^ n)‖ := norm_add_le _ _
      _ = ‖w‖ * ‖z ^ n - w ^ n - (n : ℂ) * w ^ (n - 1) * (z - w)‖ +
          ‖z - w‖ * ‖z ^ n - w ^ n‖ := by rw [norm_mul, norm_mul]
      _ ≤ r * ((n : ℝ) ^ 2 * ‖z - w‖ ^ 2 * r ^ (n - 2)) +
          ‖z - w‖ * ((n : ℝ) * ‖z - w‖ * r ^ (n - 1)) := by gcongr
      _ = ((n : ℝ) ^ 2 + n) * ‖z - w‖ ^ 2 * r ^ (n - 1) := by rw [← hrP]; ring
      _ ≤ ((n + 1 : ℕ) : ℝ) ^ 2 * ‖z - w‖ ^ 2 * r ^ (n + 1 - 2) := by
        rw [show n + 1 - 2 = n - 1 by omega]
        push_cast
        gcongr
        nlinarith [show 0 ≤ (n : ℝ) by positivity]


example : ‖(1 : ℂ)‖ ≤ (1 : ℝ) ∧ ‖(1 / 2 : ℂ)‖ ≤ (1 : ℝ) ∧ 2 ≤ (2 : ℕ) := by
  norm_num

/-- A relative correction in a power is controlled by two literal errors.
Source: arXiv:2412.09080v3, §5.4 `eq:br-9.10`, the first correction
under the characteristic-function power, with a common modulus bound. -/
theorem norm_complex_pow_sub_correction_le (z w b : ℂ) {r : ℝ}
    (hz : ‖z‖ ≤ r) (hw : ‖w‖ ≤ r) {n : ℕ} (hn : 2 ≤ n) :
    ‖z ^ n - w ^ n - (n : ℂ) * b * w ^ n‖ ≤
      (n : ℝ) ^ 2 * ‖z - w‖ ^ 2 * r ^ (n - 2) +
        (n : ℝ) * ‖z - w - w * b‖ * r ^ (n - 1) := by
  have hr : 0 ≤ r := (norm_nonneg z).trans hz
  have hwP : w ^ (n - 1) * w = w ^ n := pow_sub_one_mul (by omega) w
  have he : z ^ n - w ^ n - (n : ℂ) * b * w ^ n =
      (z ^ n - w ^ n - (n : ℂ) * w ^ (n - 1) * (z - w)) +
        (n : ℂ) * w ^ (n - 1) * (z - w - w * b) := by
    rw [← hwP]
    ring
  rw [he]
  calc
    _ ≤ ‖z ^ n - w ^ n - (n : ℂ) * w ^ (n - 1) * (z - w)‖ +
        ‖(n : ℂ) * w ^ (n - 1) * (z - w - w * b)‖ := norm_add_le _ _
    _ ≤ (n : ℝ) ^ 2 * ‖z - w‖ ^ 2 * r ^ (n - 2) +
        (n : ℝ) * ‖z - w - w * b‖ * r ^ (n - 1) := by
      apply add_le_add (norm_complex_pow_sub_linear_le z w hz hw hn)
      rw [norm_mul, norm_mul, Complex.norm_natCast, norm_pow]
      calc
        _ ≤ (n : ℝ) * r ^ (n - 1) * ‖z - w - w * b‖ := by gcongr
        _ = _ := by ring

example : ‖(1 : ℂ)‖ ≤ (1 : ℝ) ∧ ‖(1 : ℂ)‖ ≤ (1 : ℝ) ∧ 2 ≤ (2 : ℕ) := by
  norm_num

/-- The actual characteristic power retains the Gaussian-damped cubic correction.
Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`, and §5.4 `eq:br-9.10`.
This estimate precedes frequency normalization and uses a supplied common
modulus bound; the small-frequency theorem supplies that bound later. -/
theorem norm_characteristic2_pow_sub_gaussian_cubic_le
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (hμ : IsStandardized μ)
    (hmom : MemLp id 4 μ) (ξ : ℝ × ℝ) (hξ : ‖ξ‖ ≤ 1) {r : ℝ}
    (hχ : ‖characteristic2 μ ξ‖ ≤ r) (hG : ‖characteristic2 stdGauss2 ξ‖ ≤ r)
    {n : ℕ} (hn : 2 ≤ n) :
    ‖characteristic2 μ ξ ^ n - characteristic2 stdGauss2 ξ ^ n -
      (n : ℂ) * cubicCharacteristicCorrection μ ξ * characteristic2 stdGauss2 ξ ^ n‖ ≤
      (n : ℝ) ^ 2 * (4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1) ^ 2 * ‖ξ‖ ^ 6 * r ^ (n - 2) +
        (n : ℝ) * ((8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) + 1 +
          (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)) * ‖ξ‖ ^ 4 * r ^ (n - 1) := by
  have hM3 : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 3 ∂μ := integral_nonneg fun z => by positivity
  have hM4 : 0 ≤ ∫ z : ℝ × ℝ, ‖z‖ ^ 4 ∂μ := integral_nonneg fun z => by positivity
  have hD : ‖characteristic2 μ ξ - characteristic2 stdGauss2 ξ‖ ≤
      (4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1) * ‖ξ‖ ^ 3 := by
    rw [characteristic2_stdGauss2]
    exact norm_characteristic2_sub_gaussian μ hμ (hmom.mono_exponent (by norm_num)) ξ hξ
  have hB := norm_characteristic2_sub_gaussian_mul_cubicCorrection μ hμ hmom ξ hξ
  have hpow := norm_complex_pow_sub_correction_le (characteristic2 μ ξ)
    (characteristic2 stdGauss2 ξ) (cubicCharacteristicCorrection μ ξ) hχ hG hn
  have hr : 0 ≤ r := (norm_nonneg _).trans hχ
  calc
    _ ≤ (n : ℝ) ^ 2 * ‖characteristic2 μ ξ - characteristic2 stdGauss2 ξ‖ ^ 2 * r ^ (n - 2) +
        (n : ℝ) * ‖characteristic2 μ ξ - characteristic2 stdGauss2 ξ -
          characteristic2 stdGauss2 ξ * cubicCharacteristicCorrection μ ξ‖ * r ^ (n - 1) := hpow
    _ ≤ (n : ℝ) ^ 2 * ((4 * (∫ z, ‖z‖ ^ 3 ∂μ) + 1) * ‖ξ‖ ^ 3) ^ 2 * r ^ (n - 2) +
        (n : ℝ) * (((8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 4 ∂μ) + 1 +
          (8 / 3 : ℝ) * (∫ z, ‖z‖ ^ 3 ∂μ)) * ‖ξ‖ ^ 4) * r ^ (n - 1) := by gcongr
    _ = _ := by ring

example : IsProbabilityMeasure stdGauss2 ∧ IsStandardized stdGauss2 ∧
    MemLp id 4 stdGauss2 ∧ ‖((1, 0) : ℝ × ℝ)‖ ≤ 1 ∧
    ‖characteristic2 stdGauss2 (1, 0)‖ ≤ 1 ∧
    ‖characteristic2 stdGauss2 (1, 0)‖ ≤ 1 ∧ 2 ≤ (2 : ℕ) :=
  ⟨inferInstance, isStandardized_stdGauss2, IsGaussian.memLp_id _ _ (by simp),
    by norm_num, norm_characteristic2_le_one _ _, norm_characteristic2_le_one _ _, by omega⟩

end Transformer.Modes
