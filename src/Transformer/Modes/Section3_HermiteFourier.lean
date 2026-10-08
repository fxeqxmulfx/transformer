import Transformer.Modes.Section3_ScalarGaussianFourier
import Mathlib.MeasureTheory.Integral.Prod

/-!
# The actual two-dimensional Gaussian Hermite transform

The four third-order basis functions in arXiv:2412.09080v3, §3.1
`eq:psi`, factor into scalar Hermite-weighted Gaussians. Their
integrability is obtained from the actual scalar polynomial moments,
and the characteristic phase does not change their norm.

The product integral gives the exact positive-sign transform. Each
coordinate contributes `(i ξ_j)^k`; their total degree is three, so
the phase factor is `i³ = -i`. The two factors `sqrt(2π)` cancel the
literal density normalization `(2π)⁻¹`. No normalization is suppressed.

This corrects the §5.4 display before `eq:higher-error-goal`, which
writes a Hermite polynomial in frequency in place of the cubic
monomial multiplier. The functions transformed here are precisely
the spatial Gaussian Hermite basis of §3.1, without altering `ψ`.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- The literal two-dimensional density factors into its coordinate Gaussians.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem phi2_eq_scalar_gaussian_product (x : ℝ × ℝ) :
    phi2 x = (2 * Real.pi)⁻¹ * Real.exp (-(x.1 ^ 2 / 2)) * Real.exp (-(x.2 ^ 2 / 2)) := by
  unfold phi2
  rw [mul_assoc, ← Real.exp_add]
  congr 2
  ring

/-- The positive-sign phase gives the exact scalar Hermite monomial multiplier.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integral_hermite1_complex_gaussian_phase (k : ℕ) (hk : k ≤ 3) (t : ℝ) :
    (∫ x : ℝ, Complex.exp (Complex.I * (t : ℂ) * (x : ℂ)) *
      ((hermite1 k x : ℝ) : ℂ) * ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ)) =
      (Complex.I * (t : ℂ)) ^ k * ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) *
        ((Real.exp (-(t ^ 2 / 2)) : ℝ) : ℂ) := by
  have ht : 2 * Real.pi * (-t / (2 * Real.pi)) = -t := by field_simp
  have hc : -2 * (Real.pi : ℂ) * Complex.I * ((-t / (2 * Real.pi) : ℝ) : ℂ) =
      Complex.I * (t : ℂ) := by
    push_cast
    field_simp
  have h := fourier_hermite1_mul_complex_scalar_gaussian k hk (-t / (2 * Real.pi))
  rw [Real.fourier_real_eq_integral_exp_smul, ht, hc, neg_sq] at h
  convert h using 1
  congr 1
  funext x
  rw [smul_eq_mul]
  have hp : Complex.exp (Complex.I * (t : ℂ) * (x : ℂ)) =
      Complex.exp (((-2 * Real.pi * x * (-t / (2 * Real.pi)) : ℝ) : ℂ) * Complex.I) := by
    congr 1
    push_cast
    field_simp
  rw [hp]
  ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The actual third-order two-dimensional Hermite-weighted density is integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_hermite3_mul_phi2 (k : ℕ) (hk : k ≤ 3) :
    Integrable (fun x : ℝ × ℝ => ((phi2 x * hermite3 k x : ℝ) : ℂ)) := by
  have h := (integrable_hermite1_mul_complex_scalar_gaussian k hk).mul_prod
    (integrable_hermite1_mul_complex_scalar_gaussian (3 - k) (by omega))
  convert h.const_mul (((2 * Real.pi)⁻¹ : ℝ) : ℂ) using 1
  funext x
  rw [phi2_eq_scalar_gaussian_product]
  dsimp [hermite3]
  simp only [Complex.ofReal_mul]
  ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The actual characteristic phase preserves Hermite-weighted density integrability.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_hermite3_phi2_characteristic (k : ℕ) (hk : k ≤ 3) (ξ : ℝ × ℝ) :
    Integrable (fun x : ℝ × ℝ => Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ)) *
      ((phi2 x * hermite3 k x : ℝ) : ℂ)) := by
  have hc : Continuous (fun x : ℝ × ℝ =>
      Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ))) := by fun_prop
  exact (integrable_hermite3_mul_phi2 k hk).bdd_mul hc.aestronglyMeasurable
    (ae_of_all _ fun x => by
      rw [mul_comm Complex.I, Complex.norm_exp_ofReal_mul_I])

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The actual third-order Gaussian Hermite transform has the multiplier `-i ξ₁^k ξ₂^(3-k)`.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integral_hermite3_phi2_characteristic (k : ℕ) (hk : k ≤ 3) (ξ : ℝ × ℝ) :
    (∫ x : ℝ × ℝ, Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ)) *
      ((phi2 x * hermite3 k x : ℝ) : ℂ)) =
        -Complex.I * (ξ.1 : ℂ) ^ k * (ξ.2 : ℂ) ^ (3 - k) *
          ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) := by
  let f : ℝ → ℂ := fun u => Complex.exp (Complex.I * (ξ.1 : ℂ) * (u : ℂ)) *
    ((hermite1 k u : ℝ) : ℂ) * ((Real.exp (-(u ^ 2 / 2)) : ℝ) : ℂ)
  let g : ℝ → ℂ := fun v => Complex.exp (Complex.I * (ξ.2 : ℂ) * (v : ℂ)) *
    ((hermite1 (3 - k) v : ℝ) : ℂ) * ((Real.exp (-(v ^ 2 / 2)) : ℝ) : ℂ)
  have he (x : ℝ × ℝ) : Complex.exp (Complex.I * ((ξ.1 * x.1 + ξ.2 * x.2 : ℝ) : ℂ)) *
      ((phi2 x * hermite3 k x : ℝ) : ℂ) =
        (((2 * Real.pi)⁻¹ : ℝ) : ℂ) * (f x.1 * g x.2) := by
    rw [phi2_eq_scalar_gaussian_product]
    dsimp [f, g, hermite3]
    simp only [Complex.ofReal_mul, Complex.ofReal_add]
    rw [show Complex.I * ((ξ.1 : ℂ) * x.1 + (ξ.2 : ℂ) * x.2) =
      Complex.I * (ξ.1 : ℂ) * x.1 + Complex.I * (ξ.2 : ℂ) * x.2 by ring,
      Complex.exp_add]
    ring
  simp_rw [he]
  rw [integral_const_mul]
  change (((2 * Real.pi)⁻¹ : ℝ) : ℂ) *
    (∫ a : ℝ × ℝ, f a.1 * g a.2 ∂(volume.prod volume)) = _
  rw [integral_prod_mul]
  dsimp only [f, g]
  rw [integral_hermite1_complex_gaussian_phase k hk,
    integral_hermite1_complex_gaussian_phase (3 - k) (by omega)]
  have hI : Complex.I ^ k * Complex.I ^ (3 - k) = -Complex.I := by
    rw [← pow_add, show k + (3 - k) = 3 by omega]
    norm_num [pow_succ, Complex.I_sq]
  have hs : ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) * ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) =
      ((2 * Real.pi : ℝ) : ℂ) := by
    rw [← Complex.ofReal_mul, Real.mul_self_sqrt (by positivity)]
  have heG : ((Real.exp (-(ξ.1 ^ 2 / 2)) : ℝ) : ℂ) *
      ((Real.exp (-(ξ.2 ^ 2 / 2)) : ℝ) : ℂ) =
        ((Real.exp (-(ξ.1 ^ 2 + ξ.2 ^ 2) / 2) : ℝ) : ℂ) := by
    rw [← Complex.ofReal_mul, ← Real.exp_add]
    congr 2
    ring
  simp only [mul_pow]
  calc
    _ = ((((2 * Real.pi)⁻¹ : ℝ) : ℂ) *
        (((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) * ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ))) *
      (Complex.I ^ k * Complex.I ^ (3 - k)) * (ξ.1 : ℂ) ^ k * (ξ.2 : ℂ) ^ (3 - k) *
      (((Real.exp (-(ξ.1 ^ 2 / 2)) : ℝ) : ℂ) * ((Real.exp (-(ξ.2 ^ 2 / 2)) : ℝ) : ℂ)) := by ring
    _ = _ := by rw [hI, hs, heG]; simp only [← Complex.ofReal_mul, inv_mul_cancel₀ (by positivity : (2 * Real.pi : ℝ) ≠ 0), Complex.ofReal_one, one_mul]

example : (3 : ℕ) ≤ 3 := le_rfl

/-- Every recorded third-order Hermite correction has zero total integral.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, the centered Gaussian Hermite basis. -/
theorem integral_hermite3_mul_phi2_zero (k : ℕ) (hk : k ≤ 3) :
    (∫ x : ℝ × ℝ, ((phi2 x * hermite3 k x : ℝ) : ℂ)) = 0 := by
  have h := integral_hermite3_phi2_characteristic k hk (0, 0)
  interval_cases k <;> simpa using h

example : (3 : ℕ) ≤ 3 := le_rfl

end Transformer.Modes
