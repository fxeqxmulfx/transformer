import Transformer.Modes.Section3_GaussianHermite
import Mathlib.Analysis.SpecialFunctions.Gaussian.FourierTransform
import Mathlib.Analysis.Fourier.FourierTransformDeriv

/-!
# Fourier transform of the actual low-order Gaussian Hermite basis

Section 3.1 of arXiv:2412.09080v3 defines `ψ` with Gaussian-weighted
Hermite polynomials. The scalar Gaussian transform is computed from the
proved Gaussian integral, retaining Mathlib's `2π` frequency convention.
Every polynomial Gaussian moment and every derivative used below is
proved integrable, so the derivative transform formula applies to the
actual functions through order three.

The resulting multiplier is `(-2π i t)^k`. In particular the §5.4 display
before `eq:higher-error-goal`, which writes Hermite polynomials in the
frequency correction, needs this monomial multiplier instead. The
spatial correction in §3.1 `eq:psi` is retained exactly.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology FourierTransform
namespace Transformer.Modes

/-- The actual scalar Gaussian phase integral includes its exact normalization.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integral_complex_scalar_gaussian_phase (t : ℝ) :
    (∫ x : ℝ, Complex.exp (-Complex.I * (t : ℂ) * (x : ℂ)) *
      ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ)) =
        ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) * ((Real.exp (-(t ^ 2 / 2)) : ℝ) : ℂ) := by
  have h := fourierIntegral_gaussian (b := (1 / 2 : ℂ)) (by norm_num) (-(t : ℂ))
  have hpow : ((Real.pi : ℂ) / (1 / 2 : ℂ)) ^ (1 / 2 : ℂ) =
      ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) := by
    calc
      _ = (((2 * Real.pi : ℝ) : ℂ)) ^ (((1 / 2 : ℝ) : ℂ)) := by congr 1 <;> push_cast <;> ring
      _ = (((2 * Real.pi : ℝ) ^ (1 / 2 : ℝ) : ℝ) : ℂ) :=
        (Complex.ofReal_cpow (by positivity : 0 ≤ 2 * Real.pi) (1 / 2)).symm
      _ = _ := by rw [← Real.sqrt_eq_rpow]
  rw [hpow] at h
  convert h using 1
  · congr 1
    funext x
    rw [Complex.ofReal_exp]
    congr 2 <;> push_cast <;> ring
  · rw [Complex.ofReal_exp]
    congr 2
    push_cast
    ring

/-- The actual scalar Gaussian transform has the precise `2π` frequency scale.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem fourier_complex_scalar_gaussian (t : ℝ) :
    (𝓕 (fun x : ℝ => ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ))) t =
      ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) *
        ((Real.exp (-((2 * Real.pi * t) ^ 2 / 2)) : ℝ) : ℂ) := by
  rw [Real.fourier_real_eq_integral_exp_smul]
  convert integral_complex_scalar_gaussian_phase (2 * Real.pi * t) using 1
  congr 1
  funext x
  rw [smul_eq_mul]
  congr 2
  push_cast
  ring

/-- Every scalar polynomial-weighted Gaussian is integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_pow_mul_complex_scalar_gaussian (k : ℕ) :
    Integrable (fun x : ℝ => ((x ^ k : ℝ) : ℂ) * ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ)) := by
  have hR := integrable_rpow_mul_exp_neg_mul_sq (b := 1 / 2) (by norm_num)
    (s := (k : ℝ)) (by linarith [show 0 ≤ (k : ℝ) by positivity])
  have h : Integrable (fun x : ℝ =>
      ((x ^ (k : ℝ) * Real.exp (-(1 / 2 : ℝ) * x ^ 2) : ℝ) : ℂ)) := hR.ofReal
  convert h using 1
  funext x
  rw [Real.rpow_natCast]
  push_cast
  congr 2
  ring


/-- The recorded low-order Hermite-weighted Gaussians are integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_hermite1_mul_complex_scalar_gaussian (k : ℕ) (hk : k ≤ 3) :
    Integrable (fun x : ℝ => ((hermite1 k x : ℝ) : ℂ) * ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ)) := by
  have h0 := integrable_pow_mul_complex_scalar_gaussian 0
  have h1 := integrable_pow_mul_complex_scalar_gaussian 1
  have h2 := integrable_pow_mul_complex_scalar_gaussian 2
  have h3 := integrable_pow_mul_complex_scalar_gaussian 3
  interval_cases k
  · simpa only [hermite1, pow_zero, Complex.ofReal_one, one_mul] using h0
  · simpa only [hermite1, pow_one] using h1
  · convert h2.sub h0 using 1
    funext x
    simp only [Pi.sub_apply, hermite1, pow_zero, Complex.ofReal_one, one_mul]
    push_cast
    ring
  · convert h3.sub (h1.const_mul (3 : ℂ)) using 1
    funext x
    simp only [Pi.sub_apply, hermite1, pow_one]
    push_cast
    ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The actual low-order Gaussian derivatives are integrable.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem integrable_iteratedDeriv_complex_scalar_gaussian (k : ℕ) (hk : k ≤ 3) :
    Integrable (iteratedDeriv k (fun x : ℝ => ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ))) := by
  convert (integrable_hermite1_mul_complex_scalar_gaussian k hk).const_mul ((-1 : ℂ) ^ k) using 1
  funext x
  rw [iteratedDeriv_complex_scalar_gaussian_eq_hermite1 k hk]
  push_cast
  ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The Hermite-weighted Gaussian transforms to a monomial-weighted Gaussian.
Source: arXiv:2412.09080v3, §3.1 `eq:psi` and §5.4 `eq:higher-error-goal`. -/
theorem fourier_hermite1_mul_complex_scalar_gaussian (k : ℕ) (hk : k ≤ 3) (t : ℝ) :
    (𝓕 (fun x : ℝ => ((hermite1 k x : ℝ) : ℂ) * ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ))) t =
      (-2 * (Real.pi : ℂ) * Complex.I * (t : ℂ)) ^ k *
        ((Real.sqrt (2 * Real.pi) : ℝ) : ℂ) *
          ((Real.exp (-((2 * Real.pi * t) ^ 2 / 2)) : ℝ) : ℂ) := by
  let f : ℝ → ℂ := fun x => ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ)
  have hf : ContDiff ℝ 3 f := Complex.ofRealCLM.contDiff.comp
    (by fun_prop : ContDiff ℝ 3 (fun x : ℝ => Real.exp (-(x ^ 2 / 2))))
  have hI (j : ℕ) (hj : (j : ℕ∞) ≤ 3) : Integrable (iteratedDeriv j f) :=
    integrable_iteratedDeriv_complex_scalar_gaussian j (by exact_mod_cast hj)
  have hp : (-1 : ℂ) ^ k * (-1 : ℂ) ^ k = 1 := by rw [← mul_pow]; norm_num
  have he : (fun x : ℝ => ((hermite1 k x : ℝ) : ℂ) * ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ)) =
      ((-1 : ℂ) ^ k) • iteratedDeriv k f := by
    funext x
    simp only [Pi.smul_apply, smul_eq_mul]
    dsimp only [f]
    rw [iteratedDeriv_complex_scalar_gaussian_eq_hermite1 k hk]
    simp only [Complex.ofReal_mul, Complex.ofReal_pow, Complex.ofReal_neg, Complex.ofReal_one]
    calc
      _ = ((-1 : ℂ) ^ k * (-1 : ℂ) ^ k) *
          (((hermite1 k x : ℝ) : ℂ) * ((Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ)) := by rw [hp]; ring
      _ = _ := by ring
  have hFT (c : ℂ) (g : ℝ → ℂ) : (𝓕 (c • g)) t = c * (𝓕 g) t := by
    rw [Real.fourier_real_eq_integral_exp_smul, Real.fourier_real_eq_integral_exp_smul,
      ← integral_const_mul]
    simp only [Pi.smul_apply, smul_eq_mul]
    congr 1
    funext x
    ring
  rw [he, hFT]
  rw [Real.fourier_iteratedDeriv hf hI (by exact_mod_cast hk)]
  change (-1 : ℂ) ^ k * ((2 * (Real.pi : ℂ) * Complex.I * (t : ℂ)) ^ k * (𝓕 f) t) = _
  dsimp only [f]
  rw [fourier_complex_scalar_gaussian]
  have hpow : (-2 * (Real.pi : ℂ) * Complex.I * (t : ℂ)) ^ k =
      (-1 : ℂ) ^ k * (2 * (Real.pi : ℂ) * Complex.I * (t : ℂ)) ^ k := by
    rw [← mul_pow]
    congr 1
    ring
  rw [hpow]
  ring

example : (3 : ℕ) ≤ 3 := le_rfl

end Transformer.Modes
