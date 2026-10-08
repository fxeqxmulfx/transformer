import Transformer.Modes.Section3_Hermite
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.RingTheory.Polynomial.Hermite.Gaussian
/-!
# Actual Gaussian derivatives and the third Hermite basis

Section 3.1 of arXiv:2412.09080v3 defines the multivariate Hermite
polynomials by `H^α = (-1)^|α| φ⁻¹ ∂^α φ`. The project's literal
polynomials of degrees zero through three are identified here with
Mathlib's probabilists' Hermite polynomials, whose Gaussian derivative
formula is already proved. This supplies actual derivatives, with every
sign retained, for the Gaussian and its complex-valued frequency form.

For the two-dimensional density, factoring the literal Gaussian into
its coordinate functions gives each partial derivative. Their mixed
third derivative is exactly `-H^(k,3-k) φ`, and positivity of `φ` then
gives the source's defining identity. These identities justify using
the recorded Hermite basis in the Fourier proof of the actual `ψ` term.
The complex scaling result includes arbitrary real scales, as needed
when translating the library's Fourier phase convention to the paper's.
-/

open Real MeasureTheory ProbabilityTheory Filter
open scoped ENNReal Topology
namespace Transformer.Modes

/-- The literal low-order polynomials agree with the probabilists' Hermite polynomials.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem hermite1_eq_aeval (k : ℕ) (hk : k ≤ 3) (x : ℝ) :
    hermite1 k x = Polynomial.aeval x (Polynomial.hermite k) := by
  interval_cases k <;> norm_num [hermite1, Polynomial.hermite] <;> ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The actual scalar Gaussian derivatives have the recorded Hermite coefficients.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem iteratedDeriv_scalar_gaussian_eq_hermite1 (k : ℕ) (hk : k ≤ 3) (x : ℝ) :
    iteratedDeriv k (fun y : ℝ => Real.exp (-(y ^ 2 / 2))) x =
      (-1 : ℝ) ^ k * hermite1 k x * Real.exp (-(x ^ 2 / 2)) := by
  rw [iteratedDeriv_eq_iterate, hermite1_eq_aeval k hk]
  exact Polynomial.deriv_gaussian_eq_hermite_mul_gaussian k x

example : (3 : ℕ) ≤ 3 := le_rfl

/-- A smooth real function's actual iterated derivatives commute with complex inclusion.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem iteratedDeriv_complex_ofReal (f : ℝ → ℝ) (hf : ContDiff ℝ ⊤ f) (k : ℕ) (x : ℝ) :
    iteratedDeriv k (fun y : ℝ => ((f y : ℝ) : ℂ)) x = ((iteratedDeriv k f x : ℝ) : ℂ) := by
  have he : (fun y : ℝ => ((f y : ℝ) : ℂ)) = Complex.ofRealCLM ∘ f := rfl
  rw [iteratedDeriv_eq_iteratedFDeriv, he,
    Complex.ofRealCLM.iteratedFDeriv_comp_left hf.contDiffAt (by exact le_top)]
  rfl

example : ContDiff ℝ ⊤ (fun x : ℝ => x) := by fun_prop

/-- The complex-valued actual Gaussian derivatives retain the Hermite formula.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem iteratedDeriv_complex_scalar_gaussian_eq_hermite1 (k : ℕ) (hk : k ≤ 3) (x : ℝ) :
    iteratedDeriv k (fun y : ℝ => ((Real.exp (-(y ^ 2 / 2)) : ℝ) : ℂ)) x =
      (( (-1 : ℝ) ^ k * hermite1 k x * Real.exp (-(x ^ 2 / 2)) : ℝ) : ℂ) := by
  rw [iteratedDeriv_complex_ofReal _ (by fun_prop), iteratedDeriv_scalar_gaussian_eq_hermite1 k hk]



example : (3 : ℕ) ≤ 3 := le_rfl

/-- A real frequency scale gives the actual Gaussian derivative factor `(-c)^k`.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem iteratedDeriv_complex_scaled_gaussian_eq_hermite1 (k : ℕ) (hk : k ≤ 3) (c x : ℝ) :
    iteratedDeriv k (fun y : ℝ => ((Real.exp (-((c * y) ^ 2 / 2)) : ℝ) : ℂ)) x =
      (((-c) ^ k * hermite1 k (c * x) * Real.exp (-((c * x) ^ 2 / 2)) : ℝ) : ℂ) := by
  have hC : ContDiff ℝ k (fun y : ℝ => ((Real.exp (-(y ^ 2 / 2)) : ℝ) : ℂ)) :=
    Complex.ofRealCLM.contDiff.comp
      (by fun_prop : ContDiff ℝ k (fun y : ℝ => Real.exp (-(y ^ 2 / 2))))
  rw [iteratedDeriv_comp_const_smul hC c]
  change c ^ k • iteratedDeriv k (fun y : ℝ => ((Real.exp (-(y ^ 2 / 2)) : ℝ) : ℂ)) (c * x) = _
  rw [iteratedDeriv_complex_scalar_gaussian_eq_hermite1 k hk, Complex.real_smul, ← Complex.ofReal_mul]
  congr 1
  have hp : (-c) ^ k = (-1 : ℝ) ^ k * c ^ k := by
    rw [show -c = (-1 : ℝ) * c by ring, mul_pow]
  rw [hp]
  ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- First-coordinate derivatives of the actual two-dimensional density have the Hermite factor.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem iteratedDeriv_phi2_fst_eq_hermite1 (k : ℕ) (hk : k ≤ 3) (x z : ℝ) :
    iteratedDeriv k (fun y : ℝ => phi2 (y, z)) x =
      (-1 : ℝ) ^ k * hermite1 k x * phi2 (x, z) := by
  let C : ℝ := (2 * Real.pi)⁻¹ * Real.exp (-(z ^ 2 / 2))
  have he (y : ℝ) : phi2 (y, z) = C * Real.exp (-(y ^ 2 / 2)) := by
    dsimp [phi2, C]
    rw [mul_assoc, ← Real.exp_add]
    congr 2
    ring
  have hf : (fun y : ℝ => phi2 (y, z)) = fun y => C * Real.exp (-(y ^ 2 / 2)) := funext he
  rw [hf, iteratedDeriv_const_mul_field, iteratedDeriv_scalar_gaussian_eq_hermite1 k hk, he x]
  ring

example : (3 : ℕ) ≤ 3 := le_rfl

/-- Second-coordinate derivatives of the actual two-dimensional density have the Hermite factor.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem iteratedDeriv_phi2_snd_eq_hermite1 (k : ℕ) (hk : k ≤ 3) (x z : ℝ) :
    iteratedDeriv k (fun y : ℝ => phi2 (z, y)) x =
      (-1 : ℝ) ^ k * hermite1 k x * phi2 (z, x) := by
  have he (y : ℝ) : phi2 (z, y) = phi2 (y, z) := by
    dsimp [phi2]
    congr 2
    ring
  rw [funext he, iteratedDeriv_phi2_fst_eq_hermite1 k hk, he x]

example : (3 : ℕ) ≤ 3 := le_rfl

/-- The actual mixed third Gaussian derivative is the negative Hermite-weighted density.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem mixed_iteratedDeriv_phi2_eq_hermite3 (k : ℕ) (hk : k ≤ 3) (x : ℝ × ℝ) :
    iteratedDeriv k (fun u : ℝ => iteratedDeriv (3 - k) (fun v : ℝ => phi2 (u, v)) x.2) x.1 =
      -hermite3 k x * phi2 x := by
  have he : (fun u : ℝ => iteratedDeriv (3 - k) (fun v : ℝ => phi2 (u, v)) x.2) =
      fun u => ((-1 : ℝ) ^ (3 - k) * hermite1 (3 - k) x.2) * phi2 (u, x.2) := by
    funext u
    exact iteratedDeriv_phi2_snd_eq_hermite1 (3 - k) (by omega) x.2 u
  rw [he, iteratedDeriv_const_mul_field, iteratedDeriv_phi2_fst_eq_hermite1 k hk]
  have hs : (-1 : ℝ) ^ (3 - k) * (-1 : ℝ) ^ k = -1 := by
    rw [← pow_add, show 3 - k + k = 3 by omega]
    norm_num
  dsimp [hermite3]
  calc
    _ = ((-1 : ℝ) ^ (3 - k) * (-1 : ℝ) ^ k) *
        (hermite1 k x.1 * hermite1 (3 - k) x.2) * phi2 x := by ring
    _ = _ := by rw [hs]; ring


example : (3 : ℕ) ≤ 3 := le_rfl

/-- The recorded third Hermite basis equals the source's actual partial-derivative definition.
Source: arXiv:2412.09080v3, §3.1 `eq:psi`, its Gaussian Hermite basis. -/
theorem hermite3_eq_gaussian_partial (k : ℕ) (hk : k ≤ 3) (x : ℝ × ℝ) :
    hermite3 k x = -(phi2 x)⁻¹ *
      iteratedDeriv k (fun u : ℝ => iteratedDeriv (3 - k) (fun v : ℝ => phi2 (u, v)) x.2) x.1 := by
  have hp : 0 < phi2 x := by unfold phi2; positivity
  rw [mixed_iteratedDeriv_phi2_eq_hermite3 k hk]
  field_simp [hp.ne']

example : (3 : ℕ) ≤ 3 := le_rfl

end Transformer.Modes
