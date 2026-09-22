/-
# Kinetic theory for Transformers — the periodic Gaussian

The Poisson summation formula for the periodic Gaussian that mollifies the
retrieval indicator in `eq:Acc-soft` of arXiv:2605.09213v1:

  `Σ_k exp(-M²(D - 2πk)²/(2π²)) = (√(π/2)/M) Σ_n e^{-π²n²/(2M²)} e^{inD}`,

read off Jacobi's transformation formula `Complex.tsum_exp_neg_quadratic`.

Source: arXiv:2605.09213v1, `eq:Acc-soft-Fourier` ("By the Poisson summation
formula").
-/

import Mathlib.Analysis.SpecialFunctions.Gaussian.PoissonSummation

open Real Complex

namespace Transformer
namespace Kinetic

/-- **Poisson summation for the periodic Gaussian**, pointwise:
`Σ_k exp(-M²(D - 2πk)²/(2π²)) = (√(π/2)/M) Σ_n e^{-π²n²/(2M²)} e^{inD}`.
arXiv:2605.09213v1, `eq:Acc-soft-Fourier`. -/
theorem tsum_periodicGaussian_eq {M : ℝ} (hM : 0 < M) (D : ℝ) :
    ((∑' k : ℤ, Real.exp (-(M ^ 2 / (2 * π ^ 2)) * (D - 2 * π * k) ^ 2) : ℝ) : ℂ) =
      (Real.sqrt (π / 2) / M : ℝ) • ∑' n : ℤ,
        (Real.exp (-(π ^ 2 / (2 * M ^ 2)) * (n : ℝ) ^ 2) : ℝ) • cexp ((n : ℂ) * (D : ℂ) * I) := by
  set c : ℝ := M ^ 2 / (2 * π ^ 2)
  set a : ℂ := ((2 * M ^ 2 / π : ℝ) : ℂ)
  set b : ℂ := ((M ^ 2 * D / π ^ 2 : ℝ) : ℂ)
  have hπ : (π : ℂ) ≠ 0 := by exact_mod_cast Real.pi_ne_zero
  have hM' : (M : ℂ) ≠ 0 := by exact_mod_cast hM.ne'
  have ha : 0 < a.re := by simp only [a, Complex.ofReal_re]; positivity
  have h1 : ∀ k : ℤ, ((Real.exp (-c * (D - 2 * π * k) ^ 2) : ℝ) : ℂ) =
      cexp (-(c * D ^ 2 : ℝ)) * cexp (-π * a * k ^ 2 + 2 * π * b * k) := by
    intro k
    rw [Complex.ofReal_exp, ← Complex.exp_add]
    congr 1
    simp only [a, b, c]
    push_cast
    field_simp
    ring
  have h2 : ∀ n : ℤ, cexp (-π / a * (n + I * b) ^ 2) =
      cexp ((c * D ^ 2 : ℝ)) * ((Real.exp (-(π ^ 2 / (2 * M ^ 2)) * ((-n : ℤ) : ℝ) ^ 2) : ℝ) •
        cexp (((-n : ℤ) : ℂ) * (D : ℂ) * I)) := by
    intro n
    rw [Complex.real_smul, Complex.ofReal_exp, ← Complex.exp_add, ← Complex.exp_add]
    congr 1
    simp only [a, b, c]
    push_cast
    field_simp
    ring_nf
    rw [Complex.I_sq]
    ring
  rw [Complex.ofReal_tsum]
  simp_rw [h1]
  rw [tsum_mul_left, Complex.tsum_exp_neg_quadratic ha b]

  simp_rw [h2]
  have hneg := (Equiv.neg ℤ).tsum_eq fun n : ℤ => ((Real.exp (-(π ^ 2 / (2 * M ^ 2)) *
      ((n : ℤ) : ℝ) ^ 2) : ℝ) • cexp (((n : ℤ) : ℂ) * (D : ℂ) * I))
  simp only [Equiv.neg_apply] at hneg
  rw [tsum_mul_left, hneg, ← mul_assoc, ← mul_assoc, mul_comm (cexp _) (1 / _), mul_assoc (1 / _),
    ← Complex.exp_add, neg_add_cancel, Complex.exp_zero, mul_one, Complex.real_smul]
  congr 1
  have hsq : a ^ (1 / 2 : ℂ) = ((√(2 * M ^ 2 / π) : ℝ) : ℂ) := by
    rw [Real.sqrt_eq_rpow, Complex.ofReal_cpow (by positivity)]
    simp only [a]; push_cast; rfl
  rw [hsq, Real.sqrt_div' _ (by positivity : (0:ℝ) ≤ π), Real.sqrt_mul' _ (by positivity),
    Real.sqrt_sq hM.le, Real.sqrt_div' _ (by positivity : (0:ℝ) ≤ 2)]
  have h2 : (0:ℝ) < √2 := by positivity
  have hp : (0:ℝ) < √π := Real.sqrt_pos.2 Real.pi_pos
  push_cast
  field_simp

/-- The Fourier coefficients of the periodic Gaussian are summable:
`Σ_n e^{-c n²} < ∞` for `c > 0`. -/
theorem summable_exp_neg_mul_int_sq {c : ℝ} (hc : 0 < c) :
    Summable fun n : ℤ => Real.exp (-c * (n : ℝ) ^ 2) := by
  have hN : Summable fun n : ℕ => Real.exp (-c * (n : ℝ) ^ 2) :=
    Real.summable_exp_nat_mul_of_ge (neg_lt_zero.2 hc) (f := fun i : ℕ => (i : ℝ) ^ 2)
      fun i => by exact_mod_cast Nat.le_self_pow two_ne_zero i
  exact Summable.of_nat_of_neg (by simpa using hN) (by simpa using hN)

/-- The hypotheses of `tsum_periodicGaussian_eq` and
`summable_exp_neg_mul_int_sq` are satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

end Kinetic
end Transformer
