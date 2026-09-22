/-
# The number of modes of a Gaussian KDE — the moments behind §5.2

The five expectations of `lem:moments-p` are all read the same way in §5.2 of
arXiv:2412.09080v3: complete the square in the exponent, shift, and integrate
a polynomial of degree at most four against `e^{-au²}`.  This file carries
that reading once, for every such integrand.

* `integral_odd_pow_mul_exp_neg_mul_sq`, `integral_even_pow_mul_exp_neg_mul_sq`:
  the full-line moments `∫ uᵐ e^{-au²}`, zero for odd `m` and
  `Γ((m+1)/2) a^{-(m+1)/2}` for even `m` — the full-line form of the first
  display of `lem:gaussian-int`.
* `integral_quartic_mul_exp`: a quartic against `e^{-au²}`, the odd
  coefficients dropping out.
* `integral_gaussianReal_of_square`: the expectation under `N(0,1)` of an
  integrand whose product with the density is a shifted quartic Gaussian.

Source: arXiv:2412.09080v3, §5.1 (`lem:gaussian-int`) and §5.2.
-/

import Transformer.Modes.Section2_GaussianInt
import Mathlib.Probability.Distributions.Gaussian.Real

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

/-- `uᵐ e^{-au²}` is integrable on the line, for `a > 0`. -/
theorem integrable_pow_mul_exp_neg_mul_sq {a : ℝ} (ha : 0 < a) (m : ℕ) :
    Integrable (fun u : ℝ => u ^ m * Real.exp (-a * u ^ 2)) := by
  have hm : (-1 : ℝ) < (m : ℝ) := by
    have := Nat.cast_nonneg (α := ℝ) m
    linarith
  convert integrable_rpow_mul_exp_neg_mul_sq ha hm using 2 with u
  rw [Real.rpow_natCast]

/-- The hypothesis of `integrable_pow_mul_exp_neg_mul_sq` is satisfiable. -/
example : Integrable (fun u : ℝ => u ^ 2 * Real.exp (-1 * u ^ 2)) :=
  integrable_pow_mul_exp_neg_mul_sq one_pos 2

/-- The odd moments vanish: `u ↦ uᵐ e^{-au²}` is odd for odd `m`.

Source: arXiv:2412.09080v3, §5.2. -/
theorem integral_odd_pow_mul_exp_neg_mul_sq (a : ℝ) {m : ℕ} (hm : Odd m) :
    ∫ u : ℝ, u ^ m * Real.exp (-a * u ^ 2) = 0 := by
  have h := integral_neg_eq_self (fun u : ℝ => u ^ m * Real.exp (-a * u ^ 2)) volume
  simp only [hm.neg_pow, neg_sq, neg_mul, integral_neg] at h ⊢
  linarith

/-- The hypothesis of `integral_odd_pow_mul_exp_neg_mul_sq` is satisfiable. -/
example : ∫ u : ℝ, u ^ 3 * Real.exp (-1 * u ^ 2) = 0 :=
  integral_odd_pow_mul_exp_neg_mul_sq 1 ⟨1, rfl⟩

/-- The even moments on the whole line: twice the half-line moment of
`lem:gaussian-int`, `∫ u^{2k} e^{-au²} = Γ(k + ½) a^{-(2k+1)/2}`.

Source: arXiv:2412.09080v3, `lem:gaussian-int`. -/
theorem integral_even_pow_mul_exp_neg_mul_sq {a : ℝ} (ha : 0 < a) (k : ℕ) :
    ∫ u : ℝ, u ^ (2 * k) * Real.exp (-a * u ^ 2)
      = Real.Gamma ((((2 * k : ℕ) : ℝ) + 1) / 2) * a ^ (-(((2 * k : ℕ) : ℝ) + 1) / 2) := by
  have h := integral_comp_abs (f := fun u : ℝ => u ^ (2 * k) * Real.exp (-a * u ^ 2))
  simp only [sq_abs, (even_two_mul k).pow_abs] at h
  rw [h, integral_pow_mul_exp_neg_mul_sq ha]
  ring

/-- The hypothesis of `integral_even_pow_mul_exp_neg_mul_sq` is satisfiable. -/
example : ∫ u : ℝ, u ^ (2 * 1) * Real.exp (-1 * u ^ 2)
    = Real.Gamma ((((2 * 1 : ℕ) : ℝ) + 1) / 2) * 1 ^ (-(((2 * 1 : ℕ) : ℝ) + 1) / 2) :=
  integral_even_pow_mul_exp_neg_mul_sq one_pos 1

/-- **A quartic against `e^{-au²}`.**  The odd coefficients drop out and the
even ones are weighted by `1`, `1/(2a)`, `3/(4a²)`:
`∫ (p₀ + p₁u + p₂u² + p₃u³ + p₄u⁴) e^{-au²} = √π/√a · (p₀ + p₂/(2a) + 3p₄/(4a²))`.

Source: arXiv:2412.09080v3, §5.2, the moments of `u` read off
`lem:gaussian-int`. -/
theorem integral_quartic_mul_exp {a : ℝ} (ha : 0 < a) (p₀ p₁ p₂ p₃ p₄ : ℝ) :
    ∫ u : ℝ, (p₀ + p₁ * u + p₂ * u ^ 2 + p₃ * u ^ 3 + p₄ * u ^ 4) * Real.exp (-a * u ^ 2)
      = √π / √a * (p₀ + p₂ / (2 * a) + 3 * p₄ / (4 * a ^ 2)) := by
  have hI := integrable_pow_mul_exp_neg_mul_sq ha
  have hsplit : (fun u : ℝ => (p₀ + p₁ * u + p₂ * u ^ 2 + p₃ * u ^ 3 + p₄ * u ^ 4)
      * Real.exp (-a * u ^ 2)) = fun u => p₀ * (u ^ 0 * Real.exp (-a * u ^ 2))
        + p₁ * (u ^ 1 * Real.exp (-a * u ^ 2)) + p₂ * (u ^ 2 * Real.exp (-a * u ^ 2))
        + p₃ * (u ^ 3 * Real.exp (-a * u ^ 2)) + p₄ * (u ^ 4 * Real.exp (-a * u ^ 2)) := by
    ext u; ring
  have g1 : Real.Gamma ((((2 * 0 : ℕ) : ℝ) + 1) / 2) = √π := by
    norm_num [Real.Gamma_one_half_eq]
  have g3 : Real.Gamma ((((2 * 1 : ℕ) : ℝ) + 1) / 2) = √π / 2 := by
    rw [show (((2 * 1 : ℕ) : ℝ) + 1) / 2 = 1 / 2 + 1 by norm_num,
      Real.Gamma_add_one (by norm_num), Real.Gamma_one_half_eq]
    ring
  have g5 : Real.Gamma ((((2 * 2 : ℕ) : ℝ) + 1) / 2) = 3 * √π / 4 := by
    rw [show (((2 * 2 : ℕ) : ℝ) + 1) / 2 = (1 / 2 + 1) + 1 by norm_num,
      Real.Gamma_add_one (by norm_num), Real.Gamma_add_one (by norm_num),
      Real.Gamma_one_half_eq]
    ring
  have r : ∀ k : ℕ, a ^ (-(((2 * k : ℕ) : ℝ) + 1) / 2) = (√a)⁻¹ * (a ^ k)⁻¹ := by
    intro k
    rw [show -(((2 * k : ℕ) : ℝ) + 1) / 2 = -(1 / 2) + -(k : ℝ) by push_cast; ring,
      Real.rpow_add ha, Real.rpow_neg ha.le, Real.rpow_neg ha.le, Real.rpow_natCast,
      ← Real.sqrt_eq_rpow]
  have m0 := integral_even_pow_mul_exp_neg_mul_sq ha 0
  have m2 := integral_even_pow_mul_exp_neg_mul_sq ha 1
  have m4 := integral_even_pow_mul_exp_neg_mul_sq ha 2
  rw [g1, r] at m0
  rw [g3, r] at m2
  rw [g5, r] at m4
  simp only [Nat.reduceMul] at m0 m2 m4
  have hs : 0 < √a := Real.sqrt_pos.mpr ha
  rw [hsplit, integral_add, integral_add, integral_add, integral_add,
    integral_const_mul, integral_const_mul, integral_const_mul, integral_const_mul,
    integral_const_mul, integral_odd_pow_mul_exp_neg_mul_sq a (m := 1) ⟨0, rfl⟩,
    integral_odd_pow_mul_exp_neg_mul_sq a (m := 3) ⟨1, rfl⟩, m0, m2, m4]
  · field_simp
    ring
  all_goals first
    | exact (hI _).const_mul _
    | exact (((hI _).const_mul _).add ((hI _).const_mul _))
    | exact ((((hI _).const_mul _).add ((hI _).const_mul _)).add ((hI _).const_mul _))
    | exact (((((hI _).const_mul _).add ((hI _).const_mul _)).add
        ((hI _).const_mul _)).add ((hI _).const_mul _))

/-- The hypothesis of `integral_quartic_mul_exp` is satisfiable. -/
example : ∫ u : ℝ, (1 + 0 * u + 0 * u ^ 2 + 0 * u ^ 3 + 0 * u ^ 4) * Real.exp (-1 * u ^ 2)
    = √π / √1 * (1 + 0 / (2 * 1) + 3 * 0 / (4 * 1 ^ 2)) :=
  integral_quartic_mul_exp one_pos 1 0 0 0 0

/-- The half-integer powers the closed forms of §5.2 are written in:
`x^{(2k+1)/2} = xᵏ √x`. -/
theorem rpow_odd_half {x : ℝ} (hx : 0 < x) (k : ℕ) :
    x ^ ((2 * (k : ℝ) + 1) / 2) = x ^ k * √x := by
  rw [show (2 * (k : ℝ) + 1) / 2 = (k : ℝ) + 1 / 2 by ring, Real.rpow_add hx,
    Real.rpow_natCast, ← Real.sqrt_eq_rpow]

/-- The hypothesis of `rpow_odd_half` is satisfiable. -/
example : (4 : ℝ) ^ ((2 * ((1 : ℕ) : ℝ) + 1) / 2) = 4 ^ 1 * √4 := rpow_odd_half (by norm_num) 1

/-- **Completing the square, under `N(0,1)`.**  If the density's exponential
times `f` is a quartic in `x - c` against `e^{-a(x-c)²}`, up to a constant `C`,
then `E f(X)` is that quartic's integral: shift by `c`, and read the moments
off `integral_quartic_mul_exp`.

Source: arXiv:2412.09080v3, §5.2, the step shared by all five moments. -/
theorem integral_gaussianReal_of_square {f : ℝ → ℝ} {a c C p₀ p₁ p₂ p₃ p₄ : ℝ} (ha : 0 < a)
    (hf : ∀ x, Real.exp (-x ^ 2 / 2) * f x = C * ((p₀ + p₁ * (x - c) + p₂ * (x - c) ^ 2
      + p₃ * (x - c) ^ 3 + p₄ * (x - c) ^ 4) * Real.exp (-a * (x - c) ^ 2))) :
    ∫ x, f x ∂gaussianReal 0 1
      = C / √(2 * π) * (√π / √a * (p₀ + p₂ / (2 * a) + 3 * p₄ / (4 * a ^ 2))) := by
  rw [integral_gaussianReal_eq_integral_smul one_ne_zero]
  have e : (fun x => gaussianPDFReal 0 1 x • f x) = fun x => C / √(2 * π) *
      ((fun u => (p₀ + p₁ * u + p₂ * u ^ 2 + p₃ * u ^ 3 + p₄ * u ^ 4)
        * Real.exp (-a * u ^ 2)) (x - c)) := by
    ext x
    simp only [gaussianPDFReal_def, smul_eq_mul, NNReal.coe_one, mul_one, sub_zero]
    rw [mul_assoc, hf]
    ring
  rw [e, integral_const_mul, integral_sub_right_eq_self (μ := volume)
    (fun u => (p₀ + p₁ * u + p₂ * u ^ 2 + p₃ * u ^ 3 + p₄ * u ^ 4) * Real.exp (-a * u ^ 2)) c,
    integral_quartic_mul_exp ha]

/-- The hypotheses of `integral_gaussianReal_of_square` are satisfiable: at
`f = 1`, `c = 0`, `a = ½`, `C = 1`, `E 1 = 1`. -/
example : ∫ _x, (1 : ℝ) ∂gaussianReal 0 1
    = 1 / √(2 * π) * (√π / √(1 / 2) * (1 + 0 / (2 * (1 / 2)) + 3 * 0 / (4 * (1 / 2) ^ 2))) :=
  integral_gaussianReal_of_square (p₁ := 0) (p₃ := 0) (by norm_num) fun x => by
    rw [sub_zero]; ring_nf

end Modes
end Transformer
