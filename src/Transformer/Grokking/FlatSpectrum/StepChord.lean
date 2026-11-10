module

public import Transformer.Grokking.FlatSpectrum.StepHalfPlane

/-!
# The chord of the sum

Source: `review/flat_spectrum.md`, *The step is the phase of a sum of two phases* (added),
items CH1 and CH2.

* The sum of two phases is the product of their half phases, scaled by the cosine of half the
  difference of the angles (`exp_mul_I_add_exp_mul_I`):
  `exp (iα) + exp (iβ) = 2 cos ((α − β) / 2) · exp (i (α + β) / 2)`. Normalizing its modulus
  gives a product, and a sign that compares the two angles.
* Along the values `a`, `b`, `x = (a + b) % p` let `d₁ = b − a` and `d₂ = modStep p a b` be the
  steps read, and `z = exp (2πi d₁ / p) · exp (3πi d₂ / p)` the phase of the sum `n = b + x`, a
  kernel of rank one in the two steps (`exp_sum_eq_mul`). The chord from `−exp (−iπ / 2p)` to
  `z` is `stepChord p a b = z + exp (−iπ / 2p)`, and (`stepChord_eq`)
  `stepChord p a b = 2 cos φ · exp (i (φ − π / 2p))`, `φ = π (n + ½) / 2p`.
* The angle `φ` lies in `(0, π)`, below `π / 2` exactly when the sum does not wrap
  (`cos_halfAngle_pos`, `cos_halfAngle_neg`). So the chord is not zero (`stepChord_ne_zero`),
  its length is `2 |cos φ|` (`norm_stepChord`), its phase halves the angle of `z`, and it
  turns by `π` at the wrap. `StepPhase` reads the step from it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Complex Real

/-- **The sum of two phases**, `review/flat_spectrum.md`, item CH1: it is the product of the
two half phases, scaled by twice the cosine of half the difference of the angles. -/
theorem exp_mul_I_add_exp_mul_I (α β : ℝ) :
    cexp (α * I) + cexp (β * I) =
      ((2 * Real.cos ((α - β) / 2) : ℝ) : ℂ) * cexp ((((α + β) / 2 : ℝ) : ℂ) * I) := by
  have h1 : (α : ℂ) * I = (((α - β) / 2 : ℝ) : ℂ) * I + (((α + β) / 2 : ℝ) : ℂ) * I := by
    push_cast
    ring
  have h2 : (β : ℂ) * I = -(((α - β) / 2 : ℝ) : ℂ) * I + (((α + β) / 2 : ℝ) : ℂ) * I := by
    push_cast
    ring
  rw [h1, h2, Complex.exp_add, Complex.exp_add, ← add_mul, ← Complex.two_cos,
    Complex.ofReal_mul, Complex.ofReal_cos, Complex.ofReal_ofNat]

/-- Half the angle of the sum, moved by half a step: `φ = π (n + ½) / 2p` for the sum
`n = b + (a + b) % p`; `review/flat_spectrum.md`, item CH2. -/
noncomputable def halfAngle (p a b : ℕ) : ℝ :=
  π * (((b + (a + b) % p : ℕ) : ℝ) + 1 / 2) / (2 * p)

/-- The chord from `−exp (−iπ / 2p)` to the phase of the sum
`exp (2πi d₁ / p) · exp (3πi d₂ / p)`: a kernel of rank one in the two steps read, plus a
constant; `review/flat_spectrum.md`, item CH2. -/
noncomputable def stepChord (p a b : ℕ) : ℂ :=
  cexp (2 * π * I * ((b : ℂ) - a) / p) * cexp (3 * π * I * (modStep p a b : ℂ) / p) +
    cexp (-(π * I) / (2 * p))

/-- **The chord in polar form**, `review/flat_spectrum.md`, item CH2:
`stepChord = 2 cos φ · exp (i (φ − π / 2p))`. -/
theorem stepChord_eq {p a b : ℕ} (ha : a < p) (hb : b < p) :
    stepChord p a b = ((2 * Real.cos (halfAngle p a b) : ℝ) : ℂ) *
      cexp (((halfAngle p a b - π / (2 * p) : ℝ) : ℂ) * I) := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h1 : π * I * ((b + (a + b) % p : ℕ) : ℂ) / p =
      (halfAngle p a b : ℂ) * I + ((halfAngle p a b - π / (2 * p) : ℝ) : ℂ) * I := by
    simp only [halfAngle]
    push_cast
    field_simp
    ring
  have h2 : -(π * I) / (2 * p) =
      -(halfAngle p a b : ℂ) * I + ((halfAngle p a b - π / (2 * p) : ℝ) : ℂ) * I := by
    push_cast
    ring
  rw [stepChord, ← exp_sum_eq_mul ha hb, h1, h2, Complex.exp_add, Complex.exp_add, ← add_mul,
    ← Complex.two_cos, Complex.ofReal_mul, Complex.ofReal_cos, Complex.ofReal_ofNat]

example := stepChord_eq (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- Without a wrap the half angle is below `π / 2`. -/
theorem cos_halfAngle_pos {p a b : ℕ} (h : b + (a + b) % p < p) :
    0 < Real.cos (halfAngle p a b) := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (by omega : 0 < p)
  have h1 : ((b + (a + b) % p : ℕ) : ℝ) + 1 ≤ p := by exact_mod_cast h
  refine cos_pos_of_mem_Ioo ⟨?_, ?_⟩
  · have h2 : 0 ≤ halfAngle p a b := by
      unfold halfAngle
      positivity
    linarith [pi_pos]
  · unfold halfAngle
    rw [div_lt_iff₀ (by positivity)]
    nlinarith [pi_pos]

example : 0 < Real.cos (halfAngle 97 10 20) := cos_halfAngle_pos (by norm_num)

/-- With a wrap the half angle is between `π / 2` and `π`. -/
theorem cos_halfAngle_neg {p a b : ℕ} (hb : b < p) (h : p ≤ b + (a + b) % p) :
    Real.cos (halfAngle p a b) < 0 := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (by omega : 0 < p)
  have hx : (a + b) % p < p := Nat.mod_lt _ (by omega)
  have h1 : (p : ℝ) ≤ ((b + (a + b) % p : ℕ) : ℝ) := by exact_mod_cast h
  have h2 : ((b + (a + b) % p : ℕ) : ℝ) + 2 ≤ 2 * p := by exact_mod_cast (by omega)
  refine cos_neg_of_pi_div_two_lt_of_lt ?_ ?_
  · unfold halfAngle
    rw [lt_div_iff₀ (by positivity)]
    nlinarith [pi_pos]
  · unfold halfAngle
    rw [div_lt_iff₀ (by positivity)]
    nlinarith [pi_pos]

/-- From `50` and `90` the next value is `43`, and `90 + 43` wraps. -/
example : Real.cos (halfAngle 97 50 90) < 0 := cos_halfAngle_neg (by norm_num) (by norm_num)

/-- The chord is not zero: the phase of the sum is never `−exp (−iπ / 2p)`. -/
theorem stepChord_ne_zero {p a b : ℕ} (ha : a < p) (hb : b < p) : stepChord p a b ≠ 0 := by
  rw [stepChord_eq ha hb]
  refine mul_ne_zero (Complex.ofReal_ne_zero.mpr (mul_ne_zero two_ne_zero ?_))
    (Complex.exp_ne_zero _)
  rcases lt_or_ge (b + (a + b) % p) p with h | h
  · exact (cos_halfAngle_pos h).ne'
  · exact (cos_halfAngle_neg hb h).ne

example := stepChord_ne_zero (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- The length of the chord: `2 |cos φ|`. -/
theorem norm_stepChord {p a b : ℕ} (ha : a < p) (hb : b < p) :
    ‖stepChord p a b‖ = 2 * |Real.cos (halfAngle p a b)| := by
  rw [stepChord_eq ha hb, norm_mul, Complex.norm_real, Complex.norm_exp_ofReal_mul_I, mul_one,
    Real.norm_eq_abs, abs_mul, abs_two]

example := norm_stepChord (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum
