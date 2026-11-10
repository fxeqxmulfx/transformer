module

public import Transformer.Grokking.FlatSpectrum.StepHalfPlane

/-!
# The sign of the step is a real kernel of rank two with its modulus normalized

Source: `review/flat_spectrum.md`, *The step is the phase of a sum of two phases* (added),
item CH6. Along the values `a`, `b`, `x = (a + b) % p` let `d₁ = b − a` and `d₂ = modStep p a b`
be the steps read. The half-plane value of CW7 is the real number

`stepHalfPlane p a b = Im (exp (iπ / 2p) · exp (2πi d₁ / p) · exp (3πi d₂ / p))`.

* It is a sum of two products of a real function of `d₁` and a real function of `d₂`
  (`stepHalfPlane_eq_add`): a real kernel of rank at most two in the two steps.
* It is `sin (π (n + ½) / p)` for the sum `n = b + x` (`stepHalfPlane_eq_sin`), so it is never
  zero (`stepHalfPlane_ne_zero`): positive without a wrap and negative with one.
* With its modulus normalized it is the sign of the step to write
  (`stepHalfPlane_div_abs`): `ℓ / |ℓ|` is `−1` where the next step wraps and `1` elsewhere.

So the comparison of `WrapFill` is one real channel of a bilinear map of the two steps, with
the modulus of that channel normalized.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Complex Real

/-- The half-plane value of the sum, `review/flat_spectrum.md`, items CW7 and CH6:
`Im (exp (iπ / 2p) · exp (2πi d₁ / p) · exp (3πi d₂ / p))`. -/
noncomputable def stepHalfPlane (p a b : ℕ) : ℝ :=
  (cexp (π * I / (2 * p)) *
    (cexp (2 * π * I * ((b : ℂ) - a) / p) * cexp (3 * π * I * (modStep p a b : ℂ) / p))).im

/-- **A real kernel of rank at most two**, `review/flat_spectrum.md`, item CH6: the half-plane
value is a sum of two products of a real function of `d₁` and a real function of `d₂`. -/
theorem stepHalfPlane_eq_add (p a b : ℕ) :
    stepHalfPlane p a b =
      (cexp (π * I / (2 * p)) * cexp (2 * π * I * ((b : ℂ) - a) / p)).re *
          (cexp (3 * π * I * (modStep p a b : ℂ) / p)).im +
        (cexp (π * I / (2 * p)) * cexp (2 * π * I * ((b : ℂ) - a) / p)).im *
          (cexp (3 * π * I * (modStep p a b : ℂ) / p)).re := by
  rw [stepHalfPlane, ← mul_assoc, Complex.mul_im]

/-- The half-plane value is the sine of the angle of the sum, moved by half a step. -/
theorem stepHalfPlane_eq_sin {p a b : ℕ} (ha : a < p) (hb : b < p) :
    stepHalfPlane p a b = Real.sin (π * (((b + (a + b) % p : ℕ) : ℝ) + 1 / 2) / p) := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h : π * I / (2 * p) + π * I * ((b + (a + b) % p : ℕ) : ℂ) / p =
      ((π * (((b + (a + b) % p : ℕ) : ℝ) + 1 / 2) / p : ℝ) : ℂ) * I := by
    push_cast
    field_simp
    ring
  rw [stepHalfPlane, ← exp_sum_eq_mul ha hb, ← Complex.exp_add, h, exp_ofReal_mul_I_im]

example := stepHalfPlane_eq_sin (p := 97) (a := 50) (b := 90) (by norm_num) (by norm_num)

/-- A half-plane: for `n < p`, `sin (π (n + ½) / p)` is positive. -/
theorem sin_pos_of_lt {p n : ℕ} (hn : n < p) : 0 < Real.sin (π * (n + 1 / 2) / p) := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (by omega : 0 < p)
  have h1 : (n : ℝ) + 1 ≤ p := by exact_mod_cast hn
  refine sin_pos_of_pos_of_lt_pi (by positivity) ?_
  rw [div_lt_iff₀ hp]
  nlinarith [pi_pos]

example : 0 < Real.sin (π * ((73 : ℕ) + 1 / 2) / (97 : ℕ)) := sin_pos_of_lt (by norm_num)

/-- The half-plane value is positive exactly when the next step does not wrap. -/
theorem stepHalfPlane_pos_iff {p a b : ℕ} (ha : a < p) (hb : b < p) :
    0 < stepHalfPlane p a b ↔ ¬modStep p b ((a + b) % p) < 0 := by
  have hx : (a + b) % p < p := Nat.mod_lt _ (by omega)
  rw [modStep_neg_iff hb hx, stepHalfPlane_eq_sin ha hb]
  constructor
  · intro h hw
    have := (sin_neg_iff_le (p := p) (n := b + (a + b) % p) (by omega)).mpr hw
    linarith
  · intro h
    exact sin_pos_of_lt (by omega)

example := stepHalfPlane_pos_iff (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- The half-plane value is never zero: no sum lies on the boundary of the half-plane. -/
theorem stepHalfPlane_ne_zero {p a b : ℕ} (ha : a < p) (hb : b < p) :
    stepHalfPlane p a b ≠ 0 := by
  by_cases h : modStep p b ((a + b) % p) < 0
  · have := (modStep_neg_iff_im_neg ha hb).mp h
    exact ne_of_lt this
  · exact ((stepHalfPlane_pos_iff ha hb).mpr h).ne'

example := stepHalfPlane_ne_zero (p := 97) (a := 50) (b := 90) (by norm_num) (by norm_num)

/-- **The sign of the step is a normalized modulus**, `review/flat_spectrum.md`, item CH6:
the real kernel of rank at most two `stepHalfPlane`, divided by its absolute value, is `−1`
where the step to write is negative and `1` where it is not: the entry of `wrapKernel`. -/
theorem stepHalfPlane_div_abs {p a b : ℕ} (ha : a < p) (hb : b < p) :
    stepHalfPlane p a b / |stepHalfPlane p a b| =
      if modStep p b ((a + b) % p) < 0 then -1 else 1 := by
  by_cases h : modStep p b ((a + b) % p) < 0
  · have hneg : stepHalfPlane p a b < 0 := (modStep_neg_iff_im_neg ha hb).mp h
    rw [abs_of_neg hneg, div_neg, div_self hneg.ne]
    simp [h]
  · have hpos := (stepHalfPlane_pos_iff ha hb).mpr h
    rw [abs_of_pos hpos, div_self hpos.ne']
    simp [h]

/-- From `50` and `90` the next value is `43` and the step after it wraps. -/
example := stepHalfPlane_div_abs (p := 97) (a := 50) (b := 90) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum
