module

public import Transformer.Grokking.FlatSpectrum.StepChord

/-!
# The step is the phase of a sum of two phases

Source: `review/flat_spectrum.md`, *The step is the phase of a sum of two phases* (added),
items CH3 and CH4. Along the values `a`, `b`, `x = (a + b) % p` let `d₁ = b − a` and
`d₂ = modStep p a b` be the steps read and `d₃ = modStep p b x` the step to write.

* The phase of order `2p` of the step, which tells `b` from `b − p`, is the phase of the chord
  of `StepChord`, times a phase of the last step read (`exp_modStep_eq_stepChord`):
  `exp (iπ d₃ / p) = exp (iπ / 4p) · exp (−iπ d₂ / 2p) · stepChord / ‖stepChord‖`.
* With a character `ψ` of `ℤ/p`, `ψ (d₃) · exp (iπ d₃ / p)` is a character of odd order of
  `ℤ/2p`, and it is `ψ (d₁) · ψ (d₂)` times that phase (`addChar_mul_exp_modStep`).
* The chord is, up to a factor of modulus one, the sum of one phase of each step
  (`stepPhaseSum`, `stepPhaseSum_eq`, `norm_stepPhaseSum`):
  `exp (iπ / 2p) · exp (3πi d₂ / p) + exp (−2πi d₁ / p)`. So the step is the phase of a sum of
  a function of each of the two steps read, times a kernel of rank one
  (`exp_modStep_eq_stepPhaseSum`).

The wrap, a comparison for a readout of the sign (`WrapFill`), is the branch of the half
angle when the modulus of the sum is normalized.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Complex Real

/-- **The step is the phase of a chord**, `review/flat_spectrum.md`, item CH3: along `a`,
`b`, `(a + b) % p`, with `d₂ = modStep p a b` and `d₃` the step to write,
`exp (iπ d₃ / p) = exp (iπ / 4p) · exp (−iπ d₂ / 2p) · stepChord / ‖stepChord‖`. -/
theorem exp_modStep_eq_stepChord {p a b : ℕ} (ha : a < p) (hb : b < p) :
    cexp (π * I * (modStep p b ((a + b) % p) : ℂ) / p) =
      cexp (π * I / (4 * p)) * cexp (-(π * I * (modStep p a b : ℂ)) / (2 * p)) *
        (stepChord p a b / (‖stepChord p a b‖ : ℂ)) := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hx : (a + b) % p < p := Nat.mod_lt _ (by omega)
  have hx' : (((a + b) % p : ℕ) : ℂ) = (modStep p a b : ℂ) + b := by
    rw [modStep, Int.cast_sub, Int.cast_natCast, Int.cast_natCast, sub_add_cancel]
  have hE : cexp (π * I / (4 * p)) * cexp (-(π * I * (modStep p a b : ℂ)) / (2 * p)) *
      cexp (((halfAngle p a b - π / (2 * p) : ℝ) : ℂ) * I) = cexp (π * I * (b : ℂ) / p) := by
    rw [← Complex.exp_add, ← Complex.exp_add]
    congr 1
    simp only [halfAngle]
    push_cast
    rw [hx']
    field_simp
    ring
  rw [norm_stepChord ha hb, stepChord_eq ha hb]
  rcases lt_or_ge (b + (a + b) % p) p with h | h
  · have hc := cos_halfAngle_pos h
    have hd : modStep p b ((a + b) % p) = b := by
      rw [modStep_eq hb hx]
      simp [not_le.mpr h]
    have hc' : ((2 * Real.cos (halfAngle p a b) : ℝ) : ℂ) ≠ 0 :=
      Complex.ofReal_ne_zero.mpr (by positivity)
    rw [hd, Int.cast_natCast, abs_of_pos hc, mul_div_cancel_left₀ _ hc', hE]
  · have hc := cos_halfAngle_neg hb h
    have hd : modStep p b ((a + b) % p) = b - p := by
      rw [modStep_eq hb hx]
      simp [h]
    have hc' : ((2 * Real.cos (halfAngle p a b) : ℝ) : ℂ) ≠ 0 :=
      Complex.ofReal_ne_zero.mpr (mul_ne_zero two_ne_zero hc.ne)
    have hneg : ((2 * -Real.cos (halfAngle p a b) : ℝ) : ℂ) =
        -((2 * Real.cos (halfAngle p a b) : ℝ) : ℂ) := by
      push_cast
      ring
    have hw : cexp (π * I * ((b : ℂ) - p) / p) = -cexp (π * I * (b : ℂ) / p) := by
      have h1 : π * I * ((b : ℂ) - p) / p = π * I * (b : ℂ) / p + -(π * I) := by
        field_simp
        ring
      rw [h1, Complex.exp_add, Complex.exp_neg, Complex.exp_pi_mul_I, inv_neg, inv_one, mul_neg,
        mul_one]
    rw [hd, Int.cast_sub, Int.cast_natCast, Int.cast_natCast, hw, abs_of_neg hc, hneg, div_neg,
      mul_div_cancel_left₀ _ hc', mul_neg, hE]

/-- From `50` and `60` the next value is `13`, and `60 + 13` does not wrap. -/
example := exp_modStep_eq_stepChord (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- From `50` and `90` the next value is `43`, and `90 + 43` wraps. -/
example := exp_modStep_eq_stepChord (p := 97) (a := 50) (b := 90) (by norm_num) (by norm_num)

/-- **The characters of odd order of the step**, `review/flat_spectrum.md`, item CH3: a
character `ψ` of `ℤ/p` times the phase of order `2p` is the kernel of rank one
`ψ (d₁) · ψ (d₂)` times the phase of the chord. -/
theorem addChar_mul_exp_modStep {p a b : ℕ} (ψ : AddChar (ZMod p) ℂ) (ha : a < p) (hb : b < p) :
    ψ ((modStep p b ((a + b) % p) : ℤ) : ZMod p) *
        cexp (π * I * (modStep p b ((a + b) % p) : ℂ) / p) =
      ψ (((b : ℤ) - a : ℤ) : ZMod p) * ψ ((modStep p a b : ℤ) : ZMod p) *
        (cexp (π * I / (4 * p)) * cexp (-(π * I * (modStep p a b : ℂ)) / (2 * p)) *
          (stepChord p a b / (‖stepChord p a b‖ : ℂ))) := by
  rw [addChar_modStep, exp_modStep_eq_stepChord ha hb]

example (ψ : AddChar (ZMod 97) ℂ) :=
  addChar_mul_exp_modStep ψ (a := 50) (b := 90) (by norm_num) (by norm_num)

/-- The sum of a phase of the last step read, `exp (iπ / 2p) · exp (3πi d₂ / p)`, and a phase
of the step before it, `exp (−2πi d₁ / p)`; `review/flat_spectrum.md`, item CH4. -/
noncomputable def stepPhaseSum (p a b : ℕ) : ℂ :=
  cexp (π * I / (2 * p)) * cexp (3 * π * I * (modStep p a b : ℂ) / p) +
    cexp (-(2 * π * I * ((b : ℂ) - a)) / p)

/-- The sum of the two phases is the chord, turned: `review/flat_spectrum.md`, item CH4. -/
theorem stepPhaseSum_eq (p a b : ℕ) :
    stepPhaseSum p a b =
      cexp (π * I / (2 * p)) * cexp (-(2 * π * I * ((b : ℂ) - a)) / p) * stepChord p a b := by
  have h1 : cexp (-(2 * π * I * ((b : ℂ) - a)) / p) * cexp (2 * π * I * ((b : ℂ) - a) / p) =
      1 := by
    rw [← Complex.exp_add, neg_div, neg_add_cancel, Complex.exp_zero]
  have h2 : cexp (π * I / (2 * p)) * cexp (-(π * I) / (2 * p)) = 1 := by
    rw [← Complex.exp_add, neg_div, add_neg_cancel, Complex.exp_zero]
  unfold stepPhaseSum stepChord
  set E1 := cexp (π * I / (2 * p))
  set E3 := cexp (3 * π * I * (modStep p a b : ℂ) / p)
  set Em := cexp (-(2 * π * I * ((b : ℂ) - a)) / p)
  linear_combination (-(E1 * E3)) * h1 + (-Em) * h2

/-- The sum of the two phases has the length of the chord. -/
theorem norm_stepPhaseSum (p a b : ℕ) : ‖stepPhaseSum p a b‖ = ‖stepChord p a b‖ := by
  have h1 : ‖cexp (π * I / (2 * p))‖ = 1 := by
    have h : (π : ℂ) * I / (2 * p) = ((π / (2 * p) : ℝ) : ℂ) * I := by
      push_cast
      ring
    rw [h, Complex.norm_exp_ofReal_mul_I]
  have h2 : ‖cexp (-(2 * π * I * ((b : ℂ) - a)) / p)‖ = 1 := by
    have h : -(2 * π * I * ((b : ℂ) - a)) / p =
        ((-(2 * π * ((b : ℝ) - a)) / p : ℝ) : ℂ) * I := by
      push_cast
      ring
    rw [h, Complex.norm_exp_ofReal_mul_I]
  rw [stepPhaseSum_eq, norm_mul, norm_mul, h1, h2, one_mul, one_mul]

/-- **The step is the phase of a sum of two phases**, `review/flat_spectrum.md`, item CH4:
`exp (iπ d₃ / p)` is a kernel of rank one and modulus one in the two steps read, times the sum
of one phase of each of them with its modulus normalized. -/
theorem exp_modStep_eq_stepPhaseSum {p a b : ℕ} (ha : a < p) (hb : b < p) :
    cexp (π * I * (modStep p b ((a + b) % p) : ℂ) / p) =
      cexp (-(π * I) / (4 * p)) * cexp (2 * π * I * ((b : ℂ) - a) / p) *
        cexp (-(π * I * (modStep p a b : ℂ)) / (2 * p)) *
          (stepPhaseSum p a b / (‖stepPhaseSum p a b‖ : ℂ)) := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h1 : cexp (-(π * I) / (4 * p)) * cexp (π * I / (2 * p)) = cexp (π * I / (4 * p)) := by
    rw [← Complex.exp_add]
    congr 1
    field_simp
    ring
  have h2 : cexp (2 * π * I * ((b : ℂ) - a) / p) * cexp (-(2 * π * I * ((b : ℂ) - a)) / p) =
      1 := by
    rw [← Complex.exp_add, neg_div, add_neg_cancel, Complex.exp_zero]
  have h3 : ∀ A B F E Em S N : ℂ, B * Em = 1 →
      A * E * F * (S / N) = A * B * F * (E * Em * S / N) := by
    intro A B F E Em S N h
    linear_combination (-(A * E * F * (S / N))) * h
  rw [exp_modStep_eq_stepChord ha hb, norm_stepPhaseSum, stepPhaseSum_eq, ← h1]
  exact h3 _ _ _ _ _ _ _ h2

example := exp_modStep_eq_stepPhaseSum (p := 97) (a := 50) (b := 90) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum
