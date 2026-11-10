module

public import Transformer.Grokking.FlatSpectrum.Compare

/-!
# The step of an additive recurrence modulo `p`: a residue and a wrap

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), items CW1 and
CW6. A difference encoding of the recurrence `x ↦ (b + x) % p` writes the integer step
`modStep p b x = (b + x) % p − x` from the value `x`, where `b` is the value before `x`.

* The step is a residue and a wrap (`modStep_eq`): it is `b` if `b + x < p`, else `b − p`. Its
  residue is `b` (`intCast_modStep`), and its sign is the wrap (`modStep_neg_iff`).
* The residue of a step is the sum of the residues of the two steps before it
  (`modStep_modEq_add`). So every additive character of the step is the product of its values
  at these two steps (`addChar_modStep`): a kernel of rank one in the two steps, which the
  theorems on unimodular kernels of rank one cover.
* The sign of the step, `wrapKernel p`, is the comparison kernel with its columns reversed
  (`wrapKernel_eq_submatrix`): `b + x < p` reads `b ≤ p − 1 − x`. `WrapFill` carries the
  results on the comparison kernel over to it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix

/-- The integer step from the value `x` to the next value `(b + x) % p` of an additive
recurrence modulo `p`, where `b` is the value before `x`; `review/flat_spectrum.md`,
item CW1. -/
def modStep (p b x : ℕ) : ℤ := ((b + x) % p : ℕ) - (x : ℤ)

/-- **A step is a residue and a wrap**, `review/flat_spectrum.md`, item CW1: the step is `b`
if the sum `b + x` stays below `p`, and `b − p` if it wraps. -/
theorem modStep_eq {p b x : ℕ} (hb : b < p) (hx : x < p) :
    modStep p b x = b - if p ≤ b + x then (p : ℤ) else 0 := by
  unfold modStep
  split_ifs with h
  · rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt (by omega)]
    omega
  · rw [Nat.mod_eq_of_lt (by omega)]
    omega

/-- Modulo `97`, from `50` and `60` the next value is `13`: the step is `50 − 97`. -/
example : modStep 97 50 60 = 50 - 97 := by
  rw [modStep_eq (by norm_num) (by norm_num)]
  norm_num

/-- The step is negative exactly when the sum wraps. -/
theorem modStep_neg_iff {p b x : ℕ} (hb : b < p) (hx : x < p) :
    modStep p b x < 0 ↔ p ≤ b + x := by
  rw [modStep_eq hb hx]
  split_ifs with h <;> constructor <;> intro <;> omega

example : modStep 97 50 60 < 0 := (modStep_neg_iff (by norm_num) (by norm_num)).mpr (by norm_num)

/-- The residue of the step is the value before: `modStep p b x ≡ b`. -/
theorem intCast_modStep (p b x : ℕ) : ((modStep p b x : ℤ) : ZMod p) = b := by
  simp [modStep]

/-- **The residues of the steps add**, `review/flat_spectrum.md`, item CW1: along the values
`a`, `b`, `(a + b) % p`, the step after them is congruent to the sum of the two steps before
it, `b − a` and `modStep p a b`. -/
theorem modStep_modEq_add (p a b : ℕ) :
    modStep p b ((a + b) % p) ≡ ((b : ℤ) - a) + modStep p a b [ZMOD p] := by
  rw [← ZMod.intCast_eq_intCast_iff, intCast_modStep, Int.cast_add, intCast_modStep]
  push_cast
  ring

/-- **The phase of a step has rank one**, `review/flat_spectrum.md`, item CW1: every additive
character of the step is the product of its values at the two steps before it. -/
theorem addChar_modStep {p : ℕ} (ψ : AddChar (ZMod p) ℂ) (a b : ℕ) :
    ψ ((modStep p b ((a + b) % p) : ℤ) : ZMod p) =
      ψ (((b : ℤ) - a : ℤ) : ZMod p) * ψ ((modStep p a b : ℤ) : ZMod p) := by
  rw [← AddChar.map_add_eq_mul, ← Int.cast_add,
    (ZMod.intCast_eq_intCast_iff _ _ p).mpr (modStep_modEq_add p a b)]

/-- The sign of the step, `review/flat_spectrum.md`, item CW6: `−1` where the sum `b + x`
wraps past `p`, and `1` where it does not. -/
def wrapKernel (p : ℕ) : Matrix (Fin p) (Fin p) ℂ :=
  of fun b x => if modStep p b x < 0 then -1 else 1

/-- **The wrap is a comparison**, `review/flat_spectrum.md`, item CW6: the sign of the step
is the comparison kernel with its columns reversed, as `b + x < p` reads `b ≤ p − 1 − x`. -/
theorem wrapKernel_eq_submatrix (p : ℕ) :
    wrapKernel p = (compareKernel p).submatrix (Equiv.refl _) Fin.revPerm := by
  ext b x
  simp only [wrapKernel, compareKernel, of_apply, submatrix_apply, Equiv.refl_apply,
    Fin.revPerm_apply, modStep_neg_iff b.isLt x.isLt, Fin.le_def, Fin.val_rev]
  split_ifs <;> first | rfl | omega

/-- Reversing the columns again gives the comparison kernel back. -/
theorem wrapKernel_submatrix (p : ℕ) :
    (wrapKernel p).submatrix (Equiv.refl _) Fin.revPerm = compareKernel p := by
  rw [wrapKernel_eq_submatrix]
  ext i j
  simp

end Transformer.Grokking.FlatSpectrum
