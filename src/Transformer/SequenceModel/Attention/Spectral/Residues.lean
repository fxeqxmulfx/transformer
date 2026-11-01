module

public import Transformer.SequenceModel.Attention.Spectral.Recovery
public import Transformer.SequenceModel.Attention.Expressivity.Existence

/-!
# Residue inputs, zero numerators and unit character features

Source: `review/flat_spectrum.md`, *Unified spectral transformer certificate*,
J4. Character recovery uses the unit group. The source vocabulary is all
residues, so the zero-numerator boundary is handled explicitly: 0^0=1,
and 0^k=0 for positive k. Division claims still require a nonzero denominator.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Spectral

open Expressivity

variable {p : ℕ} [Fact p.Prime]

/-- Interpret an original vocabulary token as its field residue. -/
def residue (a : Fin p) : ZMod p := ZMod.finEquiv p a

/-- Use a nonzero original token as the unit-group argument of a recovered kernel. -/
def tokenUnit (a : Fin p) (ha : residue a ≠ 0) : (ZMod p)ˣ := Units.mk0 (residue a) ha

/-- The zero-numerator answer, including the natural exponent zero. -/
def zeroAnswer (p : ℕ) [Fact p.Prime] (k : ℕ) : Fin p :=
  (ZMod.finEquiv p).symm (if k = 0 then 1 else 0)

/-- J4: the existing label computation is the original field quotient and power. -/
theorem powerLabel_residue (k : ℕ) (a b : Fin p) :
    residue (powerLabel p k a b) = (residue a / residue b) ^ k := by
  simp [residue, powerLabel]

/-- J4: the zero row is determined algebraically and needs no unit-kernel recovery. -/
theorem powerLabel_zero (k : ℕ) (a b : Fin p) (ha : residue a = 0) :
    powerLabel p k a b = zeroAnswer p k := by
  apply (ZMod.finEquiv p).injective
  change residue (powerLabel p k a b) = _
  rw [powerLabel_residue, ha]
  cases k <;> simp [zeroAnswer]

/-- J4: a quotient of nonzero inputs remains in the unit group at every exponent. -/
theorem powerLabel_nonzero (k : ℕ) (a b : Fin p) (ha : residue a ≠ 0) (hb : residue b ≠ 0) :
    residue (powerLabel p k a b) ≠ 0 := by
  rw [powerLabel_residue]
  exact pow_ne_zero k (div_ne_zero ha hb)

/-- J4: the unit-kernel answer and the original residue-label answer coincide. -/
theorem tokenUnit_power (k : ℕ) (a b : Fin p) (ha : residue a ≠ 0) (hb : residue b ≠ 0) :
    tokenUnit (powerLabel p k a b) (powerLabel_nonzero k a b ha hb) =
      (tokenUnit a ha / tokenUnit b hb) ^ k := by
  apply Units.ext
  simp only [tokenUnit, Units.val_mk0, Units.val_pow_eq_pow_val, Units.val_div_eq_div_val]
  exact powerLabel_residue k a b

/-- J4: the original token encoding retains distinct nonzero answers. -/
theorem tokenUnit_injective (a b : Fin p) (ha : residue a ≠ 0) (hb : residue b ≠ 0)
    (h : tokenUnit a ha = tokenUnit b hb) : a = b :=
  (ZMod.finEquiv p).injective (congrArg Units.val h)

example := powerLabel_zero (p := 3) 0 0 1 (by decide)
example := powerLabel_zero (p := 3) 5 0 2 (by decide)
example := powerLabel_nonzero (p := 3) 0 1 2 (by decide) (by decide)
example := tokenUnit_power (p := 3) 5 1 2 (by decide) (by decide)
example := tokenUnit_injective (p := 3) 1 1 (by decide) (by decide) rfl

end Transformer.SequenceModel.Attention.Spectral
