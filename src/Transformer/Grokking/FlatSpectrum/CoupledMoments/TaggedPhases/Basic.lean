module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.BalancedFeatures

/-!
# A categorical tag for constant input strips

Source: `review/flat_spectrum.md`, CM25. Embed a proposed base phase in a
three-times-larger cyclic domain. One extra residue marks an absorbing input
category; two such marks do not cancel. No arithmetic meaning of input names
or discovery of the flags/decoder is assumed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases

variable {q : ℕ}

/-- The categorical marker used on each input axis, CM25. -/
def tag (flag : Bool) : ℕ := if flag then 1 else 0

/-- Put base phases on multiples of three and flagged inputs in residue one, CM25. -/
def lift (flag : Bool) (z : ZMod q) : ZMod (3 * q) :=
  ((3 * z.val + tag flag : ℕ) : ZMod (3 * q))

/-- Decode residue zero through the base decoder and other residues to a fixed
categorical output, CM25. The latent domain, not the input vocabulary, changes. -/
def decoder {C : Type*} (constant : C) (d : ZMod q → C) (z : ZMod (3 * q)) : C :=
  if z.val % 3 = 0 then d ((z.val / 3 : ℕ) : ZMod q) else constant

/-- The actual modular sum retains both categorical marks, including wraparound,
CM25. It is not a presumed latent addition rule for the observed labels. -/
theorem lift_sum_val (a b : ZMod q) (f g : Bool) :
    (lift f a + lift g b).val =
      (3 * (a.val + b.val) + tag f + tag g) % (3 * q) := by
  unfold lift
  rw [← Nat.cast_add, ZMod.val_natCast]
  congr 1
  omega

/-- Neither a single flag nor two flags can be mistaken for an unflagged sum,
CM25. The extra factor three avoids cancellation of two binary marks. -/
theorem tag_zero_iff (a b : ZMod q) (f g : Bool) :
    (lift f a + lift g b).val % 3 = 0 ↔ f = false ∧ g = false := by
  rw [lift_sum_val, Nat.mod_mul_right_mod, Nat.add_assoc, Nat.mul_add_mod]
  cases f <;> cases g <;> norm_num [tag]

variable [NeZero q]

/-- The enlarged decoder implements an arbitrary base table with constant
flagged rows/columns, including their intersection, CM25. -/
theorem decode_lift {C : Type*} (constant : C) (d : ZMod q → C)
    (a b : ZMod q) (f g : Bool) :
    decoder constant d (lift f a + lift g b) =
      if f || g then constant else d (a + b) := by
  cases f <;> cases g
  · simp only [decoder, (tag_zero_iff a b false false).mpr ⟨rfl, rfl⟩,
      ↓reduceIte, Bool.false_or, Bool.false_eq_true]
    rw [lift_sum_val]
    simp [tag, Nat.mod_mul_right_div_self, Nat.cast_add]
  · have h : (lift false a + lift true b).val % 3 ≠ 0 := by
      intro h
      have h' := (tag_zero_iff a b false true).mp h
      cases h'.2
    simp [decoder, h]
  · have h : (lift true a + lift false b).val % 3 ≠ 0 := by
      intro h
      have h' := (tag_zero_iff a b true false).mp h
      cases h'.1
    simp [decoder, h]
  · have h : (lift true a + lift true b).val % 3 ≠ 0 := by
      intro h
      have h' := (tag_zero_iff a b true true).mp h
      cases h'.1
    simp [decoder, h]

example := lift_sum_val (4 : ZMod 5) 3 true true
example := tag_zero_iff (4 : ZMod 5) 3 true true
example := decode_lift (false : Bool) (fun z : ZMod 5 => decide (z = 2)) 4 3 false false
example := decode_lift (false : Bool) (fun z : ZMod 5 => decide (z = 2)) 4 3 true true

end Transformer.Grokking.FlatSpectrum.CoupledMoments.TaggedPhases
