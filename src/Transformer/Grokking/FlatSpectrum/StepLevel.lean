module

public import Transformer.Grokking.FlatSpectrum.StepWrap

/-!
# Levels and steps: what an encoder asks of the model

Source: `review/flat_spectrum.md`, *The encoder: steps and levels* (added), items EN1 and EN2.
A difference encoding of the recurrence `x ↦ (b + x) % p` gives the model the steps between
the values, and asks for the step to write. A level encoding gives it the values and asks for
the value.

* The level to write is the entry of the kernel of `a + b` (`addKernel_natCast`), and the
  entry determines it (`stdAddChar_eq_addKernel_iff`): the level task is the kernel of
  Proposition 2 and Theorem E, with nothing beside its residue.
* A step is not determined by its residue: after a value `0 < b < p` both `b` and `b − p` are
  written (`modStep_zero_and_pred`), and the residue of both is `b` (`intCast_modStep`).
* A step `d` is read at `p − |d|` of the `p²` states, as the first step `b − a`
  (`card_filter_sub_eq`) and as the second (`card_filter_modStep_eq`): the steps near `±p`
  are read at few states. The two levels read at a state, `b` and `a + b`, run over all pairs
  once (`levelsRead`), so a level is read at `p` states, first (`card_levelsRead_fst`) and
  second (`card_levelsRead_snd`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

/-- **EN1**, `review/flat_spectrum.md`: the level to write after the levels `b` and `x` is the
entry of the kernel of `a + b`. -/
theorem addKernel_natCast {p : ℕ} [NeZero p] (b x : ℕ) :
    addKernel p (b : ZMod p) (x : ZMod p) = ZMod.stdAddChar (((b + x) % p : ℕ) : ZMod p) := by
  simp [addKernel, taskKernel]

/-- **EN1**, `review/flat_spectrum.md`: and the entry determines the level: no other level
`y < p` has the phase of the entry. -/
theorem stdAddChar_eq_addKernel_iff {p : ℕ} [NeZero p] (b x : ℕ) {y : ℕ} (hy : y < p) :
    ZMod.stdAddChar (y : ZMod p) = addKernel p (b : ZMod p) (x : ZMod p) ↔
      y = (b + x) % p := by
  rw [addKernel_natCast, ZMod.injective_stdAddChar.eq_iff, ZMod.natCast_eq_natCast_iff',
    Nat.mod_mod, Nat.mod_eq_of_lt hy]

/-- Modulo `97`, after the levels `50` and `60`: the level `13`. -/
example : ZMod.stdAddChar ((13 : ℕ) : ZMod 97) = addKernel 97 ((50 : ℕ) : ZMod 97) (60 : ℕ) ↔
    13 = (50 + 60) % 97 :=
  stdAddChar_eq_addKernel_iff 50 60 (by norm_num)

/-- **EN1**, `review/flat_spectrum.md`: a step is not determined by its residue. After a value
`0 < b < p` the step is `b` from `x = 0` and `b − p` from `x = p − 1`. -/
theorem modStep_zero_and_pred {p b : ℕ} (hb0 : 0 < b) (hb : b < p) :
    modStep p b 0 = b ∧ modStep p b (p - 1) = b - p := by
  constructor
  · rw [modStep_eq hb (by omega)]
    split_ifs <;> omega
  · rw [modStep_eq hb (by omega)]
    split_ifs <;> omega

example : modStep 97 50 0 = 50 ∧ modStep 97 50 96 = 50 - 97 :=
  modStep_zero_and_pred (by norm_num) (by norm_num)

/-- **EN2**, `review/flat_spectrum.md`: the first step `b − a` is `d` at `p − |d|` of the
`p²` pairs `(a, b)`. -/
theorem card_filter_sub_eq (p : ℕ) (d : ℤ) :
    ((range p ×ˢ range p).filter fun s : ℕ × ℕ => (s.2 : ℤ) - s.1 = d).card =
      p - d.natAbs := by
  have key : ((range p ×ˢ range p).filter fun s : ℕ × ℕ => (s.2 : ℤ) - s.1 = d) =
      (range (p - d.natAbs)).image fun i => (i + (-d).toNat, i + d.toNat) := by
    ext ⟨a, b⟩
    simp only [mem_filter, mem_product, mem_range, mem_image, Prod.mk.injEq]
    constructor
    · rintro ⟨⟨ha, hb⟩, h⟩
      exact ⟨min a b, by omega, by omega, by omega⟩
    · rintro ⟨i, hi, rfl, rfl⟩
      exact ⟨⟨by omega, by omega⟩, by omega⟩
  rw [key, card_image_of_injective _ fun i j h => by have := (Prod.mk.inj h).1; omega,
    card_range]

/-- **EN2**, `review/flat_spectrum.md`: the second step `modStep p a b` is `d` at `p − |d|`
of the `p²` pairs `(a, b)`. -/
theorem card_filter_modStep_eq (p : ℕ) (d : ℤ) :
    ((range p ×ˢ range p).filter fun s : ℕ × ℕ => modStep p s.1 s.2 = d).card =
      p - d.natAbs := by
  have key : ((range p ×ˢ range p).filter fun s : ℕ × ℕ => modStep p s.1 s.2 = d) =
      (range (p - d.natAbs)).image fun i =>
        ((if 0 ≤ d then d else d + p).toNat, i + (-d).toNat) := by
    ext ⟨a, b⟩
    simp only [mem_filter, mem_product, mem_range, mem_image, Prod.mk.injEq]
    constructor
    · rintro ⟨⟨ha, hb⟩, h⟩
      rw [modStep_eq ha hb] at h
      refine ⟨b - (-d).toNat, ?_, ?_, ?_⟩ <;> split_ifs at h ⊢ <;> omega
    · rintro ⟨i, hi, rfl, rfl⟩
      have ha : (if 0 ≤ d then d else d + p).toNat < p := by split_ifs <;> omega
      have hb : i + (-d).toNat < p := by omega
      refine ⟨⟨ha, hb⟩, ?_⟩
      rw [modStep_eq ha hb]
      split_ifs <;> omega
  rw [key, card_image_of_injective _ fun i j h => by have := (Prod.mk.inj h).2; omega,
    card_range]

/-- Modulo `97` the step `−96` is read at one pair, and the step `0` at `97`. -/
example : ((range 97 ×ˢ range 97).filter fun s : ℕ × ℕ => modStep 97 s.1 s.2 = -96).card = 1 :=
  card_filter_modStep_eq 97 (-96)

/-- **EN2**, `review/flat_spectrum.md`: the two levels read at the state `(a, b)` are `b` and
`a + b`, and they run over all pairs of levels once. -/
def levelsRead (p : ℕ) : ZMod p × ZMod p ≃ ZMod p × ZMod p where
  toFun s := (s.2, s.1 + s.2)
  invFun t := (t.2 - t.1, t.1)
  left_inv s := by simp
  right_inv t := by simp

/-- **EN2**, `review/flat_spectrum.md`: a level is the first level read at `p` states. -/
theorem card_levelsRead_fst (p : ℕ) [NeZero p] (y : ZMod p) :
    Fintype.card {s : ZMod p × ZMod p // (levelsRead p s).1 = y} = p := by
  rw [Fintype.card_congr (⟨fun s => s.1.1, fun a => ⟨(a, y), rfl⟩, fun s => by
    obtain ⟨⟨a, b⟩, h⟩ := s
    simp only [levelsRead, Equiv.coe_fn_mk] at h
    subst h
    rfl, fun _ => rfl⟩ : {s : ZMod p × ZMod p // (levelsRead p s).1 = y} ≃ ZMod p), ZMod.card]

/-- **EN2**, `review/flat_spectrum.md`: and the second level read at `p` states. -/
theorem card_levelsRead_snd (p : ℕ) [NeZero p] (y : ZMod p) :
    Fintype.card {s : ZMod p × ZMod p // (levelsRead p s).2 = y} = p := by
  rw [Fintype.card_congr (⟨fun s => s.1.2, fun b => ⟨(y - b, b), by simp [levelsRead]⟩,
    fun s => by
      obtain ⟨⟨a, b⟩, h⟩ := s
      simp only [levelsRead, Equiv.coe_fn_mk] at h
      subst h
      simp, fun _ => rfl⟩ : {s : ZMod p × ZMod p // (levelsRead p s).2 = y} ≃ ZMod p),
    ZMod.card]

end Transformer.Grokking.FlatSpectrum
