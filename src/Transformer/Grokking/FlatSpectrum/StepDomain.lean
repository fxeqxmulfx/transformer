module

public import Transformer.Grokking.FlatSpectrum.StepCarry

/-!
# The decoder holds the level: the wrap of a step is the domain of the series

Source: `review/flat_spectrum.md`, *The decoder holds the level* (added), item DL1. A
difference encoding of the recurrence `x ↦ (b + x) % p` writes the step `modStep p b x`, and
the decoder adds it to the level `x`. The decoder holds `x`; the model reads steps and does
not.

* The step is the only integer with the residue of `b` that keeps the series in its domain,
  `0 ≤ x + d < p` (`eq_modStep_iff`). So the wrap of CW6 is a function of the residue and of
  the level: a decoder that admits only values of the domain needs the residue alone.
* The other integers of that residue, `modStep p b x + k p` with `k ≠ 0`, leave the domain
  (`modStep_add_mul_not_mem`): a wrong wrap is never an admitted answer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

/-- **DL1**, `review/flat_spectrum.md`: the step from the level `x` is the only integer with
the residue of `b` that keeps the series in its domain `[0, p)`. -/
theorem eq_modStep_iff {p : ℕ} (hp : 0 < p) (b x : ℕ) (d : ℤ) :
    d = modStep p b x ↔
      (d : ZMod p) = (b : ZMod p) ∧ 0 ≤ (x : ℤ) + d ∧ (x : ℤ) + d < p := by
  have hr : (b + x) % p < p := Nat.mod_lt _ hp
  constructor
  · rintro rfl
    refine ⟨intCast_modStep p b x, ?_, ?_⟩ <;> unfold modStep <;>
      generalize (b + x) % p = r at hr ⊢ <;> omega
  · rintro ⟨hd, h0, h1⟩
    obtain ⟨y, hy⟩ := Int.eq_ofNat_of_zero_le h0
    have hcast : ((y : ℕ) : ZMod p) = ((b + x : ℕ) : ZMod p) := by
      have h : ((y : ℤ) : ZMod p) = (((x : ℤ) + d : ℤ) : ZMod p) := by rw [hy]
      push_cast at h ⊢
      rw [h, hd]
      ring
    rw [ZMod.natCast_eq_natCast_iff', Nat.mod_eq_of_lt (by omega)] at hcast
    unfold modStep
    omega

/-- Modulo `97`, from the level `60` after the value `50`: the step `−47`, to the level `13`. -/
example : (-47 : ℤ) = modStep 97 50 60 :=
  (eq_modStep_iff (by norm_num) 50 60 (-47)).2 ⟨by decide, by norm_num, by norm_num⟩

/-- **DL1**, `review/flat_spectrum.md`: the other integers of the residue of the step leave
the domain. A wrong wrap, `k = ±1`, is never a value of the series. -/
theorem modStep_add_mul_not_mem {p : ℕ} (hp : 0 < p) (b x : ℕ) {k : ℤ} (hk : k ≠ 0) :
    ¬(0 ≤ (x : ℤ) + (modStep p b x + k * p) ∧ (x : ℤ) + (modStep p b x + k * p) < p) := by
  rintro ⟨h0, h1⟩
  have h := (eq_modStep_iff hp b x (modStep p b x + k * p)).2
    ⟨by push_cast; simp [intCast_modStep], h0, h1⟩
  have hzero : k * (p : ℤ) = 0 := by linarith
  rcases mul_eq_zero.1 hzero with h' | h'
  · exact hk h'
  · omega

/-- Modulo `97`, from the level `60` after `50`: the other wrap `−47 + 97 = 50` leads to `110`. -/
example : ¬(0 ≤ ((60 : ℕ) : ℤ) + (modStep 97 50 60 + 1 * (97 : ℕ)) ∧
    ((60 : ℕ) : ℤ) + (modStep 97 50 60 + 1 * (97 : ℕ)) < (97 : ℕ)) :=
  modStep_add_mul_not_mem (by norm_num) 50 60 one_ne_zero

end Transformer.Grokking.FlatSpectrum
