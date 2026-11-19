module

public import Transformer.Sandwich.Section5_Extrapolation

/-!
# Limitations: parity

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, Limitations: "A model needs to predict whether a
bit string has an even or odd number of ones. For example, the parity of [1, 1, 0, 1] is odd (or 1)
and the parity of [1, 0, 1, 0] is even (or 0). Unlike human-written text, every single bit is
equally important. […] Its difficulty can be explained by the recency bias effect that we
described."

* `parity bits m`: the parity of the bits `0, …, m`, and the two examples (`parity_1101`,
  `parity_1010`).
* `parity_flip`, `parity_contributes`: flipping any bit flips the parity, so every bit of the
  prefix contributes: the receptive field of parity is the whole prefix.
* `Model.not_parity`: a model with windowed attention of size `w` and `R` layers does not predict
  the parity at a position `m ≥ R(w − 1) + 1`, whatever the encoding of the bits, position
  dependent (so with an APE) or not, and the decoding of the next-token distribution: the first
  bit lies outside its TRF.

This explains the difficulty for the windowed models of §4. A real bias, as Sandwich's, KERPLE's
or T5's, masks no key (`attended_coe` in `SectionB_EfficientInference`), so for those the paper's
recency bias is measured (§5.1), not a hard limit.
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset

variable {d k H v w : ℕ}

/-- The parity of the bits `0, …, m`: `1` if an odd number of them is `1`, else `0`. -/
def parity (bits : ℕ → Bool) (m : ℕ) : ℕ := (∑ j ∈ range (m + 1), (bits j).toNat) % 2

/-- The bit string `l` followed by zeros. -/
def bitsOf (l : List Bool) (j : ℕ) : Bool := l.getD j false

/-- **The parity of `[1, 1, 0, 1]` is odd (or `1`)** (Limitations). -/
theorem parity_1101 : parity (bitsOf [true, true, false, true]) 3 = 1 := by decide

/-- **The parity of `[1, 0, 1, 0]` is even (or `0`)** (Limitations). -/
theorem parity_1010 : parity (bitsOf [true, false, true, false]) 3 = 0 := by decide

/-- **Flipping a bit flips the parity**: "every single bit is equally important" (Limitations). -/
theorem parity_flip (bits : ℕ → Bool) {j m : ℕ} (hj : j ≤ m) :
    parity (Function.update bits j (!bits j)) m ≠ parity bits m := by
  have hmem : j ∈ range (m + 1) := mem_range.2 (by omega)
  have hrest : ∑ i ∈ (range (m + 1)).erase j, (Function.update bits j (!bits j) i).toNat =
      ∑ i ∈ (range (m + 1)).erase j, (bits i).toNat :=
    sum_congr rfl fun i hi => by rw [Function.update_of_ne (ne_of_mem_erase hi)]
  simp only [parity]
  rw [← add_sum_erase _ _ hmem, ← add_sum_erase _ _ hmem, hrest, Function.update_self]
  cases bits j <;> simp <;> omega

/-- The hypothesis of `parity_flip` holds for the first bit. -/
example (bits : ℕ → Bool) : parity (Function.update bits 0 (!bits 0)) 3 ≠ parity bits 3 :=
  parity_flip bits (Nat.zero_le 3)

/-- **Every bit of the prefix contributes to its parity**: the receptive field of parity at the
position `m` is all `m + 1` bits. -/
theorem parity_contributes {j m : ℕ} (hj : j ≤ m) : Contributes parity m j :=
  ⟨Function.update (fun _ => false) j (!false), fun _ => false,
    fun _ hi => Function.update_of_ne hi _ _, parity_flip (fun _ => false) hj⟩

/-- The hypothesis of `parity_contributes` holds for the first bit. -/
example (m : ℕ) : Contributes parity m 0 := parity_contributes (Nat.zero_le m)

/-- **A windowed model does not predict parity past its TRF** (Limitations): with windowed
attention of size `w ≥ 1` and `R` layers, no encoding `enc` of the bits, position dependent or
not, and no decoding `dec` of the next-token distribution give the parity of the bits `0, …, m`
at a position `m ≥ R(w − 1) + 1`: the first bit lies outside the TRF. -/
theorem Model.not_parity (M : Model d k H v) {b : Fin H → ℕ → EReal}
    (hb : ∀ h, IsWindowed w (b h)) (hw : 1 ≤ w) (enc : ℕ → Bool → Fin d → ℝ)
    (dec : (Fin v → ℝ) → ℕ) {m : ℕ} (hm : M.depth * (w - 1) + 1 ≤ m) :
    ¬∀ bits : ℕ → Bool, dec (M.predict b (fun j => enc j (bits j)) m) = parity bits m := by
  intro h
  obtain ⟨x, y, hxy, hne⟩ := parity_contributes (Nat.zero_le m)
  refine M.not_contributes hb hw (n := 0) (m := m) (by omega)
    ⟨fun j => enc j (x j), fun j => enc j (y j), fun j hj => congrArg (enc j) (hxy j hj),
      fun heq => hne ?_⟩
  rw [← h x, ← h y, heq]

/-- The hypotheses of `Model.not_parity` hold for windowed attention of size `100` at the end of
the TRF. -/
example (M : Model d k H v) (enc : ℕ → Bool → Fin d → ℝ) (dec : (Fin v → ℝ) → ℕ) :
    ¬∀ bits : ℕ → Bool, dec (M.predict (fun _ => windowBias 100) (fun j => enc j (bits j))
      (M.depth * 99 + 1)) = parity bits (M.depth * 99 + 1) :=
  M.not_parity (fun _ => isWindowed_windowBias 100) (by norm_num) enc dec (by norm_num)

end Transformer.Sandwich
