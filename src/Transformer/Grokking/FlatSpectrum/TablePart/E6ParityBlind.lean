module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6CosTable
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Fill

/-!
# PT6: M misses the table of `parity-cubic-blind`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The table is half of
`parity-cubic`, rounded down. Its kernel still correlates with the flat kernel of `a³ + a b` after a
turn by `ψ(24)`, and its seen cells lean towards `ψ(23)`. The kernel of Lean adds up the certified
lower bounds on the cosines over all 9409 cells, and `not_returned_of_pairing` closes the argument.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset

attribute [local instance] fact_prime_97

/-- The table of `parity-cubic-blind`: `parity-cubic` halved and rounded down. -/
def parityBlindTable (a b : ZMod 97) : ZMod 97 :=
  (((2 * ((a ^ 3 + a * b).val / 2) + (a.val + b.val) % 2) % 97 / 2 : ℕ) : ZMod 97)

/-- The kernel adds up the lower bounds on the pairing with `a³ + a b`, in units of `10⁻⁶`. -/
theorem parityBlind_pair : (5900 * 10 ^ 6 : ℤ) ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
    if (true : Bool) then cosLow (parityBlindTable a b - (a ^ 3 + a * b) + 24).val else 0 := by
  decide +kernel

/-- The kernel adds up the lower bounds on the lean of the seen cells, in units of `10⁻⁶`. -/
theorem parityBlind_seen : (2950 * 10 ^ 6 : ℤ) ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
    if halfMask.testBit (97 * a.val + b.val) then
      cosLow (parityBlindTable a b - 23).val else 0 := by
  decide +kernel

/-- An integer lower bound on a sum of table entries bounds the sum of the real parts. -/
lemma sum_re_ge {s : ℤ} (M : ZMod 97 → ZMod 97 → Bool) (e : ZMod 97 → ZMod 97 → ZMod 97)
    (h : s * 10 ^ 6 ≤ ∑ a : ZMod 97, ∑ b : ZMod 97, if M a b then cosLow (e a b).val else 0) :
    (s : ℝ) ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
      if M a b then (ZMod.stdAddChar (e a b) : ℂ).re else 0 := by
  have h1 : ((s * 10 ^ 6 : ℤ) : ℝ) ≤ ((∑ a : ZMod 97, ∑ b : ZMod 97,
      if M a b then cosLow (e a b).val else 0 : ℤ) : ℝ) := Int.cast_le.2 h
  push_cast at h1
  have h2 : (∑ a : ZMod 97, ∑ b : ZMod 97,
      if M a b then (cosLow (e a b).val : ℝ) else 0) / 10 ^ 6 ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
      if M a b then (ZMod.stdAddChar (e a b) : ℂ).re else 0 := by
    simp only [Finset.sum_div]
    refine Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun b _ => ?_
    by_cases hm : M a b = true
    · simpa only [hm, ↓reduceIte] using re_stdAddChar_ge_cosLow (e a b)
    · simp [hm]
  refine le_trans ?_ h2
  rw [le_div_iff₀ (by positivity)]
  norm_num
  exact h1

example : ((2950 : ℤ) : ℝ) ≤ ∑ a : ZMod 97, ∑ b : ZMod 97,
    if halfMask.testBit (97 * a.val + b.val) then
      (ZMod.stdAddChar (parityBlindTable a b - 23) : ℂ).re else 0 :=
  sum_re_ge (fun a b => halfMask.testBit (97 * a.val + b.val))
    (fun a b => parityBlindTable a b - 23) parityBlind_seen

/-- **M does not return the table of `parity-cubic-blind`** from the training half. -/
theorem not_returned_parityCubicBlind : NotReturned parityBlindTable := by
  have h1 := sum_re_ge (s := 5900) (fun _ _ => true)
    (fun a b => parityBlindTable a b - (a ^ 3 + a * b) + 24) parityBlind_pair
  have h2 := sum_re_ge (s := 2950) (fun a b => halfMask.testBit (97 * a.val + b.val))
    (fun a b => parityBlindTable a b - 23) parityBlind_seen
  simp only [↓reduceIte] at h1
  exact not_returned_of_pairing (by exact_mod_cast h1) (by exact_mod_cast h2)

example : NotReturned parityBlindTable := not_returned_parityCubicBlind

end Transformer.Grokking.FlatSpectrum.TablePart.E6
