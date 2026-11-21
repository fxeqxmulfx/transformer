module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6SumWitnesses
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Sums

/-!
# PT6: the eleven invertible tables of `a + b`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The tasks `(a + b)^k`
for `k = 3, 4, 6, 8, 12, 16, 24, 32, 48`, the interval `a + b ≤ 47` and the nonzero squares, which
by Euler's criterion are the `x` with `x ^ 48 = 1`. Each kernel is invertible by its vector in
`E6SumWitnesses`, and each anti-diagonal has a cell outside the training half, so M returns none.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

attribute [local instance] fact_prime_97

/-- M does not return `(a + b)^k` when `ConvCheck` holds for it. -/
theorem not_returned_sum_pow (k : ℕ) {W : List ℕ} (h : ConvCheck (fun s => s ^ k % 97) 0 W) :
    NotReturned fun a b => (a + b) ^ k := by
  have h0 : NotReturned fun a b : ZMod 97 => (((a + b).val ^ k % 97 : ℕ) : ZMod 97) :=
    not_returned_sum (e := fun s => s ^ k % 97) h
  have hfun : (fun a b : ZMod 97 => (((a + b).val ^ k % 97 : ℕ) : ZMod 97)) =
      fun a b => (a + b) ^ k := by
    funext a b
    rw [ZMod.natCast_mod, Nat.cast_pow, ZMod.natCast_zmod_val]
  rwa [hfun] at h0

example := not_returned_sum_pow 3 checkPow3

theorem not_returned_sumPow3 : NotReturned fun a b => (a + b) ^ 3 :=
  not_returned_sum_pow 3 checkPow3

theorem not_returned_sumPow4 : NotReturned fun a b => (a + b) ^ 4 :=
  not_returned_sum_pow 4 (W := wPow4) (by decide +kernel)

theorem not_returned_sumPow6 : NotReturned fun a b => (a + b) ^ 6 :=
  not_returned_sum_pow 6 (W := wPow6) (by decide +kernel)

theorem not_returned_sumPow8 : NotReturned fun a b => (a + b) ^ 8 :=
  not_returned_sum_pow 8 (W := wPow8) (by decide +kernel)

theorem not_returned_sumPow12 : NotReturned fun a b => (a + b) ^ 12 :=
  not_returned_sum_pow 12 (W := wPow12) (by decide +kernel)

theorem not_returned_sumPow16 : NotReturned fun a b => (a + b) ^ 16 :=
  not_returned_sum_pow 16 (W := wPow16) (by decide +kernel)

theorem not_returned_sumPow24 : NotReturned fun a b => (a + b) ^ 24 :=
  not_returned_sum_pow 24 (W := wPow24) (by decide +kernel)

theorem not_returned_sumPow32 : NotReturned fun a b => (a + b) ^ 32 :=
  not_returned_sum_pow 32 (W := wPow32) (by decide +kernel)

theorem not_returned_sumPow48 : NotReturned fun a b => (a + b) ^ 48 :=
  not_returned_sum_pow 48 (W := wPow48) (by decide +kernel)

/-- **M does not return the interval task**, `1` when the residue of `a + b` is at most `47`. -/
theorem not_returned_sumInterval :
    NotReturned fun a b => if (a + b).val ≤ 47 then 1 else 0 := by
  have h : ConvCheck (fun s => if s ≤ 47 then 1 else 0) 0 wInterval := by decide +kernel
  have h0 : NotReturned fun a b : ZMod 97 =>
      ((if (a + b).val ≤ 47 then 1 else 0 : ℕ) : ZMod 97) :=
    not_returned_sum (e := fun s => if s ≤ 47 then 1 else 0) h
  have hfun : (fun a b : ZMod 97 => ((if (a + b).val ≤ 47 then 1 else 0 : ℕ) : ZMod 97)) =
      fun a b => if (a + b).val ≤ 47 then 1 else 0 := by
    funext a b
    split_ifs <;> simp
  rwa [hfun] at h0

/-- **M does not return the task of nonzero squares**, `1` when `(a + b) ^ 48 = 1`. -/
theorem not_returned_sumSquares :
    NotReturned fun a b => if (a + b) ^ 48 = 1 then 1 else 0 := by
  have h : ConvCheck (fun s => if s ^ 48 % 97 = 1 then 1 else 0) 0 wSquares := by decide +kernel
  have h0 : NotReturned fun a b : ZMod 97 =>
      ((if (a + b).val ^ 48 % 97 = 1 then 1 else 0 : ℕ) : ZMod 97) :=
    not_returned_sum (e := fun s => if s ^ 48 % 97 = 1 then 1 else 0) h
  have key : ∀ x : ZMod 97, x.val ^ 48 % 97 = 1 ↔ x ^ 48 = 1 := by decide +kernel
  have hfun : (fun a b : ZMod 97 =>
      ((if (a + b).val ^ 48 % 97 = 1 then 1 else 0 : ℕ) : ZMod 97)) =
      fun a b => if (a + b) ^ 48 = 1 then 1 else 0 := by
    funext a b
    by_cases hx : (a + b) ^ 48 = 1
    · simp [hx, (key _).2 hx]
    · simp [hx, mt (key _).1 hx]
  rwa [hfun] at h0

end Transformer.Grokking.FlatSpectrum.TablePart.E6
