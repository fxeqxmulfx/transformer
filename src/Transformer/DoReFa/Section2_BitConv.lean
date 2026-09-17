/-
# Dot products by bitwise kernels

arXiv:1606.06160v3, Zhou, Wu, Ni, Zhou, Wen, Zou — "DoReFa-Net: Training Low
Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients", §1 and
§2.1.

The whole point of low bitwidth training is that a dot product of two bit
vectors is a `bitcount` of an `and` (equation 1), and that a dot product of
fixed-point integers decomposes into `M · K` such kernels, one per pair of bit
planes (equation 3) — "the computation complexity is `O(MK)`, i.e., directly
proportional to bitwidth".

**A sign slip in footnote 2.**  For vectors of `{-1, 1}` the paper writes
`x · y = N - 2 · bitcount(xnor(x, y))`.  The count there has to be of the
positions where the two vectors *disagree*, which is `xor`, not `xnor`:
`dot_sign_eq` below is the identity, and `exists_ne_bitcount_xnor` shows that
the displayed form fails already for one bit.
-/

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.Ring.Int
import Mathlib.Tactic.NormNum

namespace Transformer
namespace DoReFa

variable {n M K : ℕ}

/-- A bit as the integer `0` or `1`, the `x_i ∈ {0, 1}` of equation 1. -/
def bitVal (b : Bool) : ℤ := if b then 1 else 0

/-- A bit as the integer `-1` or `1`, the `x_i ∈ {-1, 1}` of footnote 2. -/
def signVal (b : Bool) : ℤ := if b then 1 else -1

/-- `bitcount`, "the number of bits in a bit vector" (§1). -/
def bitcount (x : Fin n → Bool) : ℕ := (Finset.univ.filter fun i => x i = true).card

/-- **Equation 1**: the dot product of two bit vectors is a `bitcount` of an
`and` — the kernel every other statement of the paper reduces to. -/
theorem dot_eq_bitcount (x y : Fin n → Bool) :
    ∑ i, bitVal (x i) * bitVal (y i) = (bitcount fun i => x i && y i : ℤ) :=
  sorry

/-- **Footnote 2, corrected**: for vectors of `{-1, 1}` the dot product is
`N - 2 ·` the number of positions where they disagree. -/
theorem dot_sign_eq (x y : Fin n → Bool) :
    ∑ i, signVal (x i) * signVal (y i) = (n : ℤ) - 2 * (bitcount fun i => xor (x i) (y i) : ℤ) :=
  sorry

/-- **Footnote 2 as displayed is false**: counting the positions where the two
vectors *agree*, which is what `xnor` does, gives the negated dot product. -/
theorem exists_ne_bitcount_xnor :
    ∃ (m : ℕ) (x y : Fin m → Bool),
      ∑ i, signVal (x i) * signVal (y i) ≠
        (m : ℤ) - 2 * (bitcount fun i => !(xor (x i) (y i)) : ℤ) := by
  refine ⟨1, fun _ => true, fun _ => true, ?_⟩
  norm_num [signVal, bitcount]

/-- **Equation 3**: the dot product of a sequence of `M`-bit fixed-point
integers with a sequence of `K`-bit ones, `x · y = ∑_m ∑_k 2^{m+k}
bitcount[and(c_m(x), c_k(y))]`, where `c_m(x)` is the `m`-th bit plane. -/
theorem dot_eq_sum_bitcount (cx : Fin M → Fin n → Bool) (cy : Fin K → Fin n → Bool) :
    (∑ i, (∑ m : Fin M, bitVal (cx m i) * 2 ^ (m : ℕ)) *
        ∑ l : Fin K, bitVal (cy l i) * 2 ^ (l : ℕ)) =
      ∑ m : Fin M, ∑ l : Fin K,
        2 ^ ((m : ℕ) + (l : ℕ)) * (bitcount fun i => cx m i && cy l i : ℤ) :=
  sorry

/-- **The cost of equation 3 is `M · K` kernels**: "the computation complexity
is `O(MK)`, i.e., directly proportional to bitwidth of `x` and `y`" (§2.1).
The right-hand side above has one `bitcount` per pair of bit planes. -/
theorem card_bitcount_calls (M K : ℕ) :
    (Finset.univ : Finset (Fin M × Fin K)).card = M * K := by
  simp

end DoReFa
end Transformer
