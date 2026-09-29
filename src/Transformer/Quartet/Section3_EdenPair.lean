/-
# A vector whose rotation takes two values

The counterexample of `Transformer.Quartet.Section3_EdenBias` to the Corollary of §3.3 at
*every* dimension `d = 16 · 2^k` uses the vector `x = e₀ + (1/10) e₁` (`edenPair`): the first
two coordinates of the first group, `1` and `1/10`, and zeros elsewhere.  Its randomized
Hadamard transform, for every sign seed `ε`, has only two values, by parity of the position:

`RHT(x)_p = (σ₀ + σ₁/10) / √d` at even `p`, `(σ₀ − σ₁/10) / √d` at odd `p`,

where `σ₀`, `σ₁ ∈ {±1}` are the signs the seed gives the two nonzero entries.  The reason is
that the Walsh character of the second column is `(-1)^{bit 0 of p}` (`walsh_four_one`), and
the character of the first column is `1`.

Source: arXiv:2601.22813v2, §3.2 (the `RHT`); the vector is ours.
-/

import Transformer.Quartet.Hadamard

namespace Transformer
namespace Quartet

variable {k : ℕ}

/-- The Walsh character of the column `1` on four bits is the sign of the lowest bit of the
row. -/
theorem walsh_four_one (j : ℕ) : walsh 4 j 1 = if j % 2 = 0 then 1 else -1 := by
  have e1 : Nat.testBit 1 1 = false := by decide
  have e2 : Nat.testBit 1 2 = false := by decide
  have e3 : Nat.testBit 1 3 = false := by decide
  simp only [walsh_succ, walsh_zero_bits, e1, e2, e3, Bool.and_false, Bool.false_eq_true,
    ite_false, mul_one, one_mul, Nat.testBit_zero]
  rcases Nat.mod_two_eq_zero_or_one j with h | h <;> simp [h]

/-- `e₀ + (1/10) e₁`: a tensor of `2^k` groups of `16`, supported on the first two entries of
the first group, where it is `1` and `1/10`. -/
noncomputable def edenPair (k : ℕ) : Fin (2 ^ k) → Fin 16 → ℝ :=
  fun i j => if i = 0 ∧ j = 0 then 1 else if i = 0 ∧ j = 1 then 1 / 10 else 0

/-- **The rotation of `edenPair` takes two values**, `(σ₀ ± σ₁/10)/√d` by the parity of the
position, with `σ₀`, `σ₁` the signs the seed gives the two nonzero entries. -/
theorem rht_edenPair (ε : Fin (2 ^ k) → Fin 16 → Bool) :
    rht k ε (edenPair k) = fun (_ : Fin (2 ^ k)) (j : Fin 16) =>
      if (j : ℕ) % 2 = 0 then
        ((if ε 0 0 then (-1 : ℝ) else 1) + (if ε 0 1 then (-1 : ℝ) else 1) / 10) /
          Real.sqrt (2 ^ (k + 4))
      else ((if ε 0 0 then (-1 : ℝ) else 1) - (if ε 0 1 then (-1 : ℝ) else 1) / 10) /
          Real.sqrt (2 ^ (k + 4)) := by
  funext i j
  unfold rht
  rw [Finset.sum_eq_single (0 : Fin (2 ^ k)) (fun b _ hb => ?_) (by simp)]
  · rw [Finset.sum_eq_add_of_mem (0 : Fin 16) 1 (Finset.mem_univ _) (Finset.mem_univ _)
      (by decide) (fun c _ hc => by simp [edenPair, hc.1, hc.2])]
    simp only [hadamard_eq, edenPair, Fin.val_zero, Fin.val_one, walsh_zero_right, walsh_four_one]
    rcases Nat.mod_two_eq_zero_or_one (j : ℕ) with h | h <;> cases ε 0 0 <;> cases ε 0 1 <;>
      simp [h] <;> ring
  · exact Finset.sum_eq_zero fun j' _ => by simp [edenPair, hb]

/-- The rotation formula is a genuine identity at the smallest dimension, for the all-`false`
seed: `RHT(e₀ + e₁/10)` has the two values `11/40` and `9/40`. -/
example : rht 0 (fun _ _ => false) (edenPair 0) 0 0 = 11 / 40 ∧
    rht 0 (fun _ _ => false) (edenPair 0) 0 1 = 9 / 40 := by
  have h4 : √((2 : ℝ) ^ (0 + 4)) = 4 := by
    rw [show ((2 : ℝ) ^ (0 + 4)) = 4 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]
  rw [rht_edenPair, h4]
  norm_num

end Quartet
end Transformer
