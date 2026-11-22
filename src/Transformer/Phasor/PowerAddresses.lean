module

public import Transformer.Phasor.PairValues

/-!
# Computing the shared table addresses from per-token residues

Source: the host address timing trial in `experiments/phasor_value_transport/README.md`.
The vocabulary, including composite sizes, fixes the first three power residues
of every input token. Each operand pair can reuse them in all thirteen power
tables. A bounded conditional gives the exact sum or difference representative.
-/

@[expose] public section

namespace Transformer.Phasor

variable {p : ℕ}

/-- The host computes each square and cube once, reducing after each product. -/
def tokenPowers (a : Fin p) : Fin 3 → ℕ :=
  let sq := a.val * a.val % p
  ![a.val, sq, sq * a.val % p]

/-- The cache contains precisely the three finite-ring powers used by all
table indices, including when the vocabulary size is composite. -/
theorem tokenPowers_eq [NeZero p] (a : Fin p) (i : Fin 3) :
    tokenPowers a i = (a ^ (i.val + 1)).val := by
  fin_cases i <;> simp [tokenPowers, pow_succ, Fin.val_mul]

/-- Each of the nine sum addresses uses a cached pair and a comparison with
the first wrapping address. Neither case needs integer division. -/
theorem PairKey.sumPowers_cached [NeZero p] (a b : Fin p) (i j : Fin 3) :
    ((PairKey.sumPowers i j).address a b).val =
      (if tokenPowers a i < p - tokenPowers b j then tokenPowers a i + tokenPowers b j
       else tokenPowers a i - (p - tokenPowers b j)) := by
  simpa only [PairKey.address, tokenPowers_eq] using
    transportedAddress_val (a ^ (i.val + 1)) (b ^ (j.val + 1))

/-- Both product indices select from the same per-token cache as the nine
sum tables. Multiplication still reduces modulo the vocabulary size. -/
theorem PairKey.productPowers_cached [NeZero p] (a b : Fin p) (i j : Fin 2) :
    ((PairKey.productPowers i j).address a b).val =
      tokenPowers a (i.castLE (by decide)) * tokenPowers b (j.castLE (by decide)) % p := by
  simp only [PairKey.address, tokenPowers_eq, Fin.val_castLE, Fin.val_mul]

/-- Subtract representatives directly when they are ordered; otherwise
subtract their positive distance from the vocabulary size. -/
theorem PairKey.difference_bounded [NeZero p] (a b : Fin p) :
    (PairKey.difference.address a b).val =
      if b.val ≤ a.val then a.val - b.val else p - (b.val - a.val) := by
  change (p - b.val + a.val) % p = _
  have ha := a.isLt
  have hb := b.isLt
  split
  · rw [Nat.mod_eq_sub_mod (by omega : p ≤ p - b.val + a.val),
      Nat.mod_eq_of_lt (by omega : p - b.val + a.val - p < p)]
    omega
  · rw [Nat.mod_eq_of_lt (by omega : p - b.val + a.val < p)]
    omega

/-- Both wrapping and non-wrapping sums and differences occur at a
composite vocabulary size, including a power that collapses to zero. -/
example :
    (tokenPowers (4 : Fin 8) 1, ((PairKey.sumPowers 0 0).address (5 : Fin 8) 6).val,
      ((PairKey.sumPowers 1 1).address (3 : Fin 8) 6).val,
      (PairKey.difference.address (1 : Fin 8) 6).val,
      (PairKey.difference.address (6 : Fin 8) 1).val) = (0, 3, 5, 3, 5) := by decide

end Transformer.Phasor
