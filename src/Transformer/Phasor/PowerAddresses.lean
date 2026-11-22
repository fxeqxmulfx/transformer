module

public import Transformer.Phasor.PairValues

/-!
# Computing the shared table addresses from per-token residues

Source: the host address timing trial in `experiments/phasor_value_transport/README.md`.
The vocabulary, including composite sizes, fixes the first three power residues
of every input token. Each operand pair can reuse them in all thirteen power
tables. A bounded conditional gives the exact sum or difference representative.
The dense pair-address trial materializes whole address vectors and reuses
them before applying the history-dependent edits at each position.
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

/-- Row-major storage of one static address vector for every pair of tokens.
The value type can contain every component of the model's address row. -/
def pairAddressCache {κ : Type*} (p : ℕ) (address : ℕ → ℕ → κ) : Array κ :=
  Array.ofFn (fun i : Fin (p * p) => address (i.val / p) (i.val % p))

/-- The dense pair-address trial in the experiment README. Valid tokens give
an in-bounds address, and decoding its row and column recovers both operands
even for a composite vocabulary or a single-token vocabulary. -/
theorem pairAddressCache_read {κ : Type*} (address : ℕ → ℕ → κ) (a b : Fin p) :
    (pairAddressCache p address)[a.val * p + b.val]? = some (address a.val b.val) := by
  have bound : a.val * p + b.val < p * p := by
    simpa only [Nat.mul_comm p a.val] using Nat.mul_add_lt_mul_of_lt_of_lt a.isLt b.isLt
  have order : a.val * p + b.val = b.val + p * a.val := by ac_rfl
  have pos : 0 < p := Nat.zero_lt_of_lt b.isLt
  rw [pairAddressCache, Array.getElem?_ofFn, dite_eq_left bound]
  simp [order, Nat.add_mul_div_left _ _ pos, Nat.div_eq_of_lt b.isLt, Nat.mod_eq_of_lt b.isLt]

/-- The dense pair-address trial in the experiment README. A chronological
scan can patch its static addresses using the current tokens and any earlier
state. Every resulting state, including a log of all reads or gradient
owners in row order, agrees with a scan that computes the vectors directly. -/
theorem pairAddressCache_scan {κ σ : Type*} (address : ℕ → ℕ → κ)
    (update : σ → (Fin p × Fin p) → κ → σ) (initial : σ) (rows : List (Fin p × Fin p)) :
    rows.foldlM (m := Option) (fun s ab =>
      ((pairAddressCache p address)[ab.1.val * p + ab.2.val]?).map (update s ab)) initial =
    some (rows.foldl (fun s ab => update s ab (address ab.1.val ab.2.val)) initial) := by
  induction rows generalizing initial with
  | nil => rfl
  | cons pair rest ih =>
    rcases pair with ⟨a,b⟩
    simpa [pairAddressCache_read] using ih (update initial (a,b) (address a.val b.val))

/-- Repeated reads and exchanged operands at a composite vocabulary keep
their separate positions in the chronological output. -/
example :
    let pairs : List (Fin 6 × Fin 6) := [(5,1), (1,5), (5,1)]
    pairs.foldlM (m := Option) (fun (trace : List ℕ) ab => do
      let key ← (pairAddressCache 6 (fun a b => (a * a + b) % 6))[ab.1.val * 6 + ab.2.val]?
      return key :: trace) [] = some [2,0,2] := by decide

end Transformer.Phasor
