module

public import Transformer.Grokking.Convex.ConvexDepth.BoundedLatent
public import Init.Data.UInt.Lemmas

/-!
# Lossless 16-bit storage of bounded observed states

Source: `review/convex/moments.md`, CM68. Pack the observed state of every fixed
program and the original raw class. Explicit range bounds prevent truncation.
The decoder rejects out-of-domain words rather than applying a remainder.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.PackedRecords

/-- One original record keeps every program state and its raw class, CM68. -/
abbrev Row (J : Type*) (s c : ℕ) := (J → Fin s) × Fin c

/-- Packed observation words, CM68. No learned parameter is represented here. -/
abbrev Words (J : Type*) := (J → UInt16) × UInt16

/-- Actual bounded-integer conversion, CM68; its unrestricted operation can
truncate, so the round-trip theorems below require explicit range hypotheses. -/
def encode {J : Type*} {s c : ℕ} (r : Row J s c) : Words J :=
  (fun j => UInt16.ofNat (r.1 j).val, UInt16.ofNat r.2.val)

/-- Every stored word is in the declared state or raw-class domain, CM68. -/
def Valid {J : Type*} (s c : ℕ) (w : Words J) : Prop :=
  (∀ j, (w.1 j).toNat < s) ∧ w.2.toNat < c

/-- Decode validated words without reordering or coalescing records, CM68. -/
def decode {J : Type*} {s c : ℕ} (w : Words J) (h : Valid s c w) : Row J s c :=
  (fun j => ⟨(w.1 j).toNat, h.1 j⟩, ⟨w.2.toNat, h.2⟩)

/-- The range check is part of the decode interface, CM68. This logical option
specification does not assert a proof about Rust allocation or machine code. -/
noncomputable def checkedDecode {J : Type*} (s c : ℕ) (w : Words J) : Option (Row J s c) := by
  classical
  exact if h : Valid s c w then some (decode w h) else none

variable {J : Type*} {s c : ℕ}

/-- Each program's actual UInt16 round trip is lossless below its size, CM68. -/
theorem encoded_state (hs : s ≤ UInt16.size) (r : Row J s c) (j : J) :
    ((encode r).1 j).toNat = (r.1 j).val := by
  exact UInt16.toNat_ofNat_of_lt' (Nat.lt_of_lt_of_le (r.1 j).isLt hs)

/-- The raw label is packed independently of the state dimension, CM68. -/
theorem encoded_label (hc : c ≤ UInt16.size) (r : Row J s c) :
    (encode r).2.toNat = r.2.val := by
  exact UInt16.toNat_ofNat_of_lt' (Nat.lt_of_lt_of_le r.2.isLt hc)

/-- Encoding produces valid words under both declared range bounds, CM68. -/
theorem encode_valid (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size) (r : Row J s c) :
    Valid s c (encode r) := by
  constructor
  · intro j
    rw [encoded_state hs]
    exact (r.1 j).isLt
  · rw [encoded_label hc]
    exact r.2.isLt

/-- Every program state and the raw label survive, CM68. The conclusion is
record equality, not equality of a weaker feature or success predicate. -/
theorem decode_encode (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size) (r : Row J s c) :
    decode (encode r) (encode_valid hs hc r) = r := by
  apply Prod.ext
  · funext j
    apply Fin.ext
    exact encoded_state hs r j
  · apply Fin.ext
    exact encoded_label hc r

/-- Valid code words have no hidden representational aliases, CM68. -/
theorem encode_decode (w : Words J) (h : Valid s c w) : encode (decode w h) = w := by
  apply Prod.ext
  · funext j
    exact UInt16.ofNat_toNat
  · exact UInt16.ofNat_toNat

/-- A checked decode accepts every bounded original record, CM68. -/
theorem checkedDecode_encode (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size)
    (r : Row J s c) : checkedDecode s c (encode r) = some r := by
  classical
  unfold checkedDecode
  rw [dite_eq_left (encode_valid hs hc r)]
  exact congrArg some (decode_encode hs hc r)

/-- Invalid stored states or classes cause rejection, CM68; they cannot
silently become another valid training observation by modular reduction. -/
theorem checkedDecode_none_iff (w : Words J) :
    checkedDecode s c w = none ↔ ¬ Valid s c w := by
  classical
  unfold checkedDecode
  split_ifs with h
  · simp [h]
  · simp [h]

/-- Encoding cannot merge distinct original records within the admitted
ranges, CM68. Repeated equal records still retain their separate outer IDs. -/
theorem encode_injective (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size) :
    Function.Injective (encode : Row J s c → Words J) := by
  intro a b hab
  have h := congrArg (checkedDecode s c) hab
  rw [checkedDecode_encode hs hc, checkedDecode_encode hs hc] at h
  cases h
  rfl

/-- Actual native upper state bound and maximal word label, CM68. -/
def boundaryRow : Row Bool 4096 65536 :=
  (fun b => if b then 4095 else 1, 65535)

example := encoded_state (by decide : 4096 ≤ UInt16.size) boundaryRow true
example := encoded_label (by decide : 65536 ≤ UInt16.size) boundaryRow
example := encode_valid (by decide : 4096 ≤ UInt16.size)
  (by decide : 65536 ≤ UInt16.size) boundaryRow
example := decode_encode (by decide : 4096 ≤ UInt16.size)
  (by decide : 65536 ≤ UInt16.size) boundaryRow
example := encode_decode (encode boundaryRow) (encode_valid (by decide) (by decide) boundaryRow)
example := checkedDecode_encode (by decide : 4096 ≤ UInt16.size)
  (by decide : 65536 ≤ UInt16.size) boundaryRow
example := encode_injective (J := Bool) (by decide : 4096 ≤ UInt16.size)
  (by decide : 65536 ≤ UInt16.size)

end Transformer.Grokking.Convex.ConvexDepth.PackedRecords
