import Transformer.GPTMini.Convex.Structured.Binding
import Transformer.GPTMini.Convex.Structured.RecallPositions
import Transformer.GPTMini.Convex.Structured.SharedInterface

/-!
# One finite shared raw matching/value/binding weight assignment

Source: the actual unrestricted all-pair binding operator at f37438a,
raw table positions at 9e6661b and MQAR's original 256-key/256-value
alphabet. Four base-four digits give independently learned Q/K witness
fields; five output digits give actual jointly trainable value fields.
The same finite table is used for every prefix, key and overwrite.

These codes, physical table slots and adjacency preferences specify
given weights and data targets only. The genuine operator reads free
raw fields and sums every visible pair, without calling these semantic
witness constructors. Its complete training objective remains convex
throughout the entire unrestricted shared/relative parameter domain.

For a positive finite gain, matching/value and exclusion gains are 65
times that gain, while chronology uses the gain itself. Initial/state
emission coordinates are zero in this pointer witness and its learned
head logits favor the pointer branch. Energy gaps, full raw recall
decoding and genuine tensor-stack realization are subsequent proofs.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped Classical
noncomputable section

/-- A bounded token-local symbol code for given matching weights; actual key IDs recover their complete 256-symbol index.
Source: MQAR's key offset 36 and a finite modulo-256 shift, independent of every input prefix. -/
def recallBindingSymbol (token : Fin 548) : Fin 256 :=
  ⟨(token.val + 220) % 256, Nat.mod_lt _ (by decide)⟩

/-- The actual raw key vocabulary maps to its precise symbol in the finite matching witness.
Source: original keyId=36+symbol and exact modulo-256 arithmetic, without an assumed encoded match. -/
theorem recallBindingSymbol_key (key : Fin 256) : recallBindingSymbol (recallKeyId key) = key := by
  apply Fin.ext
  change (36 + key.val + 220) % 256 = key.val
  omega

/-- Four complete categorical digits specify given shared Q/K matching fields, not a fixed forward interaction bank.
Source: the exact 256-symbol radix representation at a7d5e0f and the actual raw token-local witness code. -/
def recallBindingDigits (token : Fin 548) : Fin 4 → Fin 4 := recallDigit (recallBindingSymbol token)

/-- Actual key IDs retain all four of their independent matching digits through the witness table.
Source: precise raw symbol recovery and the same compact radix digit function. -/
theorem recallBindingDigits_key (key : Fin 256) : recallBindingDigits (recallKeyId key) = recallDigit key := by
  unfold recallBindingDigits
  rw [recallBindingSymbol_key]

/-- Every distinct real raw key has distinct matching digits, including the complete hard/easy key alphabet.
Source: the actual four-digit injectivity theorem, with no sample-dependent code assignment. -/
theorem recallBindingDigits_different (left right : Fin 256) (hne : left ≠ right) :
    recallBindingDigits (recallKeyId left) ≠ recallBindingDigits (recallKeyId right) := by
  rw [recallBindingDigits_key, recallBindingDigits_key]
  intro he
  exact hne (recallDigit_injective he)

example : (0 : Fin 256) ≠ 255 := by decide

/-- Given output-value weights specify the actual raw token ID's five decoder digits.
Source: the whole-vocabulary output code and genuine finite 548-to-1024 ID inclusion. -/
def recallBindingOutput (token : Fin 548) : Fin 5 → Fin 4 := outputDigit (vocabularyCode (by decide) token)

/-- Every one of the actual 52 token-local free fields receives its finite given Q, K or value logit.
Source: disjoint 16/16/20 slot arithmetic; this constructor is a weight witness, absent from learned inference. -/
def recallBindingFields (gain : ℝ) (token : Fin 548) (slot : Fin 52) : ℝ :=
  if hq : slot.val < 16 then
    sharpRowLogits (recallBindingDigits token ⟨slot.val / 4, by omega⟩) (65 * gain) ⟨slot.val % 4, Nat.mod_lt _ (by decide)⟩
  else if hk : slot.val < 32 then
    sharpRowLogits (recallBindingDigits token ⟨(slot.val - 16) / 4, by omega⟩) (65 * gain)
      ⟨(slot.val - 16) % 4, Nat.mod_lt _ (by decide)⟩
  else
    sharpRowLogits (recallBindingOutput token ⟨(slot.val - 32) / 4, by omega⟩) (65 * gain)
      ⟨(slot.val - 32) % 4, Nat.mod_lt _ (by decide)⟩

/-- One finite shared actual parameter assignment for every raw prefix of a given recall table size.
Source: learned token fields, free absolute/relative positional potentials, chronology and two-head logits; no per-input weights exist. -/
def recallBindingParameters (P : ℕ) (gain : ℝ) : BindingParameters 548 64 :=
  (fun field => match field with
    | .inl (token, slot) => recallBindingFields gain token slot
    | .inr (.inl position) => if recallTableValuePosition P position.val then 0 else -(65 * gain)
    | .inr (.inr (.inl ())) => gain
    | .inr (.inr (.inr (.inl _))) => 0
    | .inr (.inr (.inr (.inr (.inl _)))) => 0
    | .inr (.inr (.inr (.inr (.inr head)))) => sharpRowLogits (1 : Fin 2) gain head,
   fun offset => if offset.val = 64 then 0 else -(65 * gain))

/-- Actual query coordinate reads evaluate to their finite learned matching witness logits.
Source: genuine sharedQuery, the given full parameter table and exact first-sixteen slot division/remainder. -/
theorem recallBinding_query (P : ℕ) (gain : ℝ) (token : Fin 548) (g c : Fin 4) :
    sharedQuery (recallBindingParameters P gain).1 token g c = sharpRowLogits (recallBindingDigits token g) (65 * gain) c := by
  have hg := g.isLt
  have hc := c.isLt
  have hi : 4 * g.val + c.val < 16 := by omega
  have hd : (4 * g.val + c.val) / 4 = g.val := by omega
  have hm : (4 * g.val + c.val) % 4 = c.val := by omega
  change recallBindingFields gain token (querySlot g c) = _
  simp only [recallBindingFields, querySlot, hi, dite_true, hd, hm]

/-- Actual key coordinate reads retain their independent finite learned matching logits.
Source: genuine sharedKey and exact disjoint second-sixteen slot arithmetic, without a matching-probability premise. -/
theorem recallBinding_key (P : ℕ) (gain : ℝ) (token : Fin 548) (g c : Fin 4) :
    sharedKey (recallBindingParameters P gain).1 token g c = sharpRowLogits (recallBindingDigits token g) (65 * gain) c := by
  have hg := g.isLt
  have hc := c.isLt
  have hi : ¬16 + 4 * g.val + c.val < 16 := by omega
  have hj : 16 + 4 * g.val + c.val < 32 := by omega
  have hd : (16 + 4 * g.val + c.val - 16) / 4 = g.val := by omega
  have hm : (16 + 4 * g.val + c.val - 16) % 4 = c.val := by omega
  change recallBindingFields gain token (keySlot g c) = _
  simp only [recallBindingFields, keySlot, hi, dite_false, hj, dite_true, hd, hm]

/-- Actual jointly trainable value coordinates encode the raw candidate token's full decoder ID in the given witness.
Source: genuine sharedValue and exact final-twenty field indexing, with no externally supplied correct value mean. -/
theorem recallBinding_value (P : ℕ) (gain : ℝ) (token : Fin 548) (h : Fin 5) (d : Fin 4) :
    sharedValue (recallBindingParameters P gain).1 token h d = sharpRowLogits (recallBindingOutput token h) (65 * gain) d := by
  have hh := h.isLt
  have hd := d.isLt
  have hi : ¬32 + 4 * h.val + d.val < 16 := by omega
  have hj : ¬32 + 4 * h.val + d.val < 32 := by omega
  have hdiv : (32 + 4 * h.val + d.val - 32) / 4 = h.val := by omega
  have hmod : (32 + 4 * h.val + d.val - 32) % 4 = d.val := by omega
  change recallBindingFields gain token (valueSlot h d) = _
  simp only [recallBindingFields, valueSlot, hi, hj, dite_false, hdiv, hmod]

/-- The actual free absolute-position field realizes table exclusion only as a finite given weight choice.
Source: the full parameter witness, not a fixed mask inside the all-pair inference operator. -/
theorem recallBinding_position (P : ℕ) (gain : ℝ) (position : Fin 64) :
    (recallBindingParameters P gain).1 (.inr (.inl position)) =
      if recallTableValuePosition P position.val then 0 else -(65 * gain) := by
  unfold recallBindingParameters
  rfl

/-- The actual free chronology coordinate has the finite gain shared by every raw route.
Source: the full learned parameter witness, independent of key, value or prefix. -/
theorem recallBinding_chronology (P : ℕ) (gain : ℝ) :
    (recallBindingParameters P gain).1 (.inr (.inr (.inl ()))) = gain := by
  unfold recallBindingParameters
  rfl

/-- The learned relative field's finite witness favors actual physical successors without removing any other candidate.
Source: the complete relative table and exact signed-displacement index equivalence. -/
theorem recallBinding_relative (P : ℕ) (gain : ℝ) (key value : Fin 64) :
    (recallBindingParameters P gain).2 (bindingRelativeIndex key value) =
      if key.val + 1 = value.val then 0 else -(65 * gain) := by
  change (if (bindingRelativeIndex key value).val = 64 then (0 : ℝ) else -(65 * gain)) = _
  simp only [bindingRelative_adjacent]

/-- The same genuine shared global head coordinates give a finite learned preference for the pointer branch.
Source: the full freely trainable two-head parameter assignment, with actual mixed-head correctness still separate. -/
theorem recallBinding_head (P : ℕ) (gain : ℝ) (head : Fin 2) :
    (recallBindingParameters P gain).1 (.inr (.inr (.inr (.inr (.inr head))))) = sharpRowLogits (1 : Fin 2) gain head := by
  unfold recallBindingParameters
  rfl

end
end Transformer.GPTMini.Convex.Structured
