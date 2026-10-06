import Mathlib.Data.List.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Tactic

/-!
# Basis tasks on lists of integer tokens

New interface for python/src/lab/domain/basis.py and the raw-input oracles
in infrastructure/benchmarks/synthetic/, at commit cbafbe9. An oracle reads
an input prefix and its wrapper appends exactly one answer token. Depth
and recall use this wrapper at their supervised prefix positions; parity
uses it to emit an answer and then EOS.

The token IDs are the actual vocabulary.py IDs. Invalid inputs have a
specified PAD answer rather than Python's exception, making the interface
total on List Int. Task correctness will be stated on valid prefixes.
Encoding into Fin V checks both bounds, so a negative integer is never
silently turned into vocabulary token zero.
-/

namespace Transformer.Basis

/-- The requested raw-token interface; Source: the new Basis continuation convention. -/
abbrev Tokens := List ℤ

/-- Reserved vocabulary entries used by the three Basis tasks.
Source: synthetic/vocabulary.py at cbafbe9; data symbols start separately at ID 36. -/
inductive ReservedToken where
  | pad
  | bos
  | letterA
  | letterB
  | neutral
  | reject
  | accept
  | eos
  | sep
  | zeroBit
  | oneBit
  | even
  | odd
  deriving DecidableEq

/-- The actual integer IDs of the supported reserved vocabulary entries.
Source: synthetic/vocabulary.py at cbafbe9; each supported reserved symbol has its own integer ID. -/
abbrev ReservedToken.id : ReservedToken → ℤ
  | .pad => 0
  | .bos => 1
  | .letterA => 9
  | .letterB => 10
  | .neutral => 11
  | .reject => 15
  | .accept => 16
  | .eos => 17
  | .sep => 18
  | .zeroBit => 21
  | .oneBit => 22
  | .even => 24
  | .odd => 25

/-- PAD's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def pad : ℤ := ReservedToken.pad.id
/-- BOS's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def bos : ℤ := ReservedToken.bos.id
/-- The active depth letter a; Source: synthetic/vocabulary.py at cbafbe9. -/
def letterA : ℤ := ReservedToken.letterA.id
/-- The active depth letter b; Source: synthetic/vocabulary.py at cbafbe9. -/
def letterB : ℤ := ReservedToken.letterB.id
/-- The neutral depth letter; Source: synthetic/vocabulary.py at cbafbe9. -/
def neutral : ℤ := ReservedToken.neutral.id
/-- REJECT's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def reject : ℤ := ReservedToken.reject.id
/-- ACCEPT's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def accept : ℤ := ReservedToken.accept.id
/-- EOS's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def eos : ℤ := ReservedToken.eos.id
/-- The answer separator; Source: synthetic/vocabulary.py at cbafbe9. -/
def sep : ℤ := ReservedToken.sep.id
/-- Zero-bit's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def zeroBit : ℤ := ReservedToken.zeroBit.id
/-- One-bit's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def oneBit : ℤ := ReservedToken.oneBit.id
/-- EVEN's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def evenToken : ℤ := ReservedToken.even.id
/-- ODD's actual ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def oddToken : ℤ := ReservedToken.odd.id
/-- The first key/symbol ID; Source: synthetic/vocabulary.py at cbafbe9. -/
def identityBase : ℤ := 36

/-- Append one oracle answer to the complete unchanged input prefix.
Source: the new List Int to List Int formulation requested for Basis. -/
def extend (next : Tokens → ℤ) (tokens : Tokens) : Tokens := tokens ++ [next tokens]

/-- The extension contract increases every input length by exactly one.
Source: the new interface convention; no valid-input assumption is needed. -/
theorem extend_length (next : Tokens → ℤ) (tokens : Tokens) :
    (extend next tokens).length = tokens.length + 1 := by
  rw [extend, List.length_append, List.length_singleton]

/-- The input prefix is retained byte for byte, including on invalid inputs.
Source: the new Basis integer-list interface's append-only convention. -/
theorem extend_prefix (next : Tokens → ℤ) (tokens : Tokens) :
    (extend next tokens).take tokens.length = tokens := by
  simp [extend]

/-- Two extensions agree exactly when their appended answer tokens agree.
Source: the new interface; this relates task semantics to the model's discrete prediction. -/
theorem extend_eq_iff (f g : Tokens → ℤ) (tokens : Tokens) :
    extend f tokens = extend g tokens ↔ f tokens = g tokens := by
  simp [extend]

/-- Decode finite vocabulary indices back to their actual integer IDs.
Source: the new adapter to GPTMini.Model.forward, whose inputs are Fin-indexed. -/
def decodeTokens {V : ℕ} (tokens : List (Fin V)) : Tokens :=
  tokens.map fun t => (t.val : ℤ)

/-- Checked encoding into the model vocabulary, rejecting negative and out-of-range IDs.
Source: the new GPTMini adapter; Python rejects such embedding indices too. -/
def encodeTokens (V : ℕ) : Tokens → Option (List (Fin V))
  | [] => some []
  | t :: rest =>
      if h : 0 ≤ t ∧ t < (V : ℤ) then
        match encodeTokens V rest with
        | some encoded => some (⟨t.toNat, by omega⟩ :: encoded)
        | none => none
      else none

/-- Every finite token sequence round-trips through the checked integer representation.
Source: the new adapter's exact token IDs, without truncation or modulo reduction. -/
theorem encode_decode {V : ℕ} (tokens : List (Fin V)) :
    encodeTokens V (decodeTokens tokens) = some tokens := by
  induction tokens with
  | nil => rfl
  | cons t rest ih =>
      have ht : 0 ≤ (t.val : ℤ) ∧ (t.val : ℤ) < (V : ℤ) := by
        constructor
        · omega
        · exact_mod_cast t.isLt
      simp only [decodeTokens] at ih
      simp [decodeTokens, encodeTokens, ht, ih]

/-- Successful checked encoding retains all original integer tokens exactly.
Source: the new adapter; the check prevents Int.toNat's negative-input collapse. -/
theorem decode_encode {V : ℕ} {tokens : Tokens} {encoded : List (Fin V)}
    (h : encodeTokens V tokens = some encoded) : decodeTokens encoded = tokens := by
  induction tokens generalizing encoded with
  | nil => simpa [encodeTokens, decodeTokens] using h.symm
  | cons t rest ih =>
      simp only [encodeTokens] at h
      split at h
      · rename_i ht
        cases he : encodeTokens V rest with
        | none => simp [he] at h
        | some tail =>
            simp only [he, Option.some.injEq] at h
            subst encoded
            simp only [decodeTokens, List.map_cons]
            rw [Int.toNat_of_nonneg ht.1]
            exact congrArg (List.cons t) (ih he)
      · contradiction

example : encodeTokens 26 [1, 22, 18] =
    some [⟨1, by omega⟩, ⟨22, by omega⟩, ⟨18, by omega⟩] := by decide

/-- Successful encoding also preserves sequence length.
Source: the exact integer/finite-index round-trip, used for model position alignment. -/
theorem encode_length {V : ℕ} {tokens : Tokens} {encoded : List (Fin V)}
    (h : encodeTokens V tokens = some encoded) : encoded.length = tokens.length := by
  have he := congrArg List.length (decode_encode h)
  simpa only [decodeTokens, List.length_map] using he

example : encodeTokens 26 [1, 22, 18] =
    some [⟨1, by omega⟩, ⟨22, by omega⟩, ⟨18, by omega⟩] := by decide

/-- A negative ID is rejected even though Int.toNat would turn it into zero.
Source: the new checked adapter's lower bound, a meaningful invalid-input control. -/
theorem encode_negative : encodeTokens 26 [-1, 22, 18] = none := by decide

/-- An ID equal to the vocabulary size is rejected rather than wrapped around.
Source: the new checked adapter's strict upper bound, matching embedding lookup. -/
theorem encode_out_of_range : encodeTokens 26 [1, 26, 18] = none := by decide

/-- The PAD answer itself is a legal vocabulary token at Basis's smallest vocabulary.
Source: vocabulary.py's reserved IDs; PAD makes invalid-input continuations total. -/
theorem pad_encodable : encodeTokens 26 [pad] = some [⟨0, by omega⟩] := by decide

end Transformer.Basis
