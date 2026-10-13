import Transformer.GPTMini.Semantics.RecallConstruction
import Transformer.Basis.RecallPrefixPositions

/-!
# Exact integer/finite-index coupling for the real recall input

Source: MQAR's key/value intervals and original token IDs at cbafbe9,
and the checked GPTMini List Int adapter. Successful integer indexing
is connected to precisely the same finite token read by the original
embedding lookup, retaining all positions and all 548 vocabulary IDs.

Key/value validity is equivalent to the actual alphabet interval
represented by the existing compact symbol IDs. Raw integer reads can
therefore recover a genuine finite key or value symbol. A successful
optional raw read also derives its finite position bound; the bound
is not supplied as an additional parser premise. These facts concern
only input representation and will discharge the actual raw routing
conditions. They do not replace any model operator or supply a desired
hidden state, prepared score or answer.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- The complete original finite key alphabet is exactly the validated raw integer key interval.
Source: MQAR identityBase=36, symbols=256 and the actual checked key ID, in either overwrite recipe. -/
theorem recallKeyToken_iff (P : ℕ) (rewrites : Bool) (token : Fin recallConfig.vocab_size) :
    KeyToken ⟨256, P, rewrites⟩ (token.val : ℤ) ↔ ∃ symbol : Fin 256, token = recallKeyId symbol := by
  constructor
  · intro h
    dsimp [KeyToken, identityBase] at h
    let symbol : Fin 256 := ⟨token.val - 36, by omega⟩
    refine ⟨symbol, ?_⟩
    apply Fin.ext
    change token.val = 36 + (token.val - 36)
    omega
  · rintro ⟨symbol, rfl⟩
    dsimp [KeyToken, identityBase, recallKeyId]
    have h := symbol.isLt
    omega

/-- Every genuine value or filler ID is exactly an entry of the separate actual finite value alphabet.
Source: MQAR's validated interval 292..547 and the existing checked value ID definition. -/
theorem recallValueToken_iff (P : ℕ) (rewrites : Bool) (token : Fin recallConfig.vocab_size) :
    ValueToken ⟨256, P, rewrites⟩ (token.val : ℤ) ↔ ∃ symbol : Fin 256, token = recallValueId symbol := by
  constructor
  · intro h
    dsimp [ValueToken, identityBase] at h
    let symbol : Fin 256 := ⟨token.val - 292, by omega⟩
    refine ⟨symbol, ?_⟩
    apply Fin.ext
    change token.val = 292 + (token.val - 292)
    omega
  · rintro ⟨symbol, rfl⟩
    dsimp [ValueToken, identityBase, recallValueId]
    have h := symbol.isLt
    omega

/-- Decoding preserves the exact integer token at every actual bounded finite input position.
Source: the real adapter's per-token Nat-to-Int map, with no modulo reduction or reordered prefix. -/
theorem recallDecoded_index (tokens : List (Fin recallConfig.vocab_size)) (index : Fin tokens.length) :
    (decodeTokens tokens)[index.val]? = some ((tokens.get index).val : ℤ) := by
  simp only [decodeTokens, List.getElem?_map, List.getElem?_eq_getElem index.isLt, Option.map_some]
  rfl

/-- The same bounded raw read is the exact finite input token's integer ID through dependent list indexing.
Source: genuine decode map and the original finite list position, not a semantic feature array. -/
theorem recallDecoded_get (tokens : List (Fin recallConfig.vocab_size)) (index : Fin tokens.length) :
    (decodeTokens tokens)[index.val]'(by simpa only [decodeTokens, List.length_map] using index.isLt) =
      ((tokens.get index).val : ℤ) := by
  simp only [decodeTokens, List.getElem_map]
  rfl

/-- An actual successful integer read at a finite input position identifies that same token in the genuine model array.
Source: exact checked integer/finite correspondence, without an assumed embedding interpretation. -/
theorem recallDecoded_index_token (tokens : List (Fin recallConfig.vocab_size)) (index : Fin tokens.length) (token : ℤ)
    (hread : (decodeTokens tokens)[index.val]? = some token) : ((tokens.get index).val : ℤ) = token :=
  Option.some.inj ((recallDecoded_index tokens index).symm.trans hread)

example : (decodeTokens [recallKeyId 0])[0]? = some 36 := by decide

/-- A successful arbitrary raw integer read derives its own exact bounded finite input position.
Source: optional list indexing's success condition and the genuine length-preserving decode map. -/
theorem recallDecoded_position (tokens : List (Fin recallConfig.vocab_size)) (index : ℕ) (token : ℤ)
    (hread : (decodeTokens tokens)[index]? = some token) :
    ∃ position : Fin tokens.length, position.val = index ∧ ((tokens.get position).val : ℤ) = token := by
  obtain ⟨hi, _⟩ := List.getElem?_eq_some_iff.mp hread
  have hpos : index < tokens.length := by simpa only [decodeTokens, List.length_map] using hi
  exact ⟨⟨index, hpos⟩, rfl, recallDecoded_index_token tokens ⟨index, hpos⟩ token hread⟩

example : (decodeTokens [recallValueId 1])[0]? = some 293 := by decide

/-- A validated actual integer key read produces the precise compact key symbol used by the original embedding table.
Source: raw interval validation and exact finite-array decoding at that same position. -/
theorem recallDecoded_key_at (P : ℕ) (rewrites : Bool) (tokens : List (Fin recallConfig.vocab_size))
    (index : Fin tokens.length) (key : ℤ) (hread : (decodeTokens tokens)[index.val]? = some key)
    (hkey : KeyToken ⟨256, P, rewrites⟩ key) :
    ∃ symbol : Fin 256, tokens.get index = recallKeyId symbol ∧ ((36 + symbol.val : ℕ) : ℤ) = key := by
  have he := recallDecoded_index_token tokens index key hread
  obtain ⟨symbol, hs⟩ := (recallKeyToken_iff P rewrites (tokens.get index)).mp (he ▸ hkey)
  refine ⟨symbol, hs, ?_⟩
  rw [hs] at he
  exact he

example : (decodeTokens [recallKeyId 0])[0]? = some 36 ∧ KeyToken easyRecall 36 := by decide

/-- A validated actual integer value read produces precisely the independent value symbol in the original tied table.
Source: real raw value/filler interval and exact finite input position, with no selected-value feature input. -/
theorem recallDecoded_value_at (P : ℕ) (rewrites : Bool) (tokens : List (Fin recallConfig.vocab_size))
    (index : Fin tokens.length) (value : ℤ) (hread : (decodeTokens tokens)[index.val]? = some value)
    (hvalue : ValueToken ⟨256, P, rewrites⟩ value) :
    ∃ symbol : Fin 256, tokens.get index = recallValueId symbol ∧ ((292 + symbol.val : ℕ) : ℤ) = value := by
  have he := recallDecoded_index_token tokens index value hread
  obtain ⟨symbol, hs⟩ := (recallValueToken_iff P rewrites (tokens.get index)).mp (he ▸ hvalue)
  refine ⟨symbol, hs, ?_⟩
  rw [hs] at he
  exact he

example : (decodeTokens [recallValueId 1])[0]? = some 293 ∧ ValueToken hardRecall 293 := by decide

/-- The actual raw BOS integer read recovers exactly token one of the complete model vocabulary.
Source: original reserved vocabulary ID and exact decode correspondence, not a supplied BOS embedding state. -/
theorem recallDecoded_bos_at (tokens : List (Fin recallConfig.vocab_size)) (index : Fin tokens.length)
    (hread : (decodeTokens tokens)[index.val]? = some bos) : tokens.get index = ⟨1, by decide⟩ := by
  have he := recallDecoded_index_token tokens index bos hread
  apply Fin.ext
  change (tokens.get index).val = 1
  change ((tokens.get index).val : ℤ) = 1 at he
  exact_mod_cast he

example : (decodeTokens [(⟨1, by decide⟩ : Fin recallConfig.vocab_size)])[0]? = some bos := by decide

/-- An actual finite key-array entry decodes to its precise original integer key ID at the same raw position.
Source: unchanged input encoding and the actual key ID serializer, used to transfer true array matching back to raw chronology. -/
theorem recallDecoded_key_id (tokens : List (Fin recallConfig.vocab_size)) (index : Fin tokens.length) (symbol : Fin 256)
    (hkey : tokens.get index = recallKeyId symbol) :
    (decodeTokens tokens)[index.val]? = some ((36 + symbol.val : ℕ) : ℤ) := by
  rw [recallDecoded_index, hkey]
  rfl

example : ([recallKeyId 0]).get (0 : Fin 1) = recallKeyId 0 := by rfl

/-- Every actual finite value-array entry retains its original disjoint raw integer value ID at precisely that position.
Source: genuine decode map and checked value serializer, without an assumed paired-value representation. -/
theorem recallDecoded_value_id (tokens : List (Fin recallConfig.vocab_size)) (index : Fin tokens.length) (symbol : Fin 256)
    (hvalue : tokens.get index = recallValueId symbol) :
    (decodeTokens tokens)[index.val]? = some ((292 + symbol.val : ℕ) : ℤ) := by
  rw [recallDecoded_index, hvalue]
  rfl

example : ([recallValueId 1]).get (0 : Fin 1) = recallValueId 1 := by rfl

end Transformer.GPTMini.Semantics
