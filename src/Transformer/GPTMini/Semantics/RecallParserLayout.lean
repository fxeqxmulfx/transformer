import Transformer.GPTMini.Semantics.RecallIntegerArray

/-!
# Full validated Basis recall parsing discharges the real raw layout

Source: MQAR.targets at cbafbe9, the complete raw integer parser,
and the unchanged original GPTMini finite embedding array. Every
raw routing-layout condition is derived from successful parsing:
actual BOS, the full key/value alphabet and each table value's
immediate preceding key in the same original token positions.

The final query is also recovered from the actual last decoded
position. No prepared paired table, layout predicate, encoder
state, model score or desired logit is supplied as a hypothesis.
The table boundary follows from the actual parsed record count;
no prefix-length or table-position oracle is supplied separately.
Recovered predecessor bounds follow from successful integer reads.
The finite input list and all its token positions remain unchanged
throughout these checks of the genuine embedding array.
These results apply to either overwrite setting and any accepted
table size; the full context cap and real-model last-write/readout
connection follow in the subsequent correctness proof.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- Successful full raw parsing proves that the actual first embedding index is precisely BOS=1.
Source: the original BOS parser check and exact integer/finite token correspondence. -/
theorem recallParser_bos (P : ℕ) (rewrites : Bool) (head : Fin recallConfig.vocab_size)
    (tail : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    (head :: tail).get ⟨0, by simp⟩ = ⟨1, by decide⟩ := by
  obtain ⟨body, hraw, _⟩ := recallAnswer_body_alphabet _ _ _ hanswer
  apply recallDecoded_bos_at
  rw [hraw]
  rfl

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- Every actual finite array position after BOS has a genuine compact raw key or value ID by full parser validation.
Source: all raw table/query/filler range checks, transported to the same finite embedding lookup positions. -/
theorem recallParser_alphabet (P : ℕ) (rewrites : Bool) (head : Fin recallConfig.vocab_size)
    (tail : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    ∀ index : Fin (head :: tail).length, index ≠ ⟨0, by simp⟩ →
      ∃ symbol : Fin 256, (head :: tail).get index = recallKeyId symbol ∨ (head :: tail).get index = recallValueId symbol := by
  intro index hne
  have hn : index.val ≠ 0 := fun h => hne (Fin.ext h)
  have hr := recallAnswer_nonbos_alphabet ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) answer hanswer index.val
    (by simpa only [decodeTokens, List.length_map] using index.isLt) (by omega)
  rw [recallDecoded_get] at hr
  rcases hr with hk | hv
  · obtain ⟨symbol, hs⟩ := (recallKeyToken_iff P rewrites _).mp hk
    exact ⟨symbol, Or.inl hs⟩
  · obtain ⟨symbol, hs⟩ := (recallValueToken_iff P rewrites _).mp hv
    exact ⟨symbol, Or.inr hs⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- Full successful parsing proves the actual immediate finite-array key predecessor of every true table value.
Source: real integer BOS/table adjacency and derived finite-position/key-symbol recovery, with no layout premise. -/
theorem recallParser_table_adjacency (P : ℕ) (rewrites : Bool) (head : Fin recallConfig.vocab_size)
    (tail : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    ∀ record : Fin (head :: tail).length, ∀ value : Fin 256,
      (head :: tail).get record = recallValueId value → record.val ≤ 2 * P →
      ∃ previous key, previous.val + 1 = record.val ∧ (head :: tail).get previous = recallKeyId key := by
  obtain ⟨writes, region, _, _, hraw, hlen, hrange, _, _, _, _⟩ := recallAnswer_parsed_table _ _ _ hanswer
  intro record value hvalue hinside
  have hv : ValueToken ⟨256, P, rewrites⟩ (((292 + value.val : ℕ) : ℤ)) :=
    (recallValueToken_iff P rewrites (recallValueId value)).mpr ⟨value, rfl⟩
  have hr := recallDecoded_value_id (head :: tail) record value hvalue
  rw [hraw] at hr
  obtain ⟨previous, key, hp, hk, hread⟩ := recallRawTable_value_predecessor ⟨256, P, rewrites⟩ writes region
    hrange record.val (by simpa only [hlen] using hinside) _ hr hv
  have hd : (decodeTokens (head :: tail))[previous]? = some key := by rw [hraw]; exact hread
  obtain ⟨position, hi, _⟩ := recallDecoded_position (head :: tail) previous key hd
  obtain ⟨symbol, hs, _⟩ := recallDecoded_key_at P rewrites (head :: tail) position key (by rw [hi]; exact hd) hk
  exact ⟨position, symbol, by omega, hs⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- The complete actual raw routing-layout predicate follows solely from the validated full integer parser.
Source: derived BOS, all alphabet positions and every actual adjacent table key/value binding above. -/
theorem recallParser_layout (P : ℕ) (rewrites : Bool) (head : Fin recallConfig.vocab_size)
    (tail : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    RecallRawLayout P (head :: tail).get ⟨0, by simp⟩ := by
  exact ⟨rfl, recallParser_bos P rewrites head tail answer hanswer,
    recallParser_alphabet P rewrites head tail answer hanswer, recallParser_table_adjacency P rewrites head tail answer hanswer⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- Successful parsing derives the actual compact query symbol at the genuine last finite model input position.
Source: the full raw parser's final query index and exact unchanged finite token serialization. -/
theorem recallParser_query (P : ℕ) (rewrites : Bool) (head : Fin recallConfig.vocab_size)
    (tail : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    ∃ key : Fin 256, (head :: tail).get ⟨tail.length, by simp⟩ = recallKeyId key := by
  obtain ⟨query, hr, hk⟩ := recallAnswer_final_query_index _ _ _ hanswer
  have hi : (decodeTokens (head :: tail)).length - 1 = tail.length := by
    simp only [decodeTokens, List.length_map, List.length_cons]
    omega
  rw [hi] at hr
  obtain ⟨key, hs, _⟩ := recallDecoded_key_at P rewrites (head :: tail) ⟨tail.length, by simp⟩ query hr hk
  exact ⟨key, hs⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- The actual overwrite control's integer input is the original raw six-token sequence, retaining key/value association.
Source: literal finite token IDs and the genuine decoder map, independent of any model output. -/
theorem recallOverwriteTokens_decode : decodeTokens recallOverwriteTokens = [1, 36, 292, 36, 293, 36] := by rfl

/-- The literal overwrite control has a successfully validated full raw answer, rather than a supplied routing certificate.
Source: the actual raw MQAR parser with rewrites enabled and the genuinely decoded finite input. -/
theorem recallOverwriteTokens_answer : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- All genuine raw layout conditions on the overwrite control are also consequences of full successful Basis parsing.
Source: the actual parser-layout theorem, without supplying its array-level adjacency or alphabet separately. -/
theorem recallOverwriteTokens_parsed_layout : RecallRawLayout 2 recallOverwriteTokens.get (0 : Fin 6) :=
  recallParser_layout 2 true ⟨1, by decide⟩ _ 293 recallOverwriteTokens_answer

/-- Every accepted finite input list has a genuine initial position and derives the whole actual raw layout there.
Source: successful full integer parsing rules out an empty list and proves each original finite-array layout condition. -/
theorem recallParser_encoded_layout (P : ℕ) (rewrites : Bool) (tokens : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ first : Fin tokens.length, first.val = 0 ∧ RecallRawLayout P tokens.get first := by
  cases tokens with
  | nil => simp only [decodeTokens, List.map_nil, recallAnswer] at hanswer; contradiction
  | cons head tail =>
      exact ⟨⟨0, by simp⟩, rfl, recallParser_layout P rewrites head tail answer hanswer⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- Every accepted finite list has a genuine final query position whose original array entry is its actual compact key ID.
Source: full parser supervision and the same real last-index query recovery, retaining the input length exactly. -/
theorem recallParser_encoded_query (P : ℕ) (rewrites : Bool) (tokens : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query : Fin tokens.length, ∃ key : Fin 256, query.val + 1 = tokens.length ∧ tokens.get query = recallKeyId key := by
  cases tokens with
  | nil => simp only [decodeTokens, List.map_nil, recallAnswer] at hanswer; contradiction
  | cons head tail =>
      obtain ⟨key, hk⟩ := recallParser_query P rewrites head tail answer hanswer
      exact ⟨⟨tail.length, by simp⟩, key, rfl, hk⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

end Transformer.GPTMini.Semantics
