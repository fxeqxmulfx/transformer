import Transformer.GPTMini.Semantics.RecallParserLatest

/-!
# Raw Basis recall positions for the freely learned binding head

Source: MQAR.targets/sample at cbafbe9 and the independently verified
complete raw parser at 75c8c94. This module derives the actual table
value slots, their neighboring keys and the latest matching write from
successful raw parsing. Every physical position remains in the input;
post-table fillers are not accidentally treated as table records.

These facts describe the data, not a proposed head's inference rule.
The learned-binding forward considers all visible position pairs with
free positional potentials. The arithmetic table predicate below is
used only to construct finite parameter witnesses and training labels.
No attention scores, correctly encoded states, selected model routes
or desired logits are premises of the full selected-record theorem.
Both rewrite settings and every successfully parsed record count are
covered; finite energy gaps and actual learned inference remain separate.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.Semantics

/-- The physical BOS-offset positions occupied by actual table values.
Source: MQAR's two-token chronological records; this data predicate is not a forward routing mask. -/
def recallTableValuePosition (pairs position : ℕ) : Prop :=
  0 < position ∧ position ≤ 2 * pairs ∧ position % 2 = 0

/-- The raw position predicate is computably checked from the actual two integer indices.
Source: MQAR's fixed BOS-offset serialization; this instance does not restrict learned forward routes. -/
instance (pairs position : ℕ) : Decidable (recallTableValuePosition pairs position) :=
  inferInstanceAs (Decidable (0 < position ∧ position ≤ 2 * pairs ∧ position % 2 = 0))

/-- Full parsing and a physical even table slot derive its real neighboring key and value at the unchanged raw positions.
Source: exact validated record indexing after BOS, transported through the genuine finite/integer serialization. -/
theorem recallBinding_table_pair (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer)
    (previous record : Fin tokens.length) (hprevious : previous.val + 1 = record.val)
    (hslot : recallTableValuePosition P record.val) :
    ∃ key value : Fin 256, tokens.get previous = recallKeyId key ∧ tokens.get record = recallValueId value := by
  obtain ⟨writes, region, _, _, hraw, hlen, hrange, _, _, _, _⟩ :=
    recallAnswer_parsed_table ⟨256, P, rewrites⟩ (decodeTokens tokens) answer hanswer
  change writes.length = P at hlen
  rcases hslot with ⟨hpos, hinside, heven⟩
  let row : Fin writes.length := ⟨record.val / 2 - 1, by omega⟩
  have hp : previous.val = 2 * row.val + 1 := by dsimp only [row]; omega
  have hv : record.val = 2 * row.val + 2 := by dsimp only [row]; omega
  have hr := hrange (writes.get row) (List.getElem_mem row.isLt)
  have hkRead : (decodeTokens tokens)[previous.val]? = some (writes.get row).1 := by
    rw [hraw, hp]
    exact (recallRawTable_record_at writes region row).1
  have hvRead : (decodeTokens tokens)[record.val]? = some (writes.get row).2 := by
    rw [hraw, hv]
    exact (recallRawTable_record_at writes region row).2
  obtain ⟨key, hk, _⟩ := recallDecoded_key_at P rewrites tokens previous _ hkRead hr.1
  obtain ⟨value, hvalue, _⟩ := recallDecoded_value_at P rewrites tokens record _ hvRead hr.2
  exact ⟨key, value, hk, hvalue⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 ∧
    (1 : Fin 6).val + 1 = (2 : Fin 6).val ∧ recallTableValuePosition 2 (2 : Fin 6).val := by decide

/-- Any actual table value discovered in the parsed raw list has a positive even BOS-offset position.
Source: genuine raw value-row recovery, not an assumed table mask or supplied prepared role encoding. -/
theorem recallBinding_value_position (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer)
    (record : Fin tokens.length) (value : Fin 256) (hvalue : tokens.get record = recallValueId value)
    (hinside : record.val ≤ 2 * P) : recallTableValuePosition P record.val := by
  obtain ⟨writes, region, _, _, hraw, hlen, hrange, _, _, _, _⟩ :=
    recallAnswer_parsed_table ⟨256, P, rewrites⟩ (decodeTokens tokens) answer hanswer
  have hv : ValueToken ⟨256, P, rewrites⟩ (((292 + value.val : ℕ) : ℤ)) :=
    (recallValueToken_iff P rewrites (recallValueId value)).mpr ⟨value, rfl⟩
  have hread := recallDecoded_value_id tokens record value hvalue
  rw [hraw] at hread
  obtain ⟨row, hp, _⟩ := recallRawTable_value_row ⟨256, P, rewrites⟩ writes region
    hrange record.val (by simpa only [hlen] using hinside) _ hread hv
  refine ⟨?_, hinside, ?_⟩ <;> omega

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 ∧
    recallOverwriteTokens.get (4 : Fin 6) = recallValueId 1 ∧ (4 : Fin 6).val ≤ 2 * 2 := by decide

/-- Complete raw parsing derives the true query, valid table slot and actual chronological last key/value write without model premises.
Source: the independent raw selected-write theorem plus physical value-row recovery; every earlier overwrite remains visible. -/
theorem recallBinding_selected (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query previous selected : Fin tokens.length, ∃ key value : Fin 256,
      query.val + 1 = tokens.length ∧ tokens.get query = recallKeyId key ∧
      previous.val + 1 = selected.val ∧ tokens.get previous = recallKeyId key ∧
      tokens.get selected = recallValueId value ∧ recallTableValuePosition P selected.val ∧
      selected.val ≤ query.val ∧ RecallRawLatestWrite P tokens.get selected key ∧ ((292 + value.val : ℕ) : ℤ) = answer := by
  obtain ⟨query, selected, previous, key, value, hquery, hq, hp, hk, hv, hin, hvisible, hlast, ha⟩ :=
    recallParser_encoded_selected P rewrites tokens answer hanswer
  exact ⟨query, previous, selected, key, value, hquery, hq, hp, hk, hv,
    recallBinding_value_position P rewrites tokens answer hanswer selected value hv hin, hvisible, hlast, ha⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- Every parsed adjacent valid-slot route with the final query key is no later than the genuine last write.
Source: table-slot key/value recovery and the independent chronological overwrite predicate, not an attention-score premise. -/
theorem recallBinding_matching_last (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer)
    (selected previous record : Fin tokens.length) (key : Fin 256)
    (hlast : RecallRawLatestWrite P tokens.get selected key) (hprevious : previous.val + 1 = record.val)
    (hslot : recallTableValuePosition P record.val) (hkey : tokens.get previous = recallKeyId key) : record.val ≤ selected.val := by
  obtain ⟨_, value, _, hv⟩ := recallBinding_table_pair P rewrites tokens answer hanswer previous record hprevious hslot
  exact hlast record previous value hprevious hkey hv hslot.2.1

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 ∧
    RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) (0 : Fin 256) ∧
    (1 : Fin 6).val + 1 = (2 : Fin 6).val ∧ recallTableValuePosition 2 (2 : Fin 6).val ∧
    recallOverwriteTokens.get (1 : Fin 6) = recallKeyId 0 :=
  ⟨by decide, recallOverwriteTokens_raw_latest, by decide, by decide, by decide⟩

/-- A real adjacent record has exactly one neighboring key position in the same unchanged finite input.
Source: the actual physical successor indices used by raw serialization, without a supplied paired-token encoder. -/
theorem recallBinding_unique_previous {T : ℕ} (previous other record : Fin T)
    (hp : previous.val + 1 = record.val) (ho : other.val + 1 = record.val) : previous = other := by
  apply Fin.ext
  omega

example : (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧ (3 : Fin 6).val + 1 = (4 : Fin 6).val := by decide

/-- Both real table slots retain their original distinct values under a genuine earlier/later overwrite input.
Source: the literal raw MQAR IDs at their unchanged physical routes, with no table prefilter. -/
theorem recallBinding_overwrite_pairs :
    recallOverwriteTokens.get (1 : Fin 6) = recallKeyId 0 ∧ recallOverwriteTokens.get (2 : Fin 6) = recallValueId 0 ∧
    recallOverwriteTokens.get (3 : Fin 6) = recallKeyId 0 ∧ recallOverwriteTokens.get (4 : Fin 6) = recallValueId 1 := by
  constructor
  · decide
  · constructor
    · decide
    · decide

/-- The same full parsed input derives the actual late selected position and its correct value symbol.
Source: raw integer parser and chronological latest-write transport, retaining the conflicting earlier value as a rival. -/
theorem recallBinding_overwrite_latest :
    RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) (0 : Fin 256) ∧
      recallTableValuePosition 2 (4 : Fin 6).val ∧ (recallValueId (1 : Fin 256)).val = 293 := by
  exact ⟨recallOverwriteTokens_raw_latest, by decide, rfl⟩

/-- A post-table value filler occupies a position excluded by the derived table-slot data predicate.
Source: the full raw three-record BOS/table/query/filler/query serialization with two distinct queries,
not a learned-forward hard mask; the first key has a genuine earlier/later overwrite. -/
theorem recallBinding_filler_position :
    recallAnswer ⟨256, 3, true⟩
      [1, 36, 292, 36, 293, 37, 294, 37, 300, 36] = some 293 ∧
      ¬recallTableValuePosition 3 8 := by decide

end Transformer.GPTMini.Convex.Structured
