import Transformer.Basis.RecallTablePositions

/-!
# Exact adjacency and chronological positions in full raw recall prefixes

Source: MQAR.targets/sample at cbafbe9, including actual BOS, the
whole adjacent table and an arbitrary checked query/filler region.
Optional list indexing preserves the original integer positions
without an external semantic array or a bound assumed for the model.

A real table value is proved to come from its corresponding actual
record; its predecessor remains that record's raw key after BOS and
the query region are included. The chosen last write has explicit
key/value positions, and every other matching write is no later.
Successful parser inversion supplies these conditions. These facts
will be transferred to the checked finite embedding indices; they
do not inspect a transformer state or assume its predicted answer.
-/

namespace Transformer.Basis

/-- Every actual raw table value in a full prefix recovers its own chronological row and genuine value.
Source: original BOS-offset table indexing, validated interval exclusion and exact raw value serialization. -/
theorem recallRawTable_value_row (s : RecallSpec) (writes : List (ℤ × ℤ)) (region : Tokens)
    (hrange : ∀ record ∈ writes, KeyToken s record.1 ∧ ValueToken s record.2)
    (index : ℕ) (hinside : index ≤ 2 * writes.length) (value : ℤ)
    (hread : (bos :: (recallTableTokens writes ++ region))[index]? = some value) (hvalue : ValueToken s value) :
    ∃ row : Fin writes.length, index = 2 * row.val + 2 ∧ (writes.get row).2 = value := by
  have hzero : index ≠ 0 := by
    intro hz
    subst index
    rw [List.getElem?_cons_zero] at hread
    have he := Option.some.inj hread
    rw [← he] at hvalue
    dsimp [ValueToken, bos, ReservedToken.id, identityBase] at hvalue
    omega
  have hi : index - 1 < (recallTableTokens writes).length := by rw [recallTableTokens_length]; omega
  rw [List.getElem?_cons, ite_eq_right hzero, List.getElem?_append_left hi, List.getElem?_eq_getElem hi] at hread
  have he := Option.some.inj hread
  have hv : ValueToken s ((recallTableTokens writes)[index - 1]'hi) := by rw [he]; exact hvalue
  obtain ⟨row, hp, hr⟩ := recallTableTokens_value_row s writes hrange (index - 1) hi hv
  exact ⟨row, by omega, hr.symm.trans he⟩

example : (∀ record ∈ ([(36, 292), (36, 300)] : List (ℤ × ℤ)),
    KeyToken hardRecall record.1 ∧ ValueToken hardRecall record.2) ∧ (4 : ℕ) ≤ 2 * 2 ∧
    (bos :: (recallTableTokens [(36, 292), (36, 300)] ++ [36]))[4]? = some 300 ∧ ValueToken hardRecall 300 := by
  norm_num [KeyToken, ValueToken, hardRecall, identityBase, bos, recallTableTokens]

/-- Every real full-prefix table value has its actual neighboring raw key, with the original BOS offset retained.
Source: derived real row recovery and exact key lookup, not a paired-key input to the transformer. -/
theorem recallRawTable_value_predecessor (s : RecallSpec) (writes : List (ℤ × ℤ)) (region : Tokens)
    (hrange : ∀ record ∈ writes, KeyToken s record.1 ∧ ValueToken s record.2)
    (index : ℕ) (hinside : index ≤ 2 * writes.length) (value : ℤ)
    (hread : (bos :: (recallTableTokens writes ++ region))[index]? = some value) (hvalue : ValueToken s value) :
    ∃ previous key, previous + 1 = index ∧ KeyToken s key ∧
      (bos :: (recallTableTokens writes ++ region))[previous]? = some key := by
  obtain ⟨row, hi, _⟩ := recallRawTable_value_row s writes region hrange index hinside value hread hvalue
  refine ⟨2 * row.val + 1, (writes.get row).1, by omega, ?_, ?_⟩
  · exact (hrange (writes.get row) (List.getElem_mem row.isLt)).1
  · exact (recallRawTable_record_at writes region row).1

example : (∀ record ∈ ([(36, 292)] : List (ℤ × ℤ)),
    KeyToken easyRecall record.1 ∧ ValueToken easyRecall record.2) ∧ (2 : ℕ) ≤ 2 * 1 ∧
    (bos :: (recallTableTokens [(36, 292)] ++ [36]))[2]? = some 292 ∧ ValueToken easyRecall 292 := by
  norm_num [KeyToken, ValueToken, easyRecall, identityBase, bos, recallTableTokens]

/-- The raw key and value of the selected chronological last-write record occupy explicit adjacent positions in the full input.
Source: original table serialization at the before.length record, with arbitrary earlier/later records and query region. -/
theorem recallRawTable_selected_at (before after : List (ℤ × ℤ)) (region : Tokens) (query answer : ℤ) :
    (bos :: (recallTableTokens (before ++ (query, answer) :: after) ++ region))[2 * before.length + 1]? = some query ∧
      (bos :: (recallTableTokens (before ++ (query, answer) :: after) ++ region))[2 * before.length + 2]? = some answer := by
  let row : Fin (before ++ (query, answer) :: after).length := ⟨before.length, by simp⟩
  have he : (before ++ (query, answer) :: after).get row = (query, answer) := by
    rw [List.get_eq_getElem, List.getElem_append_right (le_refl _)]
    simp only [Nat.sub_self, List.getElem_cons_zero]
  have h := recallRawTable_record_at (before ++ (query, answer) :: after) region row
  rw [he] at h
  exact h

/-- Every actual table write whose neighboring key matches the final query occurs no later than its selected last write.
Source: derived raw value-row position, exact neighboring key and chronological suffix exclusion from MQAR.targets. -/
theorem recallRawTable_last_position (s : RecallSpec) (before after : List (ℤ × ℤ)) (region : Tokens) (query answer : ℤ)
    (hrange : ∀ record ∈ before ++ (query, answer) :: after, KeyToken s record.1 ∧ ValueToken s record.2)
    (hlast : ∀ record ∈ after, record.1 ≠ query) (record previous : ℕ) (value : ℤ)
    (hprev : previous + 1 = record) (hinside : record ≤ 2 * (before ++ (query, answer) :: after).length)
    (hkey : (bos :: (recallTableTokens (before ++ (query, answer) :: after) ++ region))[previous]? = some query)
    (hread : (bos :: (recallTableTokens (before ++ (query, answer) :: after) ++ region))[record]? = some value)
    (hvalue : ValueToken s value) : record ≤ 2 * before.length + 2 := by
  obtain ⟨row, hi, _⟩ := recallRawTable_value_row s (before ++ (query, answer) :: after) region
    hrange record hinside value hread hvalue
  have hp : previous = 2 * row.val + 1 := by omega
  have hk := (recallRawTable_record_at (before ++ (query, answer) :: after) region row).1
  rw [← hp, hkey] at hk
  have hr := recallWrites_last_key_index before after query answer hlast row (Option.some.inj hk).symm
  omega

example : (∀ record ∈ ([(36, 292)] ++ (36, 300) :: [(37, 293)] : List (ℤ × ℤ)),
    KeyToken hardRecall record.1 ∧ ValueToken hardRecall record.2) ∧
    (∀ record ∈ ([(37, 293)] : List (ℤ × ℤ)), record.1 ≠ 36) ∧ (1 : ℕ) + 1 = 2 ∧ (2 : ℕ) ≤ 2 * 3 ∧
    (bos :: (recallTableTokens ([(36, 292)] ++ (36, 300) :: [(37, 293)]) ++ [36]))[1]? = some 36 ∧
    (bos :: (recallTableTokens ([(36, 292)] ++ (36, 300) :: [(37, 293)]) ++ [36]))[2]? = some 292 ∧
    ValueToken hardRecall 292 := by
  norm_num [KeyToken, ValueToken, hardRecall, identityBase, bos, recallTableTokens]

/-- The actual last raw query occupies the final index of the complete BOS/table/query prefix.
Source: MQAR supervision at every query position; the supplied region ends at the current query without a prepared model row. -/
theorem recallRawTable_final_query (writes : List (ℤ × ℤ)) (queries : Tokens) (query : ℤ) :
    (bos :: (recallTableTokens writes ++ queries ++ [query]))[
      (bos :: (recallTableTokens writes ++ queries ++ [query])).length - 1]? = some query := by
  let contextPrefix := bos :: (recallTableTokens writes ++ queries)
  have he : bos :: (recallTableTokens writes ++ queries ++ [query]) = contextPrefix ++ [query] := rfl
  rw [he, List.length_append, List.length_singleton]
  simp only [show contextPrefix.length + 1 - 1 = contextPrefix.length from by omega,
    List.getElem?_append_right (le_refl contextPrefix.length), Nat.sub_self, List.getElem?_cons_zero]

/-- Every non-BOS raw token of a successfully parsed recall prefix is a genuine key or value/filler.
Source: complete table and query-region range checks derived by the actual parser. -/
theorem recallAnswer_body_alphabet (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) :
    ∃ body, tokens = bos :: body ∧ ∀ token ∈ body, KeyToken s token ∨ ValueToken s token := by
  obtain ⟨writes, region, _, _, hraw, _, hrange, halphabet, _, _, _⟩ :=
    recallAnswer_parsed_table s tokens answer hanswer
  refine ⟨recallTableTokens writes ++ region, hraw, ?_⟩
  intro token htoken
  rcases List.mem_append.mp htoken with ht | hr
  · exact recallTableTokens_alphabet s writes hrange token ht
  · exact halphabet token hr

example : recallAnswer hardRecall (rewritePrefix 292 300) = some 300 := by decide

/-- Every bounded raw position after BOS in a successfully parsed recall prefix has the genuine key/value alphabet.
Source: full checked table/query parsing and actual list-tail membership at the same integer index. -/
theorem recallAnswer_nonbos_alphabet (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) (index : ℕ) (hindex : index < tokens.length) (hpos : 0 < index) :
    KeyToken s (tokens[index]'hindex) ∨ ValueToken s (tokens[index]'hindex) := by
  obtain ⟨body, hraw, halphabet⟩ := recallAnswer_body_alphabet s tokens answer hanswer
  have hm : tokens[index]'hindex ∈ body := by
    have h := List.getElem_mem_tail tokens (by omega : index ≠ 0) hindex
    simpa only [hraw, List.tail_cons] using h
  exact halphabet _ hm

example : recallAnswer hardRecall (rewritePrefix 292 300) = some 300 ∧
    1 < (rewritePrefix 292 300).length ∧ (0 : ℕ) < 1 := by decide

/-- The final position of a successfully parsed raw recall prefix is a validated table-bound query key.
Source: the full real parser's table/query inversion and original last-query raw index. -/
theorem recallAnswer_final_query_index (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) :
    ∃ query, tokens[tokens.length - 1]? = some query ∧ KeyToken s query := by
  obtain ⟨writes, region, queries, query, hraw, _, _, _, hregion, hkey, _⟩ :=
    recallAnswer_parsed_table s tokens answer hanswer
  refine ⟨query, ?_, hkey⟩
  rw [hraw, hregion]
  simpa only [List.append_assoc] using recallRawTable_final_query writes queries query

example : recallAnswer easyRecall (bindingPrefix 292 293) = some 292 := by decide

end Transformer.Basis
