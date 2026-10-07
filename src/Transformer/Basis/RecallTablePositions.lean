import Transformer.Basis.RecallParsed

/-!
# Exact raw recall table positions and the last matching index

Source: MQAR.sample/writes at cbafbe9, serializing every chronological
record into two adjacent raw tokens, and MQAR.targets selecting its
last matching key. These statements connect actual list indexing
to the validated raw records without a paired model representation.

Every table value occupies an odd zero-based position before BOS is
prepended, and its immediate predecessor is precisely that record's
key. The key/value intervals are disjoint, so no validated key can
be mistaken for one of those value positions. A matching record
in a table decomposed around its last write has index at most the
selected record index. These data-side index facts preserve arbitrary
earlier repeated writes and later unrelated rows; query regions and
model logits do not enter them.
-/

namespace Transformer.Basis

/-- Every even raw table position reads precisely the corresponding chronological key, also out of bounds.
Source: the actual two-token serialization, proved through its full recursive list computation. -/
theorem recallTableTokens_key_at (writes : List (ℤ × ℤ)) (row : ℕ) :
    (recallTableTokens writes)[2 * row]? = (writes[row]?).map Prod.fst := by
  induction writes generalizing row with
  | nil => simp only [recallTableTokens, List.getElem?_nil, Option.map_none]
  | cons record writes ih =>
      rcases record with ⟨key, value⟩
      cases row with
      | zero => rfl
      | succ row =>
          rw [show 2 * (row + 1) = (2 * row + 1) + 1 from by omega]
          simpa only [recallTableTokens, List.getElem?_cons_succ] using ih row

/-- Every odd raw table position reads the value paired with its actual immediately preceding key.
Source: the same original recursive serialization, retaining every chronological overwrite. -/
theorem recallTableTokens_value_at (writes : List (ℤ × ℤ)) (row : ℕ) :
    (recallTableTokens writes)[2 * row + 1]? = (writes[row]?).map Prod.snd := by
  induction writes generalizing row with
  | nil => simp only [recallTableTokens, List.getElem?_nil, Option.map_none]
  | cons record writes ih =>
      rcases record with ⟨key, value⟩
      cases row with
      | zero => rfl
      | succ row =>
          rw [show 2 * (row + 1) + 1 = ((2 * row + 1) + 1) + 1 from by omega]
          simpa only [recallTableTokens, List.getElem?_cons_succ] using ih row

/-- Bounded exact raw indexing gives the real key of each finite chronological record.
Source: even-position serialization and the proved complete table length. -/
theorem recallTableTokens_key_get (writes : List (ℤ × ℤ)) (row : Fin writes.length) :
    (recallTableTokens writes)[2 * row.val]'(by rw [recallTableTokens_length]; omega) = (writes.get row).1 := by
  have h := recallTableTokens_key_at writes row.val
  rw [List.getElem?_eq_getElem (by rw [recallTableTokens_length]; omega),
    List.getElem?_eq_getElem row.isLt, Option.map_some] at h
  exact Option.some.inj h

/-- Bounded exact indexing gives the corresponding value at the next raw position.
Source: odd-position serialization, not a separately supplied key/value association. -/
theorem recallTableTokens_value_get (writes : List (ℤ × ℤ)) (row : Fin writes.length) :
    (recallTableTokens writes)[2 * row.val + 1]'(by rw [recallTableTokens_length]; omega) = (writes.get row).2 := by
  have h := recallTableTokens_value_at writes row.val
  rw [List.getElem?_eq_getElem (by rw [recallTableTokens_length]; omega),
    List.getElem?_eq_getElem row.isLt, Option.map_some] at h
  exact Option.some.inj h

/-- A validated raw table value comes from exactly an odd position of an actual chronological record.
Source: real key/value interval exclusion and both explicit serialization-index formulas, without a parity premise. -/
theorem recallTableTokens_value_row (s : RecallSpec) (writes : List (ℤ × ℤ))
    (hrange : ∀ record ∈ writes, KeyToken s record.1 ∧ ValueToken s record.2)
    (index : ℕ) (hindex : index < (recallTableTokens writes).length)
    (hvalue : ValueToken s ((recallTableTokens writes)[index]'hindex)) :
    ∃ row : Fin writes.length, index = 2 * row.val + 1 ∧
      (recallTableTokens writes)[index]'hindex = (writes.get row).2 := by
  have hlen := recallTableTokens_length writes
  have hr : index / 2 < writes.length := by omega
  let row : Fin writes.length := ⟨index / 2, hr⟩
  have hk := hrange (writes.get row) (List.getElem_mem row.isLt)
  by_cases heven : index % 2 = 0
  · have hi : index = 2 * row.val := by dsimp [row]; omega
    have hkey : KeyToken s ((recallTableTokens writes)[index]'hindex) := by
      simpa only [hi, recallTableTokens_key_get] using hk.1
    exact False.elim (keyToken_not_value s _ hkey hvalue)
  · have hi : index = 2 * row.val + 1 := by dsimp [row]; omega
    refine ⟨row, hi, ?_⟩
    simpa only [hi] using recallTableTokens_value_get writes row

example : (∀ record ∈ ([(36, 292), (36, 300)] : List (ℤ × ℤ)),
    KeyToken hardRecall record.1 ∧ ValueToken hardRecall record.2) ∧
    1 < (recallTableTokens [(36, 292), (36, 300)]).length ∧
    ValueToken hardRecall ((recallTableTokens [(36, 292), (36, 300)])[1]) := by
  norm_num [KeyToken, ValueToken, hardRecall, identityBase, recallTableTokens]

/-- Every validated table value has an actual immediately preceding raw key within the same table.
Source: the derived odd record position and the real chronological key at twice the row index. -/
theorem recallTableTokens_value_predecessor (s : RecallSpec) (writes : List (ℤ × ℤ))
    (hrange : ∀ record ∈ writes, KeyToken s record.1 ∧ ValueToken s record.2)
    (index : ℕ) (hindex : index < (recallTableTokens writes).length)
    (hvalue : ValueToken s ((recallTableTokens writes)[index]'hindex)) :
    ∃ previous, ∃ hp : previous < (recallTableTokens writes).length,
      previous + 1 = index ∧ KeyToken s ((recallTableTokens writes)[previous]'hp) := by
  obtain ⟨row, he, _⟩ := recallTableTokens_value_row s writes hrange index hindex hvalue
  have hp : 2 * row.val < (recallTableTokens writes).length := by rw [recallTableTokens_length]; omega
  refine ⟨2 * row.val, hp, he.symm, ?_⟩
  rw [recallTableTokens_key_get]
  exact (hrange (writes.get row) (List.getElem_mem row.isLt)).1

example : (∀ record ∈ ([(36, 292)] : List (ℤ × ℤ)),
    KeyToken easyRecall record.1 ∧ ValueToken easyRecall record.2) ∧
    1 < (recallTableTokens [(36, 292)]).length ∧ ValueToken easyRecall ((recallTableTokens [(36, 292)])[1]) := by
  norm_num [KeyToken, ValueToken, easyRecall, identityBase, recallTableTokens]

/-- Prepending actual BOS and any suffix preserves both exact raw positions of every table record.
Source: MQAR.sample's one-position BOS offset and its complete fixed adjacent table before the query/filler region. -/
theorem recallRawTable_record_at (writes : List (ℤ × ℤ)) (region : Tokens) (row : Fin writes.length) :
    (bos :: (recallTableTokens writes ++ region))[2 * row.val + 1]? = some (writes.get row).1 ∧
      (bos :: (recallTableTokens writes ++ region))[2 * row.val + 2]? = some (writes.get row).2 := by
  have hk : 2 * row.val < (recallTableTokens writes).length := by rw [recallTableTokens_length]; omega
  have hv : 2 * row.val + 1 < (recallTableTokens writes).length := by rw [recallTableTokens_length]; omega
  constructor
  · rw [List.getElem?_cons_succ, List.getElem?_append_left hk, recallTableTokens_key_at,
      List.getElem?_eq_getElem row.isLt, Option.map_some]
    rfl
  · rw [show 2 * row.val + 2 = (2 * row.val + 1) + 1 from by omega,
      List.getElem?_cons_succ, List.getElem?_append_left hv, recallTableTokens_value_at,
      List.getElem?_eq_getElem row.isLt, Option.map_some]
    rfl

/-- Every matching chronological record is no later than the record selected by actual last-write semantics.
Source: MQAR.targets's table overwrite rule and the validated decomposition with no matching key in its later suffix. -/
theorem recallWrites_last_key_index (before after : List (ℤ × ℤ)) (query answer : ℤ)
    (hlast : ∀ record ∈ after, record.1 ≠ query) (row : Fin (before ++ (query, answer) :: after).length)
    (hkey : ((before ++ (query, answer) :: after).get row).1 = query) : row.val ≤ before.length := by
  by_contra hn
  have hr : row.val < before.length + 1 + after.length := by
    have h := row.isLt
    simp only [List.length_append, List.length_cons] at h
    omega
  have hnzero : row.val - before.length ≠ 0 := by omega
  have ht : row.val - before.length - 1 < after.length := by omega
  change ((before ++ (query, answer) :: after)[row.val]).1 = query at hkey
  rw [List.getElem_append_right (by omega)] at hkey
  simp only [List.getElem_cons, dite_eq_right hnzero] at hkey
  exact (hlast _ (List.getElem_mem ht)) hkey

example : (∀ record ∈ ([(37, 293)] : List (ℤ × ℤ)), record.1 ≠ 36) ∧
    (([(36, 292)] ++ (36, 300) :: [(37, 293)]).get (1 : Fin 3)).1 = 36 := by decide

end Transformer.Basis
