import Transformer.GPTMini.Semantics.RecallParserLayout

/-!
# Actual finite last-write routing from the complete raw Basis parser

Source: MQAR.targets at cbafbe9, the full chronological integer
parser and the original unchanged finite GPTMini embedding array.
The selected row and its immediately preceding key are recovered
from successful integer reads; their finite bounds are derived.
Every actual matching table write is then proved no later than
this row, including earlier overwrites with different raw values.

The final query and answer symbols are the actual raw integer IDs
at their original positions. No selected position, layout predicate,
encoded state, attention score or desired model output is supplied
to the final parser theorem. Its conclusion provides the raw facts
needed by the already verified original finite-softmax model.
Both rewrite settings and arbitrary accepted record counts are
covered. Context limits and the public model callback are connected
in the subsequent full correctness theorem.
-/

namespace Transformer.GPTMini.Semantics

open Transformer.Basis

/-- Raw last-write chronology transfers to the same actual finite array, without any attention-score premise.
Source: full BOS/table/query integer positions and the original key/value ID serialization. -/
theorem recallRawTable_latest_finite (P : ℕ) (rewrites : Bool)
    (tokens : List (Fin recallConfig.vocab_size)) (before after : List (ℤ × ℤ)) (region : Tokens)
    (rawKey answer : ℤ) (selected : Fin tokens.length) (key : Fin 256)
    (hraw : decodeTokens tokens = bos :: (recallTableTokens (before ++ (rawKey, answer) :: after) ++ region))
    (hlen : (before ++ (rawKey, answer) :: after).length = P)
    (hrange : ∀ row ∈ before ++ (rawKey, answer) :: after,
      KeyToken ⟨256, P, rewrites⟩ row.1 ∧ ValueToken ⟨256, P, rewrites⟩ row.2)
    (hlast : ∀ row ∈ after, row.1 ≠ rawKey) (hselected : selected.val = 2 * before.length + 2)
    (hkey : ((36 + key.val : ℕ) : ℤ) = rawKey) : RecallRawLatestWrite P tokens.get selected key := by
  intro record previous value hprev hk hv hinside
  have hkr := recallDecoded_key_id tokens previous key hk
  have hvr := recallDecoded_value_id tokens record value hv
  rw [hkey, hraw] at hkr
  rw [hraw] at hvr
  have hvalue : ValueToken ⟨256, P, rewrites⟩ (((292 + value.val : ℕ) : ℤ)) :=
    (recallValueToken_iff P rewrites (recallValueId value)).mpr ⟨value, rfl⟩
  have h := recallRawTable_last_position ⟨256, P, rewrites⟩ before after region rawKey answer
    hrange hlast record.val previous.val _ hprev (by simpa only [hlen] using hinside) hkr hvr hvalue
  omega

example : decodeTokens recallOverwriteTokens = bos :: (recallTableTokens ([(36, 292)] ++ (36, 293) :: []) ++ [36]) ∧
    ([(36, 292)] ++ (36, 293) :: [] : List (ℤ × ℤ)).length = 2 ∧
    (∀ row ∈ ([(36, 292)] ++ (36, 293) :: [] : List (ℤ × ℤ)),
      KeyToken ⟨256, 2, true⟩ row.1 ∧ ValueToken ⟨256, 2, true⟩ row.2) ∧
    (∀ row ∈ ([] : List (ℤ × ℤ)), row.1 ≠ 36) ∧
    (4 : Fin 6).val = 2 * ([(36, 292)] : List (ℤ × ℤ)).length + 2 ∧
    (((36 + (0 : Fin 256).val : ℕ) : ℤ)) = 36 := by decide

/-- Full successful parsing derives the selected actual last write, its neighboring key, final query and answer ID.
Source: raw MQAR parser inversion and exact chronological finite/integer positions, with no routing certificate premise. -/
theorem recallParser_selected (P : ℕ) (rewrites : Bool) (head : Fin recallConfig.vocab_size)
    (tail : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    ∃ selected previous : Fin (head :: tail).length, ∃ key value : Fin 256,
      (head :: tail).get ⟨tail.length, by simp⟩ = recallKeyId key ∧
      previous.val + 1 = selected.val ∧ (head :: tail).get previous = recallKeyId key ∧
      (head :: tail).get selected = recallValueId value ∧ selected.val ≤ 2 * P ∧ selected.val ≤ tail.length ∧
      RecallRawLatestWrite P (head :: tail).get selected key ∧ ((292 + value.val : ℕ) : ℤ) = answer := by
  obtain ⟨writes, region, queries, rawKey, hraw, hlen, hrange, _, hregion, hkey, hlookup⟩ :=
    recallAnswer_parsed_table ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) answer hanswer
  obtain ⟨before, after, hwrites, hlast⟩ := (latestValue_some_iff writes rawKey answer).mp hlookup
  have hfull : decodeTokens (head :: tail) =
      bos :: (recallTableTokens (before ++ (rawKey, answer) :: after) ++ region) := by rw [hraw, hwrites]
  have hcount : (before ++ (rawKey, answer) :: after).length = P := by rw [← hwrites]; exact hlen
  have hranges : ∀ row ∈ before ++ (rawKey, answer) :: after,
      KeyToken ⟨256, P, rewrites⟩ row.1 ∧ ValueToken ⟨256, P, rewrites⟩ row.2 := by
    rw [← hwrites]
    exact hrange
  have hreads := recallRawTable_selected_at before after region rawKey answer
  have hprevious : (decodeTokens (head :: tail))[2 * before.length + 1]? = some rawKey := by
    rw [hfull]
    exact hreads.1
  have hselected : (decodeTokens (head :: tail))[2 * before.length + 2]? = some answer := by
    rw [hfull]
    exact hreads.2
  obtain ⟨previous, hp, _⟩ := recallDecoded_position (head :: tail) _ rawKey hprevious
  obtain ⟨selected, hs, _⟩ := recallDecoded_position (head :: tail) _ answer hselected
  obtain ⟨key, hk, hkeyCode⟩ := recallDecoded_key_at P rewrites (head :: tail) previous rawKey
    (by rw [hp]; exact hprevious) hkey
  obtain ⟨value, hv, hvalueCode⟩ := recallDecoded_value_at P rewrites (head :: tail) selected answer
    (by rw [hs]; exact hselected) (recallAnswer_value_token _ _ _ hanswer)
  have hquery : (decodeTokens (head :: tail))[tail.length]? = some rawKey := by
    have hr : (decodeTokens (head :: tail))[(decodeTokens (head :: tail)).length - 1]? = some rawKey := by
      rw [hraw, hregion]
      simpa only [List.append_assoc] using recallRawTable_final_query writes queries rawKey
    have hi : (decodeTokens (head :: tail)).length - 1 = tail.length := by
      simp only [decodeTokens, List.length_map, List.length_cons]
      omega
    rw [hi] at hr
    exact hr
  obtain ⟨queryKey, hq, hqueryCode⟩ := recallDecoded_key_at P rewrites (head :: tail)
    ⟨tail.length, by simp⟩ rawKey hquery hkey
  have he : queryKey = key := by apply Fin.ext; omega
  subst queryKey
  have hinside : selected.val ≤ 2 * P := by
    simp only [List.length_append, List.length_cons] at hcount
    omega
  have hvisible : selected.val ≤ tail.length := by
    have hi := selected.isLt
    simp only [List.length_cons] at hi
    omega
  exact ⟨selected, previous, key, value, hq, by omega, hk, hv, hinside, hvisible,
    recallRawTable_latest_finite P rewrites (head :: tail) before after region rawKey answer selected key
      hfull hcount hranges hlast hs hkeyCode, hvalueCode⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- Any successfully parsed finite list derives its actual final query and chronological selected write in the unchanged input.
Source: complete raw parser success rules out an empty list and supplies all real finite last-write routing conditions. -/
theorem recallParser_encoded_selected (P : ℕ) (rewrites : Bool)
    (tokens : List (Fin recallConfig.vocab_size)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query selected previous : Fin tokens.length, ∃ key value : Fin 256,
      query.val + 1 = tokens.length ∧ tokens.get query = recallKeyId key ∧
      previous.val + 1 = selected.val ∧ tokens.get previous = recallKeyId key ∧
      tokens.get selected = recallValueId value ∧ selected.val ≤ 2 * P ∧ selected.val ≤ query.val ∧
      RecallRawLatestWrite P tokens.get selected key ∧ ((292 + value.val : ℕ) : ℤ) = answer := by
  cases tokens with
  | nil => simp only [decodeTokens, List.map_nil, recallAnswer] at hanswer; contradiction
  | cons head tail =>
      obtain ⟨selected, previous, key, value, hq, hp, hk, hv, hin, hvis, hl, ha⟩ :=
        recallParser_selected P rewrites head tail answer hanswer
      exact ⟨⟨tail.length, by simp⟩, selected, previous, key, value, rfl, hq, hp, hk, hv, hin, hvis, hl, ha⟩

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := by decide

/-- The literal overwrite input's actual finite last-write predicate follows from its raw chronological serialization.
Source: the first write is (36,292), the later write is (36,293), and the final query is the same raw key 36.
The proof transfers that original integer chronology directly, rather than supplying the earlier model-array witness.
The final selected value occupies actual input position four, with key at position three and query at position five.
The initial value at position two remains a real competing write and is not discarded from the input.
All six tokens retain their ordinary finite IDs through the same genuine decode map.
This control exercises the hard-mode overwrite condition independently of any transformer logit or hidden state. -/
theorem recallOverwriteTokens_raw_latest :
    RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) (0 : Fin 256) := by
  apply recallRawTable_latest_finite 2 true recallOverwriteTokens [(36, 292)] [] [36] 36 293
  · decide
  · decide
  · decide
  · decide
  · decide
  · decide

end Transformer.GPTMini.Semantics
