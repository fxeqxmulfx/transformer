module

public import Transformer.Basis.RecallParsed

/-!
# Moving one write across the same MQAR table

Source: MQAR.targets in synthetic/retrieval.py at cbafbe9, following
arXiv:2312.04927v1 §3. The experimental intervention inserts the same
key and value at the beginning or end of the writes. The query region
keeps all its tokens and positions. An early write preserves every
previously defined answer. A late write changes only its own key.
-/

@[expose] public section

namespace Transformer.Basis

/-- Every successful query-prefix answer follows a change of stored values.
Earlier queries must still be found and cannot be repeated. Source:
MQAR.targets, the post-table scan through the final supervised key. -/
theorem recallQueries_revalue (s : RecallSpec) (before after : List (ℤ × ℤ))
    (f : ℤ → ℤ → ℤ)
    (hlookup : ∀ key old, latestValue before key = some old →
      latestValue after key = some (f key old))
    (seen prior : Tokens) (query answer : ℤ)
    (hanswer : recallQueries s before seen (prior ++ [query]) = some answer) :
    recallQueries s after seen (prior ++ [query]) = some (f query answer) := by
  induction prior generalizing seen with
  | nil =>
    by_cases hk : KeyToken s query
    · by_cases hs : query ∈ seen
      · simp [recallQueries, hk, hs] at hanswer
      · cases h : latestValue before query with
        | none => simp [recallQueries, hk, hs, h] at hanswer
        | some old =>
          have he : old = answer := by simpa [recallQueries, hk, hs, h] using hanswer
          simp [recallQueries, hk, hs, hlookup query old h, he]
    · simp [recallQueries, hk] at hanswer
  | cons x xs ih =>
    have hn : xs ++ [query] ≠ [] := List.append_ne_nil_of_right_ne_nil xs (by simp)
    by_cases hk : KeyToken s x
    · by_cases hs : x ∈ seen
      · simp [recallQueries, hk, hs] at hanswer
      · cases h : latestValue before x with
        | none => simp [recallQueries, hk, hs, h] at hanswer
        | some old =>
          simp only [List.cons_append, recallQueries, ite_eq_left hk, ite_eq_right hs, h, ite_eq_right hn] at hanswer
          simpa only [List.cons_append, recallQueries, ite_eq_left hk, ite_eq_right hs,
            hlookup x old h, ite_eq_right hn] using ih (x :: seen) hanswer
    · by_cases hv : ValueToken s x
      · simp only [List.cons_append, recallQueries, ite_eq_right hk, ite_eq_left hv] at hanswer
        simpa only [List.cons_append, recallQueries, ite_eq_right hk, ite_eq_left hv] using ih seen hanswer
      · simp [recallQueries, hk, hv] at hanswer

example : recallQueries hardRecall [(36, 300), (37, 293)] [] [37, 295, 36] = some 300 := by
  apply recallQueries_revalue hardRecall [(36, 292), (37, 294)] _
    (fun k a => if k = 36 then 300 else if k = 37 then 293 else a)
    _ [] [37, 295] 36 292 (by decide)
  intro k a ha
  by_cases h₁ : k = 36 <;> by_cases h₂ : k = 37 <;> simp_all [latestValue, eq_comm]

/-- Moving the added write across the table determines precisely which
supervised answers change, through any prefix of queries and fillers.
Source: MQAR.targets, table overwrites and repeated query validation. -/
theorem recallQueries_move_write (s : RecallSpec) (writes : List (ℤ × ℤ))
    (seen prior : Tokens) (query answer key value : ℤ)
    (hanswer : recallQueries s writes seen (prior ++ [query]) = some answer) :
    recallQueries s ((key, value) :: writes) seen (prior ++ [query]) = some answer ∧
      recallQueries s (writes ++ [(key, value)]) seen (prior ++ [query]) =
        some (if key = query then value else answer) := by
  constructor
  · apply recallQueries_revalue s writes _ (fun _ a => a) _ seen prior query answer hanswer
    intro k a ha
    simp [latestValue, ha]
  · apply recallQueries_revalue s writes _ (fun k a => if key = k then value else a) _
      seen prior query answer hanswer
    intro k a ha
    rw [latestValue_append]
    by_cases hk : key = k <;> simp [latestValue, hk, ha]

example : recallQueries hardRecall [(36, 300), (36, 292), (37, 294)] [] [37, 295, 36] = some 292 ∧
    recallQueries hardRecall [(36, 292), (37, 294), (36, 300)] [] [37, 295, 36] = some 300 := by
  exact recallQueries_move_write hardRecall [(36, 292), (37, 294)] [] [37, 295]
    36 292 36 300 (by decide)

/-- The complete raw prompts for the two orders have the same token
multiset, including BOS, all filler and every subsequent query.
Source: MQAR.sample's adjacent key/value serialization. -/
theorem recallTableTokens_move_write_perm (writes : List (ℤ × ℤ)) (region : Tokens)
    (key value : ℤ) :
    (bos :: (recallTableTokens ((key, value) :: writes) ++ region)).Perm
      (bos :: (recallTableTokens (writes ++ [(key, value)]) ++ region)) := by
  apply List.Perm.cons
  simpa only [recallTableTokens_append, recallTableTokens, List.cons_append,
    List.nil_append, List.append_assoc] using
    List.perm_append_comm_assoc [key, value] (recallTableTokens writes) region

end Transformer.Basis
