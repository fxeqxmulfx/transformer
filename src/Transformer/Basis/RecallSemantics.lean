import Transformer.Basis.Recall

/-!
# The last raw binding behind a supervised recall answer

Source: MQAR.targets in synthetic/retrieval.py at cbafbe9, following
arXiv:2312.04927v1, §3 and Appendix E.1. These are inversion properties of
the actual integer-token parser, not assumptions about an already paired
input. A successful answer has a final query, a complete adjacent table,
and a last matching write with no later occurrence of that key.

The chronological decomposition is the semantic target for a retrieval
head. It permits repeated writes and unrelated records after the selected
write; selecting the first match would not satisfy the theorem.
-/

namespace Transformer.Basis

/-- A successful parse recovers exactly the raw adjacent table and its suffix.
Source: MQAR.writes/targets at cbafbe9, with the parser's range checks. -/
theorem recallWrites_exact (s : RecallSpec) (n : ℕ) (tokens : Tokens)
    (writes : List (ℤ × ℤ)) (rest : Tokens)
    (hparse : recallWrites s n tokens = some (writes, rest)) :
    writes.length = n ∧ tokens = recallTableTokens writes ++ rest ∧
      ∀ row ∈ writes, KeyToken s row.1 ∧ ValueToken s row.2 := by
  induction n generalizing tokens writes with
  | zero =>
      simp only [recallWrites, Option.some.injEq, Prod.mk.injEq] at hparse
      rcases hparse with ⟨rfl, rfl⟩
      exact ⟨rfl, rfl, by simp⟩
  | succ n ih =>
      cases tokens with
      | nil => simp [recallWrites] at hparse
      | cons key tokens =>
          cases tokens with
          | nil => simp [recallWrites] at hparse
          | cons value tokens =>
              simp only [recallWrites] at hparse
              split at hparse
              · rename_i hrange
                cases hp : recallWrites s n tokens with
                | none => simp [hp] at hparse
                | some parsed =>
                    rcases parsed with ⟨rows, suffix⟩
                    simp only [hp, Option.map_some, Option.some.injEq, Prod.mk.injEq]
                      at hparse
                    rcases hparse with ⟨rfl, rfl⟩
                    rcases ih tokens rows hp with ⟨hlen, hraw, hranges⟩
                    refine ⟨by simp [hlen], ?_, ?_⟩
                    · simp only [recallTableTokens, List.cons_append, hraw]
                    · intro row hrow
                      rcases List.mem_cons.mp hrow with heq | hmem
                      · subst row
                        exact hrange
                      · exact hranges row hmem
              · simp at hparse

example : recallWrites hardRecall 2 [36, 292, 36, 300, 36] =
    some ([(36, 292), (36, 300)], [36]) := by decide

/-- A lookup is absent exactly when the table contains no matching key.
Source: retrieval.py's table membership check, for chronological records. -/
theorem latestValue_none_iff (writes : List (ℤ × ℤ)) (query : ℤ) :
    latestValue writes query = none ↔ ∀ row ∈ writes, row.1 ≠ query := by
  induction writes with
  | nil => simp [latestValue]
  | cons row rows ih =>
      rcases row with ⟨key, value⟩
      cases hr : latestValue rows query with
      | none =>
          have ht := ih.mp hr
          rw [List.forall_mem_cons]
          simp only [latestValue, hr]
          change (if key = query then some value else none) = none ↔
            key ≠ query ∧ ∀ row ∈ rows, row.1 ≠ query
          split_ifs with hk
          · constructor
            · intro h
              cases h
            · intro h
              exact False.elim (h.1 hk)
          · exact ⟨fun _ => ⟨hk, ht⟩, fun _ => rfl⟩
      | some answer =>
          have hn : ¬∀ row ∈ rows, row.1 ≠ query := by
            intro h
            have he := ih.mpr h
            rw [hr] at he
            cases he
          rw [List.forall_mem_cons]
          simp only [latestValue, hr]
          constructor
          · intro h
            cases h
          · intro h
            exact False.elim (hn h.2)

/-- A last-value answer is exactly a matching record followed by no later matching key.
Source: retrieval.py's chronological table updates, with arbitrary intervening keys. -/
theorem latestValue_some_iff (writes : List (ℤ × ℤ)) (query answer : ℤ) :
    latestValue writes query = some answer ↔
      ∃ before after, writes = before ++ (query, answer) :: after ∧
        ∀ row ∈ after, row.1 ≠ query := by
  constructor
  · intro hlookup
    induction writes with
    | nil => simp [latestValue] at hlookup
    | cons row rows ih =>
        rcases row with ⟨key, value⟩
        cases hr : latestValue rows query with
        | some result =>
            have ha : result = answer := by simpa [latestValue, hr] using hlookup
            subst result
            obtain ⟨before, after, hrows, hlast⟩ := ih hr
            exact ⟨(key, value) :: before, after, by simp [hrows], hlast⟩
        | none =>
            have hkv : key = query ∧ value = answer := by
              simpa [latestValue, hr] using hlookup
            rcases hkv with ⟨rfl, rfl⟩
            exact ⟨[], rows, rfl, (latestValue_none_iff rows key).mp hr⟩
  · rintro ⟨before, after, rfl, hlast⟩
    have hn := (latestValue_none_iff after query).mpr hlast
    rw [latestValue_append]
    simp [latestValue, hn]

/-- Successful query scanning ends in a real bound key, not a value filler.
Source: MQAR.targets supervises query positions only, after all table writes. -/
theorem recallQueries_final (s : RecallSpec) (writes : List (ℤ × ℤ))
    (seen tokens : Tokens) (answer : ℤ)
    (hquery : recallQueries s writes seen tokens = some answer) :
    ∃ before query, tokens = before ++ [query] ∧ KeyToken s query ∧
      latestValue writes query = some answer := by
  induction tokens generalizing seen with
  | nil => simp [recallQueries] at hquery
  | cons query rest ih =>
      simp only [recallQueries] at hquery
      split at hquery
      · rename_i hkey
        split at hquery
        · simp at hquery
        · cases hl : latestValue writes query with
          | none => simp [hl] at hquery
          | some value =>
              simp only [hl] at hquery
              split at hquery
              · rename_i hrest
                subst rest
                have ha : value = answer := Option.some.inj hquery
                exact ⟨[], query, rfl, hkey, ha ▸ hl⟩
              · obtain ⟨before, last, hraw, hk, ha⟩ := ih (query :: seen) hquery
                exact ⟨query :: before, last, by simp [hraw], hk, ha⟩
      · split at hquery
        · obtain ⟨before, last, hraw, hk, ha⟩ := ih seen hquery
          exact ⟨query :: before, last, by simp [hraw], hk, ha⟩
        · simp at hquery

example : recallQueries hardRecall [(36, 292), (36, 300), (37, 293)]
    [] [37, 294, 36] = some 300 := by decide

end Transformer.Basis
