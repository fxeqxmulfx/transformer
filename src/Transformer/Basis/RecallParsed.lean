import Transformer.Basis.RecallAnswer

/-!
# Full raw recall parsing retains alphabet, adjacency and chronology

Source: MQAR.targets in synthetic/retrieval.py at cbafbe9. Successful
table parsing checks every adjacent key/value record; successful
query scanning checks every query and filler in its original order.
These data properties are proved without a supplied paired encoder.

The raw last-write decomposition is strengthened to retain the actual
range checks on both earlier and later tables and on the complete
query region. Later records may remain nonempty, and earlier writes
may match the final query with a different value. No uniqueness is
assumed for hard-mode table keys. These are parser facts needed to
discharge actual transformer routing predicates, independently of
any hidden state, attention score, readout or model answer.
-/

namespace Transformer.Basis

/-- A validated key can never simultaneously be a validated value or filler.
Source: the two adjacent disjoint integer intervals checked by MQAR.targets. -/
theorem keyToken_not_value (s : RecallSpec) (token : ℤ) (hkey : KeyToken s token) :
    ¬ValueToken s token := by
  intro hvalue
  dsimp [KeyToken, ValueToken] at hkey hvalue
  omega

example : KeyToken easyRecall 36 := by decide

/-- Serializing a validated chronological table produces only its genuine key/value alphabet.
Source: MQAR.sample's adjacent write layout and the range checks on each parsed record. -/
theorem recallTableTokens_alphabet (s : RecallSpec) (writes : List (ℤ × ℤ))
    (hrange : ∀ row ∈ writes, KeyToken s row.1 ∧ ValueToken s row.2) :
    ∀ token ∈ recallTableTokens writes, KeyToken s token ∨ ValueToken s token := by
  induction writes with
  | nil => intro token htoken; contradiction
  | cons row rows ih =>
      have hh := hrange row (List.mem_cons.mpr (Or.inl rfl))
      have ht := ih (fun r hr => hrange r (List.mem_cons.mpr (Or.inr hr)))
      rcases row with ⟨key, value⟩
      simp only [recallTableTokens, List.forall_mem_cons]
      exact ⟨Or.inl hh.1, Or.inr hh.2, ht⟩

example : ∀ row ∈ ([(36, 292), (36, 300)] : List (ℤ × ℤ)),
    KeyToken hardRecall row.1 ∧ ValueToken hardRecall row.2 := by
  norm_num [KeyToken, ValueToken, hardRecall, identityBase]

/-- Successful scanning validates every raw query or filler, not just the final supervised key.
Source: the actual MQAR post-table loop, including arbitrary earlier distinct queries and value fillers. -/
theorem recallQueries_alphabet (s : RecallSpec) (writes : List (ℤ × ℤ)) (seen tokens : Tokens)
    (answer : ℤ) (hparse : recallQueries s writes seen tokens = some answer) :
    ∀ token ∈ tokens, KeyToken s token ∨ ValueToken s token := by
  induction tokens generalizing seen with
  | nil => simp only [recallQueries] at hparse; contradiction
  | cons query rest ih =>
      simp only [recallQueries] at hparse
      split at hparse
      · rename_i hkey
        rw [List.forall_mem_cons]
        refine ⟨Or.inl hkey, ?_⟩
        split at hparse
        · contradiction
        · cases hl : latestValue writes query with
          | none => simp only [hl] at hparse; contradiction
          | some value =>
              simp only [hl] at hparse
              split at hparse
              · rename_i hrest
                subst rest
                intro token htoken
                contradiction
              · exact ih (query :: seen) hparse
      · split at hparse
        · rename_i hvalue
          exact List.forall_mem_cons.mpr ⟨Or.inr hvalue, ih seen hparse⟩
        · contradiction

example : recallQueries hardRecall [(36, 292), (36, 300), (37, 293)] [] [37, 294, 36] = some 300 := by decide

/-- A successful raw answer retains the whole validated chronological table and query-region decomposition.
Source: actual BOS/table/query parser inversion; no model-level table or final-query premise is supplied. -/
theorem recallAnswer_parsed_table (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) :
    ∃ writes region queries query,
      tokens = bos :: (recallTableTokens writes ++ region) ∧ writes.length = s.pairs ∧
      (∀ row ∈ writes, KeyToken s row.1 ∧ ValueToken s row.2) ∧
      (∀ token ∈ region, KeyToken s token ∨ ValueToken s token) ∧
      region = queries ++ [query] ∧ KeyToken s query ∧ latestValue writes query = some answer := by
  cases tokens with
  | nil => simp only [recallAnswer] at hanswer; contradiction
  | cons head body =>
      simp only [recallAnswer] at hanswer
      split at hanswer
      · rename_i hbos
        subst head
        cases hp : recallWrites s s.pairs body with
        | none => simp only [hp] at hanswer; contradiction
        | some parsed =>
            rcases parsed with ⟨writes, region⟩
            simp only [hp] at hanswer
            split at hanswer
            · contradiction
            · obtain ⟨queries, query, hregion, hkey, hlookup⟩ := recallQueries_final s writes [] region answer hanswer
              obtain ⟨hlen, hraw, hrange⟩ := recallWrites_exact s s.pairs body writes region hp
              exact ⟨writes, region, queries, query, congrArg (List.cons bos) hraw, hlen, hrange,
                recallQueries_alphabet s writes [] region answer hanswer, hregion, hkey, hlookup⟩
      · contradiction

example : recallAnswer hardRecall (rewritePrefix 292 300) = some 300 := by decide

/-- The actual last raw binding decomposition keeps every table/query range check needed by a token-local encoder.
Source: MQAR.targets's chronological lookup and checked raw parser; both hard overwrites and later unrelated writes are retained. -/
theorem recallAnswer_validated_last_write (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) :
    ∃ before after queries query,
      tokens = bos :: (recallTableTokens before ++ query :: answer ::
        (recallTableTokens after ++ queries ++ [query])) ∧
      before.length + 1 + after.length = s.pairs ∧
      (∀ row ∈ before, KeyToken s row.1 ∧ ValueToken s row.2) ∧
      (∀ row ∈ after, KeyToken s row.1 ∧ ValueToken s row.2) ∧
      (∀ token ∈ queries, KeyToken s token ∨ ValueToken s token) ∧
      KeyToken s query ∧ ValueToken s answer ∧ ∀ row ∈ after, row.1 ≠ query := by
  obtain ⟨writes, region, queries, query, hraw, hlen, hranges, halphabet, hregion, hkey, hlookup⟩ :=
    recallAnswer_parsed_table s tokens answer hanswer
  obtain ⟨before, after, hwrites, hlast⟩ := (latestValue_some_iff writes query answer).mp hlookup
  have hmem : (query, answer) ∈ writes := by
    rw [hwrites]
    exact List.mem_append.mpr (Or.inr (List.mem_cons.mpr (Or.inl rfl)))
  have hb : ∀ row ∈ before, KeyToken s row.1 ∧ ValueToken s row.2 := by
    intro row hrow
    apply hranges
    rw [hwrites]
    exact List.mem_append.mpr (Or.inl hrow)
  have ha : ∀ row ∈ after, KeyToken s row.1 ∧ ValueToken s row.2 := by
    intro row hrow
    apply hranges
    rw [hwrites]
    exact List.mem_append.mpr (Or.inr (List.mem_cons.mpr (Or.inr hrow)))
  refine ⟨before, after, queries, query, ?_, ?_, hb, ha, ?_, hkey, (hranges _ hmem).2, hlast⟩
  · rw [hraw, hwrites, hregion, recallTableTokens_append]
    simp only [recallTableTokens, List.cons_append, List.append_assoc]
  · rw [hwrites, List.length_append, List.length_cons] at hlen
    omega
  · intro token htoken
    apply halphabet
    rw [hregion]
    exact List.mem_append.mpr (Or.inl htoken)

example : recallAnswer easyRecall (bindingPrefix 292 293) = some 292 := by decide

end Transformer.Basis
