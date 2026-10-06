import Transformer.Basis.RecallSemantics
import Transformer.Basis.Requirements

/-!
# A supervised raw recall answer names the last adjacent binding

Source: synthetic/retrieval.py at cbafbe9 and arXiv:2312.04927v1, §3,
Appendix E.1. Parser success yields an explicit raw-token decomposition:
BOS, earlier complete records, the selected key and its neighboring value,
later complete records, then a query region ending in that key.

The later records have no matching key. They may be nonempty, and earlier
records may contain that same key with a different value. The theorem
therefore covers hard-mode rewrites without replacing the raw input by
already paired records or assuming a desired final model answer.
-/

namespace Transformer.Basis

/-- Concatenation keeps the raw adjacency of chronological records.
Source: MQAR.sample writes each pair into two successive token positions. -/
theorem recallTableTokens_append (left right : List (ℤ × ℤ)) :
    recallTableTokens (left ++ right) = recallTableTokens left ++ recallTableTokens right := by
  induction left with
  | nil => rfl
  | cons row left ih =>
      rcases row with ⟨key, value⟩
      simp only [List.cons_append, recallTableTokens, ih]

/-- Every complete raw record occupies two token positions.
Source: MQAR.writes slices [key, value] with stride two at cbafbe9. -/
theorem recallTableTokens_length (writes : List (ℤ × ℤ)) :
    (recallTableTokens writes).length = 2 * writes.length := by
  induction writes with
  | nil => rfl
  | cons row rows ih =>
      rcases row with ⟨key, value⟩
      simp only [recallTableTokens, List.length_cons, ih]
      omega

/-- Every successful answer is bound to its final query by the actual last adjacent raw write.
Source: MQAR.targets at cbafbe9; chronology and raw adjacency are proved, not premises. -/
theorem recallAnswer_raw_last_write (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) :
    ∃ before after queries query,
      tokens = bos :: (recallTableTokens before ++ query :: answer ::
        (recallTableTokens after ++ queries ++ [query])) ∧
      before.length + 1 + after.length = s.pairs ∧
      KeyToken s query ∧ ValueToken s answer ∧ ∀ row ∈ after, row.1 ≠ query := by
  cases tokens with
  | nil => simp [recallAnswer] at hanswer
  | cons head body =>
      simp only [recallAnswer] at hanswer
      split at hanswer
      · rename_i hbos
        subst head
        cases hp : recallWrites s s.pairs body with
        | none => simp [hp] at hanswer
        | some parsed =>
            rcases parsed with ⟨writes, region⟩
            simp only [hp] at hanswer
            split at hanswer
            · simp at hanswer
            · obtain ⟨queries, query, hregion, hkey, hlookup⟩ :=
                recallQueries_final s writes [] region answer hanswer
              obtain ⟨before, after, hwrites, hlast⟩ :=
                (latestValue_some_iff writes query answer).mp hlookup
              rcases recallWrites_exact s s.pairs body writes region hp with
                ⟨hlen, hbody, hranges⟩
              have hmem : (query, answer) ∈ writes := by
                rw [hwrites]
                exact List.mem_append.mpr (Or.inr (List.mem_cons.mpr (Or.inl rfl)))
              refine ⟨before, after, queries, query, ?_, ?_, hkey,
                (hranges (query, answer) hmem).2, hlast⟩
              · rw [hbody, hwrites, hregion, recallTableTokens_append]
                simp only [recallTableTokens, List.cons_append, List.append_assoc]
              · rw [hwrites, List.length_append, List.length_cons] at hlen
                omega
      · simp at hanswer

example : recallAnswer hardRecall (rewritePrefix 292 300) = some 300 := by decide

/-- The selected answer really lies in the benchmark's value interval.
Source: MQAR.targets range checks, inherited through the raw last-binding decomposition. -/
theorem recallAnswer_value_token (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) : ValueToken s answer := by
  obtain ⟨before, after, queries, query, hraw, hn, hkey, hvalue, hlast⟩ :=
    recallAnswer_raw_last_write s tokens answer hanswer
  exact hvalue

example : recallAnswer easyRecall (bindingPrefix 292 293) = some 292 := by decide

/-- A supervised recall prefix includes BOS, the whole table and a final query.
Source: the actual raw parser, including the prefix before later queries have been sampled. -/
theorem recallAnswer_min_length (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) : 2 * s.pairs + 2 ≤ tokens.length := by
  obtain ⟨before, after, queries, query, hraw, hn, hkey, hvalue, hlast⟩ :=
    recallAnswer_raw_last_write s tokens answer hanswer
  rw [hraw]
  simp only [List.length_cons, List.length_append,
    recallTableTokens_length]
  omega

example : recallAnswer hardRecall (rewritePrefix 300 292) = some 292 := by decide

/-- The easy counterexample selects its first value and leaves seven unrelated records after it.
Source: the actual eight-write Basis input used to detect the bag encoder's binding loss. -/
theorem binding_prefix_selected_record :
    bindingPrefix 292 293 = bos :: (recallTableTokens ([] : List (ℤ × ℤ)) ++
      36 :: 292 :: (recallTableTokens
        [(37, 293), (38, 294), (39, 295), (40, 296), (41, 297), (42, 298), (43, 299)] ++
          ([] : Tokens) ++ [36])) ∧
    ∀ row ∈ ([(37, 293), (38, 294), (39, 295), (40, 296), (41, 297), (42, 298),
      (43, 299)] : List (ℤ × ℤ)), row.1 ≠ 36 := by decide

/-- The hard input selects the ninth write; later writes of other keys do not supersede it.
Source: the actual sixteen-write Basis overwrite example, retaining all adjacent raw tokens. -/
theorem rewrite_prefix_selected_record :
    rewritePrefix 292 300 = bos :: (recallTableTokens
      [(36, 292), (37, 293), (38, 294), (39, 295), (40, 296), (41, 297), (42, 298), (43, 299)] ++
      36 :: 300 :: (recallTableTokens
        [(37, 301), (38, 302), (39, 303), (40, 304), (41, 305), (42, 306), (43, 307)] ++
          ([] : Tokens) ++ [36])) ∧
    ∀ row ∈ ([(37, 301), (38, 302), (39, 303), (40, 304), (41, 305), (42, 306),
      (43, 307)] : List (ℤ × ℤ)), row.1 ≠ 36 := by decide

/-- The total integer answer agrees with the successful raw binding, including hard rewrites.
Source: recallNext's PAD fallback is used only when no validated answer exists. -/
theorem recallNext_of_answer (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) : recallNext s tokens = answer := by
  unfold recallNext
  rw [hanswer]
  rfl

example : recallAnswer hardRecall (rewritePrefix 292 300) = some 300 := by decide

/-- The full List Int continuation appends the value of the selected raw last binding.
Source: the checked MQAR oracle and the common answer-append convention. -/
theorem recallFunction_of_answer (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hanswer : recallAnswer s tokens = some answer) :
    recallFunction s tokens = tokens ++ [answer] ∧
      ∃ before after queries query,
        tokens = bos :: (recallTableTokens before ++ query :: answer ::
          (recallTableTokens after ++ queries ++ [query])) ∧
        ∀ row ∈ after, row.1 ≠ query := by
  constructor
  · unfold recallFunction extend
    rw [recallNext_of_answer s tokens answer hanswer]
  · obtain ⟨before, after, queries, query, hraw, hn, hk, hv, hlast⟩ :=
      recallAnswer_raw_last_write s tokens answer hanswer
    exact ⟨before, after, queries, query, hraw, hlast⟩

example : recallAnswer easyRecall (bindingPrefix 293 292) = some 293 := by decide

end Transformer.Basis
