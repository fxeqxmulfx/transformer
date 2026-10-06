import Transformer.Basis.Encoding.Basic

/-!
# Every accepted raw MQAR prefix is encodable

Source: MQAR.targets and MQAR.vocab in synthetic/retrieval.py at cbafbe9.
The proof follows actual raw parsing: adjacent table keys and values are
validated, then every query and filler is validated, then BOS is checked.
It covers arbitrary valid prefix lengths and rewrite parameters, without
assuming that the model predicts the selected value.
-/

namespace Transformer.Basis.Encoding

/-- The actual MQAR vocabulary contains reserved IDs and equal key/value intervals.
Source: MQAR.vocab at cbafbe9; the first data token is ID 36. -/
def recallVocabulary (s : RecallSpec) : ℕ := 36 + 2 * s.symbols

/-- Every validated query/table key is a legal model embedding index.
Source: MQAR.targets's key-range check; both integer bounds follow from that interval. -/
theorem key_legal (s : RecallSpec) (t : ℤ) (h : KeyToken s t) :
    0 ≤ t ∧ t < (recallVocabulary s : ℤ) := by
  dsimp [KeyToken, identityBase, recallVocabulary] at *
  omega

example : KeyToken easyRecall 36 := by decide

/-- Every validated value or filler is a legal model embedding index.
Source: MQAR.targets's disjoint value-range check at cbafbe9. -/
theorem value_legal (s : RecallSpec) (t : ℤ) (h : ValueToken s t) :
    0 ≤ t ∧ t < (recallVocabulary s : ℤ) := by
  dsimp [ValueToken, identityBase, recallVocabulary] at *
  omega

example : ValueToken easyRecall 292 := by decide

/-- A successfully parsed raw table followed by a legal query region contains only legal IDs.
Source: recallWrites implements MQAR.targets's exact adjacent-record parser. -/
theorem writes_legal (s : RecallSpec) (n : ℕ) (tokens : Tokens)
    (writes : List (ℤ × ℤ)) (rest : Tokens)
    (hparse : recallWrites s n tokens = some (writes, rest))
    (hr : Legal (recallVocabulary s) rest) : Legal (recallVocabulary s) tokens := by
  induction n generalizing tokens writes rest with
  | zero =>
      simp only [recallWrites, Option.some.injEq, Prod.mk.injEq] at hparse
      rw [hparse.2]
      exact hr
  | succ n ih =>
      cases tokens with
      | nil => simp only [recallWrites] at hparse; contradiction
      | cons key body =>
          cases body with
          | nil => simp only [recallWrites] at hparse; contradiction
          | cons value body =>
              simp only [recallWrites] at hparse
              split at hparse
              · rename_i hkv
                cases he : recallWrites s n body with
                | none => simp only [he, Option.map_none] at hparse; contradiction
                | some result =>
                    rcases result with ⟨rows, remaining⟩
                    simp only [he, Option.map_some, Option.some.injEq, Prod.mk.injEq] at hparse
                    have ht := ih body rows remaining he (hparse.2 ▸ hr)
                    exact (legal_cons _ key _).mpr ⟨key_legal s key hkv.1,
                      (legal_cons _ value _).mpr ⟨value_legal s value hkv.2, ht⟩⟩
              · contradiction

example : recallWrites easyRecall 1 [36, 292, 36] = some ([(36, 292)], [36]) ∧
    Legal (recallVocabulary easyRecall) [36] :=
  ⟨by decide, by norm_num [Legal, recallVocabulary, easyRecall]⟩

/-- Successful query-region parsing checks every raw key or filler, including earlier positions.
Source: recallQueries implements MQAR.targets's post-table validation loop. -/
theorem queries_legal (s : RecallSpec) (writes : List (ℤ × ℤ)) (seen tokens : Tokens)
    (answer : ℤ) (hparse : recallQueries s writes seen tokens = some answer) :
    Legal (recallVocabulary s) tokens := by
  induction tokens generalizing seen with
  | nil => simp only [recallQueries] at hparse; contradiction
  | cons query rest ih =>
      simp only [recallQueries] at hparse
      split at hparse
      · rename_i hq
        apply (legal_cons _ query rest).mpr
        refine ⟨key_legal s query hq, ?_⟩
        split at hparse
        · contradiction
        · cases he : latestValue writes query with
          | none => simp only [he] at hparse; contradiction
          | some value =>
              simp only [he] at hparse
              split at hparse
              · rename_i hrest
                subst rest
                intro t ht
                contradiction
              · exact ih (query :: seen) hparse
      · split at hparse
        · rename_i hv
          exact (legal_cons _ query rest).mpr ⟨value_legal s query hv, ih seen hparse⟩
        · contradiction

example : recallQueries easyRecall [(36, 292), (37, 293)] [] [37, 294, 36] = some 292 := by
  decide

/-- The raw MQAR answer parser can accept only a legal complete integer prefix.
Source: BOS, adjacent-table and query/filler checks in MQAR.targets, all included explicitly. -/
theorem answer_legal (s : RecallSpec) (tokens : Tokens) (answer : ℤ)
    (hparse : recallAnswer s tokens = some answer) : Legal (recallVocabulary s) tokens := by
  cases tokens with
  | nil => simp only [recallAnswer] at hparse; contradiction
  | cons head body =>
      simp only [recallAnswer] at hparse
      split at hparse
      · rename_i hbos
        cases he : recallWrites s s.pairs body with
        | none => simp only [he] at hparse; contradiction
        | some result =>
            rcases result with ⟨writes, rest⟩
            simp only [he] at hparse
            split at hparse
            · contradiction
            · apply (legal_cons _ head body).mpr
              refine ⟨?_, writes_legal s s.pairs body writes rest he
                (queries_legal s writes [] rest answer hparse)⟩
              rw [hbos]
              change 0 ≤ (1 : ℤ) ∧ 1 < ((36 + 2 * s.symbols : ℕ) : ℤ)
              omega
      · contradiction

example : recallAnswer easyRecall
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] =
      some 292 := by decide

/-- Every supervised MQAR prefix fits its actual finite vocabulary, not an assumed paired-key input.
Source: the validated raw recall grammar and the successful-answer parser above. -/
theorem recallPrefix_legal (s : RecallSpec) {bound : ℕ} {tokens : Tokens}
    (h : RecallPrefix s bound tokens) : Legal (recallVocabulary s) tokens := by
  obtain ⟨_, answer, ha⟩ := h
  exact answer_legal s tokens answer ha

example : RecallPrefix easyRecall 64
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] :=
  easyRecall_prefix

/-- A supervised recall prefix is nonempty, so it supplies a real final model row.
Source: recallAnswer requires BOS before parsing any adjacent records. -/
theorem recallPrefix_nonempty (s : RecallSpec) {bound : ℕ} {tokens : Tokens}
    (h : RecallPrefix s bound tokens) : tokens ≠ [] := by
  obtain ⟨_, answer, ha⟩ := h
  intro hempty
  subst tokens
  simp only [recallAnswer] at ha
  contradiction

example : RecallPrefix easyRecall 64
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] :=
  easyRecall_prefix

end Transformer.Basis.Encoding
