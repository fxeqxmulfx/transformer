import Transformer.Basis.Basic

/-!
# Raw integer MQAR prefixes

Source: arXiv:2312.04927v1, §3 and Appendix E.1, and the actual
MQAR.targets oracle in synthetic/retrieval.py at cbafbe9. The parser reads
adjacent key/value records, not an already paired embedding. Later writes
override earlier ones, as in Basis hard. A supervised prefix ends in a
previously bound key, with value-token fillers and distinct prior queries.

The grammar covers more than the sampler: exact distinct-key counts,
changed-value rewrites, minimum lengths and the final eight-query count
are not imposed. Every sampled supervised prefix belongs to this broader
grammar. Invalid prefixes return PAD instead of raising ValueError.
-/

namespace Transformer.Basis

/-- MQAR parameters needed by the prefix oracle; Source: Basis easy/hard recipes. -/
structure RecallSpec where
  symbols : ℕ
  pairs : ℕ
  rewrites : Bool

/-- The actual easy recall recipe; Source: domain/basis.py at cbafbe9. -/
def easyRecall : RecallSpec := ⟨256, 8, false⟩

/-- The actual hard recall recipe; Source: domain/basis.py at cbafbe9. -/
def hardRecall : RecallSpec := ⟨256, 16, true⟩

/-- The benchmark's key interval; Source: MQAR.targets and vocabulary.py. -/
def KeyToken (s : RecallSpec) (t : ℤ) : Prop :=
  identityBase ≤ t ∧ t < identityBase + s.symbols

/-- The disjoint value interval; Source: MQAR.targets and vocabulary.py. -/
def ValueToken (s : RecallSpec) (t : ℤ) : Prop :=
  identityBase + s.symbols ≤ t ∧ t < identityBase + 2 * s.symbols

instance (s : RecallSpec) (t : ℤ) : Decidable (KeyToken s t) :=
  inferInstanceAs (Decidable (_ ∧ _))
instance (s : RecallSpec) (t : ℤ) : Decidable (ValueToken s t) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- Look up the last matching write in chronological records.
Source: retrieval.py's table[key] = value update, including hard-mode rewrites. -/
def latestValue : List (ℤ × ℤ) → ℤ → Option ℤ
  | [], _ => none
  | (key, value) :: rest, query =>
      match latestValue rest query with
      | some answer => some answer
      | none => if key = query then some value else none

/-- Parse exactly n adjacent raw records, retaining the remaining query region.
Source: MQAR.writes/targets at cbafbe9; each record's token ranges are checked. -/
def recallWrites (s : RecallSpec) : ℕ → Tokens → Option (List (ℤ × ℤ) × Tokens)
  | 0, tokens => some ([], tokens)
  | n + 1, key :: value :: rest =>
      if KeyToken s key ∧ ValueToken s value then
        (recallWrites s n rest).map fun result => ((key, value) :: result.1, result.2)
      else none
  | _ + 1, _ => none

/-- Scan the query/filler region and return an answer only at its final query.
Source: MQAR.targets; query_gap is zero in both Basis modes. -/
def recallQueries (s : RecallSpec) (writes : List (ℤ × ℤ)) : Tokens → Tokens → Option ℤ
  | _, [] => none
  | seen, query :: rest =>
      if KeyToken s query then
        if query ∈ seen then none
        else match latestValue writes query with
          | none => none
          | some value =>
              if rest = [] then some value else recallQueries s writes (query :: seen) rest
      else if ValueToken s query then recallQueries s writes seen rest else none

/-- The raw MQAR oracle retains adjacency and uses the latest matching value.
Source: MQAR.targets; a non-supervised or invalid prefix has no answer. -/
def recallAnswer (s : RecallSpec) : Tokens → Option ℤ
  | [] => none
  | head :: body =>
      if head = bos then
        match recallWrites s s.pairs body with
        | none => none
        | some (writes, queries) =>
            if s.rewrites = false ∧ ¬(writes.map Prod.fst).Nodup then none
            else recallQueries s writes [] queries
      else none

/-- The total integer answer; Source: the new PAD fallback for the raw MQAR oracle. -/
def recallNext (s : RecallSpec) (tokens : Tokens) : ℤ := (recallAnswer s tokens).getD pad

/-- Recall in the requested append-one-token interface; Source: the new Basis convention. -/
def recallFunction (s : RecallSpec) : Tokens → Tokens := extend (recallNext s)

/-- A supervised prefix in the validated raw grammar, bounded by the context.
Source: MQAR.targets; sampler restrictions excluded above are not part of this predicate. -/
def RecallPrefix (s : RecallSpec) (bound : ℕ) (tokens : Tokens) : Prop :=
  tokens.length ≤ bound ∧ ∃ value, recallAnswer s tokens = some value

/-- Serialize chronological records as actual adjacent raw key/value tokens.
Source: MQAR.sample at cbafbe9, before the post-table query/filler region. -/
def recallTableTokens : List (ℤ × ℤ) → Tokens
  | [] => []
  | (key, value) :: rows => key :: value :: recallTableTokens rows

/-- Parsing raw serialized records preserves their adjacency, order and remaining context.
Source: MQAR.writes/targets; token-range hypotheses validate each complete raw record. -/
theorem recallWrites_table (s : RecallSpec) (writes : List (ℤ × ℤ)) (rest : Tokens)
    (hrange : ∀ row ∈ writes, KeyToken s row.1 ∧ ValueToken s row.2) :
    recallWrites s writes.length (recallTableTokens writes ++ rest) = some (writes, rest) := by
  induction writes with
  | nil => rfl
  | cons row rows ih =>
      have hh := hrange row (List.mem_cons.mpr (Or.inl rfl))
      have ht : ∀ r ∈ rows, KeyToken s r.1 ∧ ValueToken s r.2 :=
        fun r hr => hrange r (List.mem_cons.mpr (Or.inr hr))
      rcases row with ⟨key, value⟩
      simp only [List.length_cons, recallTableTokens, List.cons_append, recallWrites,
        ite_eq_left hh, ih ht]
      rfl

example : ∀ row ∈ ([(36, 292)] : List (ℤ × ℤ)),
    KeyToken easyRecall row.1 ∧ ValueToken easyRecall row.2 := by
  norm_num [KeyToken, ValueToken, easyRecall, identityBase]

/-- The latest match in a suffix takes precedence over every earlier record.
Source: MQAR hard's last-write semantics, proved for arbitrary chronological tables. -/
theorem latestValue_append (left right : List (ℤ × ℤ)) (query : ℤ) :
    latestValue (left ++ right) query =
      (latestValue right query).or (latestValue left query) := by
  induction left with
  | nil => cases hr : latestValue right query <;> simp [latestValue, hr]
  | cons row left ih =>
      rcases row with ⟨key, value⟩
      simp only [List.cons_append, latestValue, ih]
      cases latestValue right query <;> cases latestValue left query <;> simp

/-- Appending a matching write replaces the answer, even when its key already occurred.
Source: MQAR hard's overwrite semantics; no uniqueness premise is smuggled in. -/
theorem latestValue_last_write (writes : List (ℤ × ℤ)) (query value : ℤ) :
    latestValue (writes ++ [(query, value)]) query = some value := by
  rw [latestValue_append]
  simp [latestValue]

/-- A single valid query reads the chronological table's actual latest value.
Source: MQAR.targets; key validity is needed by the raw query-region parser. -/
theorem recallQueries_single (s : RecallSpec) (writes : List (ℤ × ℤ)) (query : ℤ)
    (hq : KeyToken s query) : recallQueries s writes [] [query] = latestValue writes query := by
  simp only [recallQueries, ite_eq_left hq, List.not_mem_nil, ↓reduceIte]
  cases latestValue writes query <;> rfl

example : KeyToken easyRecall 36 := by decide

/-- Every total recall continuation has exactly one appended token.
Source: the new List Int interface, also on invalid or unsupervised prefixes. -/
theorem recallFunction_length (s : RecallSpec) (tokens : Tokens) :
    (recallFunction s tokens).length = tokens.length + 1 := extend_length _ _

/-- Swapping adjacent values changes the correct answer despite the same token multiset.
Source: the raw-binding counterexample behind the Basis recall encoder repair. -/
theorem recall_binding_control :
    recallFunction ⟨256, 2, false⟩ [1, 36, 292, 37, 293, 36] =
      [1, 36, 292, 37, 293, 36, 292] ∧
    recallFunction ⟨256, 2, false⟩ [1, 36, 293, 37, 292, 36] =
      [1, 36, 293, 37, 292, 36, 293] := by decide

/-- Raw duplicate keys select their latest value when rewrites are enabled.
Source: the extension used by Basis hard; values really change in this example. -/
theorem recall_rewrite_control :
    recallFunction ⟨256, 2, true⟩ [1, 36, 292, 36, 293, 36] =
      [1, 36, 292, 36, 293, 36, 293] := by decide

/-- Easy-mode validation rejects those same duplicate keys.
Source: MQAR.targets forbids duplicate writes if task.overwrites is zero. -/
theorem recall_duplicate_control :
    recallFunction ⟨256, 2, false⟩ [1, 36, 292, 36, 293, 36] =
      [1, 36, 292, 36, 293, 36, 0] := by decide

/-- Fillers and earlier distinct queries do not alter the current table lookup.
Source: MQAR.targets scans the whole post-table region and supervises each query. -/
theorem recall_multiquery_control :
    recallFunction ⟨256, 2, false⟩ [1, 36, 292, 37, 293, 37, 294, 36] =
      [1, 36, 292, 37, 293, 37, 294, 36, 292] := by decide

/-- Repeating a query is rejected; Source: MQAR.targets requires distinct query keys. -/
theorem recall_repeat_query_control :
    recallNext ⟨256, 2, false⟩ [1, 36, 292, 37, 293, 36, 36] = pad := by decide

/-- The final-query grammar is inhabited by raw, adjacent key/value tokens.
Source: a two-record MQAR instance of the same parser used for the Basis modes. -/
theorem recallPrefix_example :
    RecallPrefix ⟨256, 2, false⟩ 64 [1, 36, 292, 37, 293, 36] := by
  exact ⟨by decide, 292, by decide⟩

end Transformer.Basis
