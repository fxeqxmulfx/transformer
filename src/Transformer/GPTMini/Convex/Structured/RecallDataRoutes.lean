import Transformer.GPTMini.Convex.Structured.RecallPositions
import Mathlib.Data.List.MinMax
import Mathlib.Data.List.FinRange

/-!
# Computable raw-data route supervision for the learned recall head

Source: complete independent MQAR raw parsing/latest-write semantics at
9e6661b and actual learned mixed Basis capability at d7be618. Training
labels scan unchanged physical table-value slots and choose the latest
whose real predecessor token matches the raw query. This is a finite
Nat-ranked list argmax, not a choice of an unknown correctness witness,
a learned head search or an exponential prefix feature bank.

All label-generation inputs are raw data and the benchmark record count.
No model parameters, hidden states, attention scores or desired logits
enter the scan. Its table/adjacency restrictions belong exclusively to
training supervision; actual inference still considers all visible pairs
with free learned potentials. Full successful parsing derives correctness
for both rewrite settings, every record count and arbitrary valid fillers.
The scan and subsequent label generation must be charged in FLOP/work
accounting; these data theorems do not assert zero preprocessing cost.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.Semantics

/-- The actual physical predecessor index, with zero retained only as a total data-side fallback.
Source: raw BOS-offset record serialization; true table-value slots have a strictly positive position. -/
def recallDataPrevious {n : ℕ} (record : Fin n) : Fin n :=
  ⟨record.val - 1, by have hr := record.isLt; omega⟩

/-- The data-side previous index is genuinely adjacent at every positive raw record position.
Source: ordinary bounded Nat predecessor arithmetic, without a prepared paired-token input. -/
theorem recallDataPrevious_adjacent {n : ℕ} (record : Fin n) (hpos : 0 < record.val) :
    (recallDataPrevious record).val + 1 = record.val := by
  change record.val - 1 + 1 = record.val
  omega

example : (0 : ℕ) < (4 : Fin 6).val := by decide

/-- A genuinely adjacent parsed raw predecessor is exactly the one used by the computable training scan.
Source: physical predecessor uniqueness, without using a model route or hidden representation. -/
theorem recallDataPrevious_eq {n : ℕ} (previous record : Fin n) (hprevious : previous.val + 1 = record.val) :
    recallDataPrevious record = previous := by
  apply Fin.ext
  change record.val - 1 = previous.val
  omega

example : (3 : Fin 6).val + 1 = (4 : Fin 6).val := by decide

/-- Data-only record ranking favors later genuine table values whose actual preceding key matches the raw query.
Source: complete MQAR serialization/latest-write supervision; no model scores or parameter-dependent routes enter this Nat function. -/
def recallDataRank (P : ℕ) (tokens : List (Fin 548)) (query record : Fin tokens.length) : ℕ :=
  if recallTableValuePosition P record.val ∧ tokens.get (recallDataPrevious record) = tokens.get query
  then record.val + 1 else 0

/-- The actual deterministic training scan enumerates physical positions once and selects their maximal data rank.
Source: List.argmax on finite raw indices and Nat comparisons; a query-index fallback makes the function total outside the valid grammar. -/
def recallDataSelected (P : ℕ) (tokens : List (Fin 548)) (query : Fin tokens.length) : Fin tokens.length :=
  ((List.finRange tokens.length).argmax (recallDataRank P tokens query)).getD query

/-- A real valid matching raw table value receives exactly its positive chronological data rank.
Source: the actual Nat scan predicate, independent of all learned embedding/head weights. -/
theorem recallDataRank_hit (P : ℕ) (tokens : List (Fin 548)) (query record : Fin tokens.length)
    (hslot : recallTableValuePosition P record.val) (hkey : tokens.get (recallDataPrevious record) = tokens.get query) :
    recallDataRank P tokens query record = record.val + 1 := by
  unfold recallDataRank
  rw [ite_eq_left ⟨hslot, hkey⟩]

example : recallTableValuePosition 2 (4 : Fin 6).val ∧
    recallOverwriteTokens.get (recallDataPrevious (4 : Fin 6)) = recallOverwriteTokens.get (5 : Fin 6) := ⟨by decide, by decide⟩

/-- Genuine full raw parsing makes the real last matching record the unique strict maximizer of the actual deterministic data scan.
Source: parsed raw adjacency, actual key IDs and complete latest-write chronology; no model confidence or correct-route assumption is used. -/
theorem recallDataRank_parsed_strict (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer)
    (query previous selected : Fin tokens.length) (key : Fin 256)
    (hq : tokens.get query = recallKeyId key) (hp : previous.val + 1 = selected.val)
    (hk : tokens.get previous = recallKeyId key) (hs : recallTableValuePosition P selected.val)
    (hlast : RecallRawLatestWrite P tokens.get selected key) :
    ∀ rival : Fin tokens.length, rival ≠ selected → recallDataRank P tokens query rival < recallDataRank P tokens query selected := by
  have hprev := recallDataPrevious_eq previous selected hp
  have hmatch : tokens.get (recallDataPrevious selected) = tokens.get query := by rw [hprev]; exact hk.trans hq.symm
  rw [recallDataRank_hit P tokens query selected hs hmatch]
  intro rival hne
  unfold recallDataRank
  split_ifs with hr
  · have hadj := recallDataPrevious_adjacent rival hr.1.1
    have hkey : tokens.get (recallDataPrevious rival) = recallKeyId key := hr.2.trans hq
    have hlate := recallBinding_matching_last P rewrites tokens answer hanswer selected (recallDataPrevious rival) rival key
      hlast hadj hr.1 hkey
    have hvalueNe : rival.val ≠ selected.val := fun h => hne (Fin.ext h)
    omega
  · omega

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 ∧
    recallOverwriteTokens.get (5 : Fin 6) = recallKeyId 0 ∧ (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧
    recallOverwriteTokens.get (3 : Fin 6) = recallKeyId 0 ∧ recallTableValuePosition 2 (4 : Fin 6).val ∧
    RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) (0 : Fin 256) :=
  ⟨by decide, by decide, by decide, by decide, by decide, recallOverwriteTokens_raw_latest⟩

/-- The actual computable data scan selects the genuine latest parsed record from raw inputs alone.
Source: independently derived strict chronological data ranks and the exact finite List.argmax characterization. -/
theorem recallDataSelected_parsed (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer)
    (query previous selected : Fin tokens.length) (key : Fin 256)
    (hq : tokens.get query = recallKeyId key) (hp : previous.val + 1 = selected.val)
    (hk : tokens.get previous = recallKeyId key) (hs : recallTableValuePosition P selected.val)
    (hlast : RecallRawLatestWrite P tokens.get selected key) : recallDataSelected P tokens query = selected := by
  have hstrict := recallDataRank_parsed_strict P rewrites tokens answer hanswer query previous selected key hq hp hk hs hlast
  have harg : (List.finRange tokens.length).argmax (recallDataRank P tokens query) = some selected := by
    apply List.argmax_eq_some_iff.mpr
    refine ⟨List.mem_finRange selected, ?_, ?_⟩
    · intro rival hr
      by_cases he : rival = selected
      · rw [he]
      · exact (hstrict rival he).le
    · intro rival hr hle
      by_cases he : rival = selected
      · rw [he]
      · exact False.elim (Nat.not_le_of_lt (hstrict rival he) hle)
  unfold recallDataSelected
  rw [harg, Option.getD_some]

example : recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 ∧
    recallOverwriteTokens.get (5 : Fin 6) = recallKeyId 0 ∧ (3 : Fin 6).val + 1 = (4 : Fin 6).val ∧
    recallOverwriteTokens.get (3 : Fin 6) = recallKeyId 0 ∧ recallTableValuePosition 2 (4 : Fin 6).val ∧
    RecallRawLatestWrite 2 recallOverwriteTokens.get (4 : Fin 6) (0 : Fin 256) :=
  ⟨by decide, by decide, by decide, by decide, by decide, recallOverwriteTokens_raw_latest⟩

/-- The real finite data scan selects the later overwrite's unchanged raw physical position.
Source: executable Nat-ranked label generation on the original six-token overwrite control. -/
theorem recallDataSelected_overwrite : recallDataSelected 2 recallOverwriteTokens (5 : Fin 6) = (4 : Fin 6) := by
  change recallDataSelected 2 [(1 : Fin 548), 36, 292, 36, 293, 36] (5 : Fin 6) = (4 : Fin 6)
  decide

/-- Real distinct raw queries choose their own data records while a later even-position value filler is excluded from the table.
Source: executable label generation on a valid two-record prefix with distinct queries and a post-table value filler. -/
theorem recallDataSelected_queries :
    recallDataSelected 2 [(1 : Fin 548), 36, 292, 37, 293, 36, 300, 37] (5 : Fin 8) = (2 : Fin 8) ∧
    recallDataSelected 2 [(1 : Fin 548), 36, 292, 37, 293, 36, 300, 37] (7 : Fin 8) = (4 : Fin 8) := by decide

/-- The total data-side predecessor fallback remains a genuine bounded index even on a one-token invalid training input.
Source: actual finite Nat predecessor, before any table/matching validity test. -/
theorem recallDataPrevious_singleton : recallDataPrevious (0 : Fin 1) = (0 : Fin 1) := by decide

/-- The data scan is actually defined on a one-token input with no matching record and retains its query-index fallback.
Source: true finite zero-rank argmax and the checked physical index, outside the supervised grammar. -/
theorem recallDataSelected_singleton : recallDataSelected 0 [(1 : Fin 548)] (0 : Fin 1) = (0 : Fin 1) := by decide

end Transformer.GPTMini.Convex.Structured
