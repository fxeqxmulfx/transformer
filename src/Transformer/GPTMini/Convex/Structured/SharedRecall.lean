import Transformer.GPTMini.Convex.Structured.RecallGap
import Transformer.GPTMini.Convex.Structured.BindingInterface
import Transformer.Basis.Encoding.Tasks

/-!
# Actual finite learned all-pair head solves full raw Basis recall

Source: complete independent MQAR raw parsing at 9e6661b, genuine finite
shared weights at b127359, whole raw gap/confidence at e5e8823 and the
actual checked all-pair integer callback. Given task/mode weights handle
all 256 keys/values, the complete eight/sixteen-record table, arbitrary
valid query/filler positions and chronological hard-mode overwrites.

No paired input encoder, hard adjacency/table mask, parsed record or
correct logits enter inference. All raw pairs and latent matching/value
channels retain finite positive probability. The actual decoder margin
comes from its entire true joint distribution. This head-level theorem
is not yet a full mixed-head tensor/prenorm/residual/tied model result,
floating-point equivalence or successful AdamW training. The globally
convex complete likelihood uses data-derived route/channel supervision;
ordinary output-only CE is not covered by that convexity guarantee.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface Transformer.GPTMini.Semantics
open scoped Classical
noncomputable section

/-- One actual finite common vocabulary/head assignment per Basis recall mode, shared by every raw prefix.
Source: true learned Q/K/value/position/chronology/binding fields at gain log(10^12), with only the benchmark table size differing. -/
def recallSharedParameters (mode : Mode) : BindingParameters 548 64 :=
  recallBindingParameters (if mode = .easy then 8 else 16) recallBindingGain

/-- The actual standalone freely learned all-pair head exposes the complete append-one integer-list signature.
Source: true checked causal raw pointer scores and deterministic whole-vocabulary greedy decoding, without semantic inference code. -/
def recallSharedFunction (mode : Mode) : Tokens → Tokens :=
  bindingFunction (by decide) (by decide) (recallSharedParameters mode)

/-- Complete raw parsing derives a true strict actual score ordering against every real vocabulary competitor.
Source: whole raw finite-weight confidence, exact compact joint decoding and the all-vocabulary margin; no correct logits are assumed. -/
theorem recallBinding_raw_strict (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64)
    (answer : ℤ) (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query : Fin tokens.length, ∃ target : Fin 548,
      query.val + 1 = tokens.length ∧ (target.val : ℤ) = answer ∧
      ∀ rival : Fin 548, rival ≠ target →
        rawBindingScore (recallBindingParameters P recallBindingGain) tokens hcap query (vocabularyCode (by decide) rival) <
          rawBindingScore (recallBindingParameters P recallBindingGain) tokens hcap query (vocabularyCode (by decide) target) := by
  obtain ⟨query, previous, selected, hquery, ha, hp⟩ := recallBinding_raw_probability P rewrites recallBindingGain
    recallBindingGain_pos.le tokens hcap answer hanswer
  have hc : (10 / 11 : ℝ) < rawBindingProbability (recallBindingParameters P recallBindingGain)
      tokens hcap query (recallBindingTarget tokens previous selected) := by
    linarith [recallBindingGain_tail, (Real.exp_pos (-recallBindingGain)).le]
  let : Nonempty (Fin tokens.length) := ⟨query⟩
  have hn : ∀ z : RawBindingConfiguration tokens,
      0 ≤ rawBindingProbability (recallBindingParameters P recallBindingGain) tokens hcap query z :=
    fun z => (rawBindingProbability_pos _ _ _ _ z).le
  have hs := rawBindingProbability_sum (recallBindingParameters P recallBindingGain) tokens hcap query
  refine ⟨query, tokens.get selected, hquery, ha, ?_⟩
  intro rival hrival
  have hne : vocabularyCode (by decide) (tokens.get selected) ≠ vocabularyCode (by decide) rival := by
    intro he
    exact hrival ((vocabularyCode_injective (by decide) he).symm)
  rw [rawBindingScore_joint, rawBindingScore_joint]
  exact weightedOutput_strict _ (fun z : RawBindingConfiguration tokens => z.2.2)
    (recallBindingTarget tokens previous selected) _ _ hn hs rfl hne hc

example : recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by decide, by decide⟩

/-- The actual original greedy rule on a genuine nonempty raw prefix returns its independently parsed recall value ID.
Source: fully derived raw strict margins and the uniquely final query position, with the real fixed vocabulary decoder. -/
theorem recallBinding_best (P : ℕ) (rewrites : Bool) (head : Fin 548) (tail : List (Fin 548))
    (hcap : (head :: tail).length ≤ 64) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    ((bestToken (show 0 < 548 by decide) (fun token : Fin 548 => rawBindingScore (recallBindingParameters P recallBindingGain)
      (head :: tail) hcap (bindingFinalPosition tail) (vocabularyCode (by decide) token))).val : ℤ) = answer := by
  obtain ⟨query, target, hquery, ha, hstrict⟩ := recallBinding_raw_strict P rewrites (head :: tail) hcap answer hanswer
  have he : query = bindingFinalPosition tail :=
    bindingFinalPosition_unique tail query (by simpa only [List.length_cons] using hquery)
  rw [he] at hstrict
  have hb := bestToken_of_strict (show 0 < 548 by decide) (fun token : Fin 548 => rawBindingScore (recallBindingParameters P recallBindingGain)
    (head :: tail) hcap (bindingFinalPosition tail) (vocabularyCode (by decide) token)) target hstrict
  rw [hb]
  exact ha

example : recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by decide, by decide⟩

/-- Actual checked inference from one finite shared weight table predicts every required raw recall answer in either Basis mode.
Source: independent successful full parsing, derived encoding/context checks and genuine all-pair finite-weight greedy correctness. -/
theorem recallShared_next (mode : Mode) (tokens : Tokens) (hprefix : TaskPrefix mode .recall tokens) :
    bindingNext (by decide) (by decide) (recallSharedParameters mode) tokens = taskNext mode .recall tokens := by
  obtain ⟨head, tail, he, hlen⟩ := Encoding.recall_inputs mode tokens hprefix
  obtain ⟨answer, ha⟩ := hprefix.2
  have hcap : (head :: tail).length ≤ 64 := by simpa only [List.length_cons] using hlen
  have hfinite : recallAnswer ⟨256, (if mode = .easy then 8 else 16), (if mode = .easy then false else true)⟩
      (decodeTokens (head :: tail)) = some answer := by
    rw [decode_encode he]
    cases mode <;> exact ha
  have hnext : taskNext mode .recall tokens = answer := by
    cases mode <;> exact recallNext_of_answer _ _ _ ha
  rw [bindingNext_of_encode (by decide) (by decide) (recallSharedParameters mode) tokens head tail he hlen, hnext]
  exact recallBinding_best _ _ head tail hcap answer hfinite

example : TaskPrefix .easy .recall (bindingPrefix 292 293) := ⟨by decide, 292, by decide⟩

/-- The genuine learned raw all-pair callback solves the entire Basis recall grammar in both modes, including every hard overwrite.
Source: actual checked predictions on all independent task-valid prefixes, with no route, encoder or correct-output premise. -/
theorem recallShared_solves (mode : Mode) : SolvesTask (recallSharedFunction mode) mode .recall := by
  apply (solvesTask_extend_iff _ mode .recall).mpr
  intro tokens hprefix
  exact recallShared_next mode tokens hprefix

/-- The actual learned callback distinguishes valid equal-multiset inputs whose first two raw values are swapped.
Source: whole raw task correctness and the independent original MQAR bindings; the final query and visible token multiset are unchanged. -/
theorem recallShared_binding_control :
    recallSharedFunction .easy (bindingPrefix 292 293) = bindingPrefix 292 293 ++ [292] ∧
    recallSharedFunction .easy (bindingPrefix 293 292) = bindingPrefix 293 292 ++ [293] := by
  have ha := recallShared_next .easy (bindingPrefix 292 293) ⟨by decide, 292, by decide⟩
  have hb := recallShared_next .easy (bindingPrefix 293 292) ⟨by decide, 293, by decide⟩
  have hna : taskNext .easy .recall (bindingPrefix 292 293) = 292 := by decide
  have hnb : taskNext .easy .recall (bindingPrefix 293 292) = 293 := by decide
  rw [hna] at ha
  rw [hnb] at hb
  exact ⟨congrArg (fun token : ℤ => bindingPrefix 292 293 ++ [token]) ha,
    congrArg (fun token : ℤ => bindingPrefix 293 292 ++ [token]) hb⟩

/-- The genuine sixteen-write hard head distinguishes chronological overwrite orders with identical visible token multisets.
Source: the independent full hard-mode overwrite pair and actual finite shared learned all-pair inference above. -/
theorem recallShared_overwrite_control :
    recallSharedFunction .hard (rewritePrefix 292 300) = rewritePrefix 292 300 ++ [300] ∧
    recallSharedFunction .hard (rewritePrefix 300 292) = rewritePrefix 300 292 ++ [292] := by
  obtain ⟨hp, hq, _, ha, hb⟩ := recall_overwrite_pair
  have hpa := recallShared_next .hard (rewritePrefix 292 300) hp
  have hqa := recallShared_next .hard (rewritePrefix 300 292) hq
  rw [ha] at hpa
  rw [hb] at hqa
  exact ⟨congrArg (fun token : ℤ => rewritePrefix 292 300 ++ [token]) hpa,
    congrArg (fun token : ℤ => rewritePrefix 300 292 ++ [token]) hqa⟩

/-- Every actual recall callback has exactly the required append-one output length on its entire raw integer domain.
Source: the genuine learned binding interface, including invalid and overlong inputs. -/
theorem recallShared_length (mode : Mode) (tokens : Tokens) : (recallSharedFunction mode tokens).length = tokens.length + 1 :=
  bindingFunction_length _ _ _ _

/-- Every original integer token remains in its original order through the actual learned recall callback.
Source: the genuine checked all-pair head's prefix-preserving continuation contract. -/
theorem recallShared_prefix (mode : Mode) (tokens : Tokens) : (recallSharedFunction mode tokens).take tokens.length = tokens :=
  bindingFunction_prefix _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
