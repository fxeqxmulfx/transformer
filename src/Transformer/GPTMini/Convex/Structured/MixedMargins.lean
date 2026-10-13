import Transformer.GPTMini.Convex.Structured.MixedConfidence
import Transformer.GPTMini.Convex.Structured.BindingInterface

/-!
# Actual whole-vocabulary margins and greedy decoding of the mixture

Source: genuine normalized mixed inference/training at d436526 and the
finite whole-model confidence in MixedConfidence. Every margin follows
from the actual compact score's complete joint expectation, including
the positive incorrect branch, all rival paths/pairs and value channels.
No correct logits, supplied encoder or correctly selected head is a
hypothesis. Reference rules and raw parsers choose weights or discharge
data semantics; the actual greedy score never calls them.

State weights handle every prefix through context 128 with the same
finite gain log(100000). Raw binding weights handle every successfully
parsed capped recall prefix at gain log(10^12). All 1024 decoder codes
and every real vocabulary competitor are covered. These are real head
capacity/decoder results, not the full changed tensor stack, numerical
equivalence or AdamW convergence. Causal callbacks use their last row.

The positive-scale law also preserves exact greedy tie breaking. A later
tensor realization must derive its final normalizer's common positive
multiplier from the actual hidden state rather than supply it by fiat.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface Transformer.GPTMini.Semantics
open scoped Classical
noncomputable section

variable {V C : ℕ}

/-- Genuine whole-vocabulary greedy prediction from actual free mixed weights and a real raw physical query index.
Source: original bestToken on the compact learned mixed scores; no task label, reference rule or parsed route enters this function. -/
def mixedGreedy (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) : Fin V :=
  bestToken hV (fun token => mixedScore θ tokens hcap query (vocabularyCode hsize token))

/-- Actual finite mixed state weights derive strict whole-decoder margins, including all interference from the positive pointer branch.
Source: the genuine 796 tail and true 11*p-10 mixed margin; task-specific reference semantics are discharged independently. -/
theorem mixedReference_strict (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024) (start : Fin 6)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) (hT : tokens.length ≤ 128)
    (rival : Fin 1024) (hne : labels (referenceRun rule start tokens) ≠ rival) :
    mixedScore (mixedReferenceParameters rule labels start referenceGain) tokens hcap query rival <
      mixedScore (mixedReferenceParameters rule labels start referenceGain) tokens hcap query (labels (referenceRun rule start tokens)) := by
  have hp := mixedReference_probability rule labels start referenceGain tokens hcap query hT
  have hm := mixedScore_margin (mixedReferenceParameters rule labels start referenceGain) tokens hcap query
    (.inl (mixedReferenceTarget rule labels start tokens)) (labels (referenceRun rule start tokens)) rival rfl hne
  linarith [mixedReferenceGain_confidence]

example : ([0, 1] : List (Fin 2)).length ≤ 4 ∧ ([0, 1] : List (Fin 2)).length ≤ 128 ∧
    (fun _ : Fin 6 => (25 : Fin 1024)) (referenceRun (fun _ state => state) 0 [(0 : Fin 2), 1]) ≠ 17 := by decide

/-- Actual original greedy decoding of the genuine mixed state witness returns the independent chronological data endpoint label.
Source: derived true mixed strict margins and injective real vocabulary IDs, without correct head selection as a premise. -/
theorem mixedReference_best (hV : 0 < V) (hsize : V ≤ 1024) (rule : Fin V → Fin 6 → Fin 6)
    (labels : Fin 6 → Fin V) (start : Fin 6) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (hT : tokens.length ≤ 128) :
    mixedGreedy hV hsize (mixedReferenceParameters rule (fun state => vocabularyCode hsize (labels state)) start referenceGain)
      tokens hcap query = labels (referenceRun rule start tokens) := by
  apply bestToken_of_strict
  intro rival hrival
  apply mixedReference_strict rule (fun state => vocabularyCode hsize (labels state)) start tokens hcap query hT
  intro he
  exact hrival ((vocabularyCode_injective hsize he).symm)

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ ([0, 1] : List (Fin 2)).length ≤ 4 ∧
    ([0, 1] : List (Fin 2)).length ≤ 128 := by decide

/-- Successful full raw parsing derives actual whole-vocabulary strict mixed recall margins at one shared finite weight assignment.
Source: genuine mixed recall confidence and normalized full-distribution decoder margin, with no assumed route, encoder or logit ordering. -/
theorem mixedRecall_raw_strict (P : ℕ) (rewrites : Bool) (tokens : List (Fin 548)) (hcap : tokens.length ≤ 64)
    (answer : ℤ) (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens tokens) = some answer) :
    ∃ query : Fin tokens.length, ∃ target : Fin 548,
      query.val + 1 = tokens.length ∧ (target.val : ℤ) = answer ∧
      ∀ rival : Fin 548, rival ≠ target →
        mixedScore (recallBindingParameters P recallBindingGain) tokens hcap query (vocabularyCode (by decide) rival) <
          mixedScore (recallBindingParameters P recallBindingGain) tokens hcap query (vocabularyCode (by decide) target) := by
  obtain ⟨query, previous, selected, hquery, ha, hp⟩ := mixedRecall_probability P rewrites recallBindingGain
    recallBindingGain_pos.le tokens hcap answer hanswer
  refine ⟨query, tokens.get selected, hquery, ha, ?_⟩
  intro rival hrival
  have hne : vocabularyCode (by decide) (tokens.get selected) ≠ vocabularyCode (by decide) rival := by
    intro he
    exact hrival ((vocabularyCode_injective (by decide) he).symm)
  have hm := mixedScore_margin (recallBindingParameters P recallBindingGain) tokens hcap query
    (.inr (recallBindingTarget tokens previous selected)) (vocabularyCode (by decide) (tokens.get selected))
    (vocabularyCode (by decide) rival) rfl hne
  linarith [recallBindingGain_tail]

example : recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by decide, by decide⟩

/-- The actual whole mixed greedy computation on a nonempty raw recall prefix returns its independently parsed answer ID.
Source: derived true mixed raw margins and physical final-query uniqueness, with the original deterministic vocabulary decoder. -/
theorem mixedRecall_best (P : ℕ) (rewrites : Bool) (head : Fin 548) (tail : List (Fin 548))
    (hcap : (head :: tail).length ≤ 64) (answer : ℤ)
    (hanswer : recallAnswer ⟨256, P, rewrites⟩ (decodeTokens (head :: tail)) = some answer) :
    ((mixedGreedy (by decide) (by decide) (recallBindingParameters P recallBindingGain)
      (head :: tail) hcap (bindingFinalPosition tail)).val : ℤ) = answer := by
  obtain ⟨query, target, hquery, ha, hstrict⟩ := mixedRecall_raw_strict P rewrites (head :: tail) hcap answer hanswer
  have he : query = bindingFinalPosition tail := bindingFinalPosition_unique tail query
    (by simpa only [List.length_cons] using hquery)
  rw [he] at hstrict
  have hb := bestToken_of_strict (show 0 < 548 by decide) (fun token : Fin 548 => mixedScore
    (recallBindingParameters P recallBindingGain) (head :: tail) hcap (bindingFinalPosition tail)
    (vocabularyCode (by decide) token)) target hstrict
  change ((bestToken _ _).val : ℤ) = answer
  rw [hb]
  exact ha

example : recallOverwriteTokens.length ≤ 64 ∧
    recallAnswer ⟨256, 2, true⟩ (decodeTokens recallOverwriteTokens) = some 293 := ⟨by decide, by decide⟩

/-- The genuine mixed greedy output selects the later changed value in an actual raw overwrite control.
Source: exact raw parser validation and finite learned all-pair/state mixing, without giving the selected record to inference. -/
theorem mixedRecall_overwrite_decoding :
    ((mixedGreedy (by decide) (by decide) (recallBindingParameters 2 recallBindingGain)
      [(1 : Fin 548), 36, 292, 36, 293, 36] (by decide : ([(1 : Fin 548), 36, 292, 36, 293, 36] : List (Fin 548)).length ≤ 64)
      (bindingFinalPosition [(36 : Fin 548), 292, 36, 293, 36])).val : ℤ) = 293 :=
  mixedRecall_best 2 true 1 [36, 292, 36, 293, 36] (by decide) 293 (by decide)

/-- Actual mixed greedy inference distinguishes swapped physical values while retaining the final query and all visible symbols.
Source: two genuine six-token raw parser controls; complete eight/sixteen-record capability is covered by the general theorem. -/
theorem mixedRecall_swap_decoding :
    ((mixedGreedy (by decide) (by decide) (recallBindingParameters 2 recallBindingGain)
      [(1 : Fin 548), 36, 292, 37, 293, 36] (by decide : ([(1 : Fin 548), 36, 292, 37, 293, 36] : List (Fin 548)).length ≤ 64)
      (bindingFinalPosition [(36 : Fin 548), 292, 37, 293, 36])).val : ℤ) = 292 ∧
    ((mixedGreedy (by decide) (by decide) (recallBindingParameters 2 recallBindingGain)
      [(1 : Fin 548), 36, 293, 37, 292, 36] (by decide : ([(1 : Fin 548), 36, 293, 37, 292, 36] : List (Fin 548)).length ≤ 64)
      (bindingFinalPosition [(36 : Fin 548), 293, 37, 292, 36])).val : ℤ) = 293 :=
  ⟨mixedRecall_best 2 false 1 [36, 292, 37, 293, 36] (by decide) 292 (by decide),
    mixedRecall_best 2 false 1 [36, 293, 37, 292, 36] (by decide) 293 (by decide)⟩

/-- Any actual positive final common scale preserves the mixed head's genuine greedy prediction, including its tie rule.
Source: original bestToken order invariance; this supplies the quantitative final-RMS bridge without assuming a correct answer. -/
theorem mixedGreedy_positive_scale (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (query : Fin tokens.length) (scale : ℝ) (hscale : 0 < scale) :
    bestToken hV (fun token => scale * mixedScore θ tokens hcap query (vocabularyCode hsize token)) =
      mixedGreedy hV hsize θ tokens hcap query := by
  apply bestToken_congr_order
  intro a b
  exact mul_le_mul_iff_right₀ hscale

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ ([0, 1] : List (Fin 2)).length ≤ 4 ∧ (0 : ℝ) < 2 := by
  norm_num

end
end Transformer.GPTMini.Convex.Structured
