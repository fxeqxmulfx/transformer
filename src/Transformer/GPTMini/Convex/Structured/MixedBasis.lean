import Transformer.GPTMini.Convex.Structured.MixedInterface
import Transformer.GPTMini.Convex.Structured.SharedDepth
import Transformer.GPTMini.Convex.Structured.SharedParity

/-!
# Full raw Basis correctness of the actual freely learned mixed head

Source: independent raw depth/parity semantics at 6c91a98/91299ae,
complete raw recall parsing/confidence at e5e8823, genuine mixed model
and globally convex complete training at d436526. Separate task/mode
weights follow the original Basis benchmark. Inference uses the same
actual compact learned two-head callback throughout; no task, reference
rule, parsed binding, desired channel or correct logit enters its forward.

Given ordinary finite raw parameter assignments solve every independent
validated prefix in all six task/mode combinations, plus two real parity
generation calls. Both heads retain positive mass. Raw order, binding
and overwrite controls reproduce the distinctions the previous bag
encoder lost. Exact integer append-one/prefix properties hold everywhere.
This completes mixed-head capacity, not tensor/prenorm/residual/tied
block realization, output-only CE convexity, floating-point equivalence
or successful AdamW training. Complete data likelihood is the proved
convex objective; auxiliary data-target generation remains explicit.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- One genuine finite common raw mixed weight assignment per independently trained Basis recipe, not per input prefix.
Source: actual shared state/value or all-pair binding witnesses; labels/rules select weights only, while learned inference remains unchanged. -/
def basisMixedParameters (mode : Mode) (task : Task) : BindingParameters (vocabularySize task) (contextSize task) :=
  match task with
  | .depth => mixedReferenceParameters depthSharedRule
      (fun state => vocabularyCode (by decide) (depthReferenceLabel (if mode = .easy then 2 else 4) state)) 0 referenceGain
  | .recall => recallBindingParameters (if mode = .easy then 8 else 16) recallBindingGain
  | .parity => mixedReferenceParameters paritySharedRule
      (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 referenceGain

/-- The actual mixed learned model consumes only an integer list after its task/mode weight assignment is fixed.
Source: genuine mixedFunction with the real task vocabulary and context; the public interface has no semantic auxiliary inputs. -/
def basisMixedFunction (mode : Mode) (task : Task) : Tokens → Tokens :=
  mixedFunction (by cases task <;> decide) (by cases task <;> decide) (basisMixedParameters mode task)

/-- Actual mixed finite-weight prediction equals the independently specified next answer for every raw Basis task/mode prefix.
Source: full genuine mixed greedy margins, independent depth/parity reference equivalence, full raw recall parsing and exact encoding checks. -/
theorem basisMixed_next (mode : Mode) (task : Task) (tokens : Tokens) (hprefix : TaskPrefix mode task tokens) :
    mixedNext (by cases task <;> decide) (by cases task <;> decide) (basisMixedParameters mode task) tokens = taskNext mode task tokens := by
  cases task with
  | depth =>
      obtain ⟨head, tail, he, hlen⟩ := Encoding.depth_inputs mode tokens hprefix
      have hT : (head :: tail).length ≤ 128 := by simpa only [List.length_cons] using hlen
      change mixedNext (by decide) (by decide) (mixedReferenceParameters (C := 128) depthSharedRule
        (fun state => vocabularyCode (by decide) (depthReferenceLabel (if mode = .easy then 2 else 4) state)) 0 referenceGain) tokens = _
      rw [mixedReferenceNext_of_encode (by decide) (by decide) depthSharedRule
        (depthReferenceLabel (if mode = .easy then 2 else 4)) 0 tokens head tail he hlen hT,
        depthShared_run, decode_encode he]
      apply depthRawReference_next _ 128 _ _ tokens hprefix
      · cases mode <;> decide
      · cases mode <;> decide
  | «recall» =>
      obtain ⟨head, tail, he, hlen⟩ := Encoding.recall_inputs mode tokens hprefix
      obtain ⟨answer, ha⟩ := hprefix.2
      have hcap : (head :: tail).length ≤ 64 := by simpa only [List.length_cons] using hlen
      have hfinite : recallAnswer ⟨256, (if mode = .easy then 8 else 16), (if mode = .easy then false else true)⟩
          (decodeTokens (head :: tail)) = some answer := by
        rw [decode_encode he]
        cases mode <;> exact ha
      have hnext : taskNext mode .recall tokens = answer := by cases mode <;> exact recallNext_of_answer _ _ _ ha
      change mixedNext (by decide) (by decide) (recallBindingParameters (if mode = .easy then 8 else 16) recallBindingGain) tokens = _
      rw [mixedNext_of_encode (by decide) (by decide) _ tokens head tail he hlen, hnext]
      exact mixedRecall_best _ _ head tail hcap answer hfinite
  | parity =>
      obtain ⟨head, tail, he, hlen⟩ := Encoding.parity_inputs mode tokens hprefix
      have hT : (head :: tail).length ≤ 128 := by simp only [List.length_cons]; omega
      change mixedNext (by decide) (by decide) (mixedReferenceParameters (C := 19) paritySharedRule
        (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 referenceGain) tokens = _
      rw [mixedReferenceNext_of_encode (by decide) (by decide) paritySharedRule paritySharedLabel 0
        tokens head tail he hlen hT, paritySharedLabel_id, parityShared_run, decode_encode he]
      exact parityReference_next 19 tokens hprefix

example : TaskPrefix .hard .recall (rewritePrefix 292 300) := ⟨by decide, 300, by decide⟩

/-- Given actual finite learned mixed weights solve every validated raw prefix in each of the six complete Basis recipes.
Source: true mixedNext correctness above and the real append-one adapter; no encoder, route, head-selection or correct-logit premise is used. -/
theorem basisMixed_solves (mode : Mode) (task : Task) : SolvesTask (basisMixedFunction mode task) mode task := by
  apply (solvesTask_extend_iff _ mode task).mpr
  intro tokens hprefix
  exact basisMixed_next mode task tokens hprefix

/-- Two actual mixed-model calls generate parity's complete label/EOS output for every legal raw bit word.
Source: full learned mixture correctness at both independent Basis supervision positions, with the first prediction reused as real input. -/
theorem basisMixed_parity_twice (mode : Mode) (bits : List Bool) (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    basisMixedFunction mode .parity (basisMixedFunction mode .parity (parityPrompt bits)) =
      parityPrompt bits ++ [parityLabel bits, eos] :=
  solvesTask_parity_twice _ mode (basisMixed_solves mode .parity) bits hmin hmax

example : 1 ≤ ([true, false, true] : List Bool).length ∧ ([true, false, true] : List Bool).length ≤ 16 := by decide

/-- Every real mixed Basis callback satisfies the requested length on all integer lists, including unsupervised and invalid ones.
Source: the actual learned mixedFunction append-one interface, independent of recipe validity or the finite witness's correctness. -/
theorem basisMixed_length (mode : Mode) (task : Task) (tokens : Tokens) :
    (basisMixedFunction mode task tokens).length = tokens.length + 1 := mixedFunction_length _ _ _ _

/-- Every input integer and its original order are preserved through each actual freely learned mixed Basis continuation.
Source: genuine mixedFunction prefix preservation, without vocabulary conversion assumptions on the input. -/
theorem basisMixed_prefix (mode : Mode) (task : Task) (tokens : Tokens) :
    (basisMixedFunction mode task tokens).take tokens.length = tokens := mixedFunction_prefix _ _ _ _

/-- The real full eight-record mixed recall model distinguishes swapped neighboring values under the same query and visible multiset.
Source: actual complete mixed Basis correctness and the independent original easy-mode raw binding pair. -/
theorem basisMixed_binding_control :
    basisMixedFunction .easy .recall (bindingPrefix 292 293) = bindingPrefix 292 293 ++ [292] ∧
    basisMixedFunction .easy .recall (bindingPrefix 293 292) = bindingPrefix 293 292 ++ [293] := by
  obtain ⟨hp, hq, _, ha, hb⟩ := recall_binding_pair
  have hpa := basisMixed_solves .easy .recall _ hp
  have hqa := basisMixed_solves .easy .recall _ hq
  unfold taskFunction extend at hpa hqa
  rw [ha] at hpa
  rw [hb] at hqa
  exact ⟨hpa, hqa⟩

/-- The genuine full sixteen-record hard mixed model respects chronological last-write order on identical visible token multisets.
Source: independently validated hard raw overwrite pair and actual whole-mixture Basis correctness, without semantic forward preprocessing. -/
theorem basisMixed_overwrite_control :
    basisMixedFunction .hard .recall (rewritePrefix 292 300) = rewritePrefix 292 300 ++ [300] ∧
    basisMixedFunction .hard .recall (rewritePrefix 300 292) = rewritePrefix 300 292 ++ [292] := by
  obtain ⟨hp, hq, _, ha, hb⟩ := recall_overwrite_pair
  have hpa := basisMixed_solves .hard .recall _ hp
  have hqa := basisMixed_solves .hard .recall _ hq
  unfold taskFunction extend at hpa hqa
  rw [ha] at hpa
  rw [hb] at hqa
  exact ⟨hpa, hqa⟩

/-- Actual learned mixed depth predictions retain the required raw order distinction despite equal token multiplicities.
Source: whole raw E2 correctness and the independently validated depth_order_pair, without a bag-encoded input. -/
theorem basisMixed_order_control :
    basisMixedFunction .easy .depth [1, 9, 9, 10, 10] = [1, 9, 9, 10, 10, 16] ∧
    basisMixedFunction .easy .depth [1, 9, 10, 9, 10] = [1, 9, 10, 9, 10, 15] := by
  obtain ⟨hp, hq, _, ha, hb⟩ := depth_order_pair
  have hpa := basisMixed_solves .easy .depth _ hp
  have hqa := basisMixed_solves .easy .depth _ hq
  unfold taskFunction extend at hpa hqa
  rw [ha] at hpa
  rw [hb] at hqa
  exact ⟨hpa, hqa⟩

end
end Transformer.GPTMini.Convex.Structured
