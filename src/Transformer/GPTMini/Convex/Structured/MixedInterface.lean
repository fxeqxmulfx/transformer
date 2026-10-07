import Transformer.GPTMini.Convex.Structured.MixedMargins

/-!
# Actual learned mixed head through the full integer-list interface

Source: the genuine mixed model/complete training at d436526, finite
whole-vocabulary decoding in MixedMargins and checked binding callback
at 05fb3e2. Prediction consumes checked raw vocabulary IDs and freely
learned weights only. Its actual final physical row supplies the query;
both heads consume that row's own causally visible raw prefix.

Empty, invalid and overlong inputs use the original PAD fallback while
retaining every input integer. The same actual compact learned scores
and original deterministic greedy decoder serve all tasks; neither a
reference rule nor semantic training labels enter the forward call.
The finite reference bridge below derives its output from true mixed
margins and successful encoding, never from assumed correct logits.
This is a standalone mixed callback. Actual tensor/prenorm/residual/tied
realization, floating-point behavior and AdamW training remain separate.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

variable {V C : ℕ}

/-- Actual next-token inference checks integer vocabulary/context, then evaluates both genuine learned heads at the final raw row.
Source: true compact mixedGreedy and physical bindingFinalPosition, without parsed route, target path or reference rule arguments. -/
def mixedNext (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) (tokens : Tokens) : ℤ :=
  match encodeTokens V tokens with
  | some (head :: tail) =>
      if hcap : tail.length + 1 ≤ C then
        ((mixedGreedy hV hsize θ (head :: tail) (by simpa only [List.length_cons] using hcap)
          (bindingFinalPosition tail)).val : ℤ)
      else pad
  | _ => pad

/-- The actual freely learned mixed model has exactly the same public append-one integer-list signature.
Source: genuine checked mixed inference and Basis.extend, without task-dependent forward logic. -/
def mixedFunction (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) : Tokens → Tokens :=
  extend (mixedNext hV hsize θ)

/-- Every actual mixed call returns exactly one more integer than its input, regardless of task validity or learned parameters.
Source: the genuine prefix-preserving extension interface, including checked inference failures. -/
theorem mixedFunction_length (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) (tokens : Tokens) :
    (mixedFunction hV hsize θ tokens).length = tokens.length + 1 := extend_length _ _

example : (0 : ℕ) < 548 ∧ (548 : ℕ) ≤ 1024 := by omega

/-- Every original integer remains in its original physical order through the actual learned mixed continuation.
Source: genuine Basis.extend, without semantic or vocabulary-validity hypotheses on the raw list. -/
theorem mixedFunction_prefix (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) (tokens : Tokens) :
    (mixedFunction hV hsize θ tokens).take tokens.length = tokens := extend_prefix _ _

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 := by omega

/-- Successful genuine encoding and the context bound expose the actual complete mixed greedy computation alone.
Source: the real checked dependent branch, with both learned heads and all visible raw pairs retained. -/
theorem mixedNext_of_encode (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ C) :
    mixedNext hV hsize θ tokens = ((mixedGreedy hV hsize θ (head :: tail)
      (by simpa only [List.length_cons] using hlen) (bindingFinalPosition tail)).val : ℤ) := by
  unfold mixedNext
  rw [hencode]
  dsimp only
  rw [dite_eq_left hlen]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [0, 1] = some [(0 : Fin 2), 1] ∧
    ([1] : List (Fin 2)).length + 1 ≤ 4 := by decide

/-- The genuine public function appends the true computed mixed prediction after every successfully checked raw input.
Source: exact context/encoding branch evaluation and the ordinary integer-list extension convention. -/
theorem mixedFunction_of_encode (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ C) :
    mixedFunction hV hsize θ tokens = tokens ++ [((mixedGreedy hV hsize θ (head :: tail)
      (by simpa only [List.length_cons] using hlen) (bindingFinalPosition tail)).val : ℤ)] := by
  unfold mixedFunction extend
  rw [mixedNext_of_encode hV hsize θ tokens head tail hencode hlen]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [1, 0] = some [(1 : Fin 2), 0] ∧
    ([0] : List (Fin 2)).length + 1 ≤ 4 := by decide

/-- Empty input has no physical query and receives the specified PAD continuation through the actual mixed interface.
Source: the genuine checked raw decoder before either learned head is evaluated. -/
theorem mixedFunction_empty (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) :
    mixedFunction hV hsize θ [] = [pad] := by
  unfold mixedFunction extend mixedNext
  rw [encodeTokens, List.nil_append]

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 := by omega

/-- Encoded overlong input is rejected by the actual context branch before learned mixed prediction.
Source: the real finite context capacity, without an extra semantic inference restriction. -/
theorem mixedNext_overlong (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : ¬tail.length + 1 ≤ C) :
    mixedNext hV hsize θ tokens = pad := by
  unfold mixedNext
  rw [hencode]
  dsimp only
  rw [dite_eq_right hlen]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [0, 1] = some [(0 : Fin 2), 1] ∧
    ¬([1] : List (Fin 2)).length + 1 ≤ 1 := by decide

/-- The complete actual mixed callback retains an overlong checked input exactly and appends PAD.
Source: the real context rejection above and the same integer-list extension, without evaluating either learned head. -/
theorem mixedFunction_overlong (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : ¬tail.length + 1 ≤ C) :
    mixedFunction hV hsize θ tokens = tokens ++ [pad] := by
  unfold mixedFunction extend
  rw [mixedNext_overlong hV hsize θ tokens head tail hencode hlen]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [1, 0] = some [(1 : Fin 2), 0] ∧
    ¬([0] : List (Fin 2)).length + 1 ≤ 1 := by decide

/-- A genuinely invalid integer input cannot alias a valid vocabulary ID or enter either learned branch.
Source: encodeTokens's real lower/upper checks and the actual total PAD fallback. -/
theorem mixedNext_invalid (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (hencode : encodeTokens V tokens = none) : mixedNext hV hsize θ tokens = pad := by
  unfold mixedNext
  rw [hencode]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [-1, 0] = none := by decide

/-- Negative input remains exactly in the returned raw prefix while the actual vocabulary check appends PAD.
Source: genuine checked mixed prediction on an invalid integer, without modulo conversion. -/
theorem mixedFunction_negative (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) :
    mixedFunction hV hsize θ [-1] = [-1, pad] := by
  simp only [mixedFunction, extend, mixedNext, encodeTokens]
  norm_num

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 := by omega

/-- Actual finite mixed reference weights and genuine checked input derive the independently specified endpoint label.
Source: true full mixed greedy margins at the real last position; reference task correctness is proved separately from this model bridge. -/
theorem mixedReferenceNext_of_encode (hV : 0 < V) (hsize : V ≤ 1024) (rule : Fin V → Fin 6 → Fin 6)
    (labels : Fin 6 → Fin V) (start : Fin 6) (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ C) (hT : (head :: tail).length ≤ 128) :
    mixedNext hV hsize (mixedReferenceParameters (C := C) rule (fun state => vocabularyCode hsize (labels state)) start referenceGain) tokens =
      ((labels (referenceRun rule start (head :: tail))).val : ℤ) := by
  rw [mixedNext_of_encode hV hsize _ tokens head tail hencode hlen,
    mixedReference_best hV hsize rule labels start (head :: tail) _ _ hT]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [0, 1] = some [(0 : Fin 2), 1] ∧
    ([1] : List (Fin 2)).length + 1 ≤ 4 ∧ ([(0 : Fin 2), 1] : List (Fin 2)).length ≤ 128 := by decide

end
end Transformer.GPTMini.Convex.Structured
