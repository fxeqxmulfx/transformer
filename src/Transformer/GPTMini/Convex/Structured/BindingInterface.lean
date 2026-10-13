import Transformer.GPTMini.Convex.Structured.RawBinding
import Transformer.GPTMini.Convex.Structured.SharedInterface

/-!
# Genuine learned all-pair pointer through the integer-list interface

Source: actual raw binding inference/loss at e0be315 and the checked
state-head callback at b91036f, ported here to the true learned pointer.
The actual final raw position supplies its query, so every candidate
pair is causally visible in that prefix. No paired encoder, parsed
table, desired route, reference rule or correct logits enter prediction.

Vocabulary/context checks retain the exact append-one contract for all
integer lists, including invalid, empty and overlong ones. True values
come from the same compact normalized model as the convex complete loss.
This is a standalone head callback; the original-shape tensor block,
prenorm/residual/tied readout and complete mixed-head realization remain
separate obligations. Task capability must derive true raw margins from
the actual finite weights rather than assume a correct decoder output.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

variable {V C : ℕ}

/-- The actual final visible raw index of a nonempty checked prefix.
Source: ordinary unchanged list positions; this is a physical index, not a table/query semantic oracle. -/
def bindingFinalPosition (tail : List (Fin V)) : Fin (tail.length + 1) :=
  ⟨tail.length, by omega⟩

/-- The genuinely derived last-query index is unique in a nonempty raw prefix.
Source: physical last-index arithmetic; this connects raw parser final-query recovery to actual checked inference. -/
theorem bindingFinalPosition_unique (tail : List (Fin V)) (query : Fin (tail.length + 1))
    (hquery : query.val + 1 = tail.length + 1) : query = bindingFinalPosition tail := by
  apply Fin.ext
  change query.val = tail.length
  omega

example : (0 : Fin 1).val + 1 = ([] : List (Fin 548)).length + 1 := by decide

/-- The actual chosen last row remains visibly inside its own raw prefix, including one-token inputs.
Source: the finite physical query index, without any semantic table selection. -/
theorem bindingFinalPosition_visible (tail : List (Fin V)) : (bindingFinalPosition tail).val < tail.length + 1 :=
  (bindingFinalPosition tail).isLt

/-- Next-token prediction from checked raw input and actual free shared binding weights alone.
Source: real all-pair compact learned scores and original bestToken, with PAD only for failed vocabulary/context checks. -/
def bindingNext (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) (tokens : Tokens) : ℤ :=
  match encodeTokens V tokens with
  | some (head :: tail) =>
      if hcap : tail.length + 1 ≤ C then
        ((bestToken hV (fun token => rawBindingScore θ (head :: tail)
          (by simpa only [List.length_cons] using hcap) (bindingFinalPosition tail) (vocabularyCode hsize token))).val : ℤ)
      else pad
  | _ => pad

/-- The true learned binding callback has the same append-one integer-list public signature as GPTMini.
Source: the checked actual head prediction and Basis.extend, without semantic inference code. -/
def bindingFunction (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) : Tokens → Tokens :=
  extend (bindingNext hV hsize θ)

/-- Every actual integer-list call has output length exactly input length plus one.
Source: the genuine checked learned-head callback, regardless of input validity or weight quality. -/
theorem bindingFunction_length (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) (tokens : Tokens) :
    (bindingFunction hV hsize θ tokens).length = tokens.length + 1 := extend_length _ _

example : (0 : ℕ) < 548 ∧ (548 : ℕ) ≤ 1024 := by omega

/-- Every actual integer-list call preserves all raw tokens and their order exactly.
Source: the true free-parameter learned-head append-one interface, without a task-validity premise. -/
theorem bindingFunction_prefix (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) (tokens : Tokens) :
    (bindingFunction hV hsize θ tokens).take tokens.length = tokens := extend_prefix _ _

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 := by omega

/-- Actual successful encoding and context checks expose only the genuine final raw-prefix greedy computation.
Source: the checked dependent context branch and actual all-pair head scores, with no correct-answer assumption. -/
theorem bindingNext_of_encode (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ C) :
    bindingNext hV hsize θ tokens =
      ((bestToken hV (fun token => rawBindingScore θ (head :: tail)
        (by simpa only [List.length_cons] using hlen) (bindingFinalPosition tail) (vocabularyCode hsize token))).val : ℤ) := by
  unfold bindingNext
  rw [hencode]
  dsimp only
  rw [dite_eq_left hlen]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [0, 1] = some [(0 : Fin 2), 1] ∧
    ([1] : List (Fin 2)).length + 1 ≤ 4 := by decide

/-- The complete actual public continuation is coupled to the real checked final-row pointer decoder.
Source: exact encoding/context branch evaluation and the original prefix-preserving continuation convention. -/
theorem bindingFunction_of_encode (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ C) :
    bindingFunction hV hsize θ tokens = tokens ++
      [((bestToken hV (fun token => rawBindingScore θ (head :: tail)
        (by simpa only [List.length_cons] using hlen) (bindingFinalPosition tail) (vocabularyCode hsize token))).val : ℤ)] := by
  unfold bindingFunction extend
  rw [bindingNext_of_encode hV hsize θ tokens head tail hencode hlen]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [1, 0] = some [(1 : Fin 2), 0] ∧
    ([0] : List (Fin 2)).length + 1 ≤ 4 := by decide

/-- Empty raw input has no query and receives exactly the specified PAD continuation.
Source: the same total checked learned binding callback, before any head computation. -/
theorem bindingFunction_empty (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) :
    bindingFunction hV hsize θ [] = [pad] := by
  unfold bindingFunction extend bindingNext
  rw [encodeTokens, List.nil_append]

example : (0 : ℕ) < 548 ∧ (548 : ℕ) ≤ 1024 := by omega

/-- A genuinely encoded overlong raw input is rejected before any true all-pair pointer computation.
Source: actual checked context capacity; the argument is a raw length check, not a semantic route constraint. -/
theorem bindingNext_overlong (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : ¬tail.length + 1 ≤ C) :
    bindingNext hV hsize θ tokens = pad := by
  unfold bindingNext
  rw [hencode]
  dsimp only
  rw [dite_eq_right hlen]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [0, 1] = some [(0 : Fin 2), 1] ∧
    ¬([1] : List (Fin 2)).length + 1 ≤ 1 := by decide

/-- A failed real vocabulary check never aliases an invalid integer to a model token or enters learned inference.
Source: genuine Basis.encodeTokens and the checked binding callback's total PAD fallback. -/
theorem bindingNext_invalid (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C)
    (tokens : Tokens) (hencode : encodeTokens V tokens = none) : bindingNext hV hsize θ tokens = pad := by
  unfold bindingNext
  rw [hencode]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 ∧ encodeTokens 2 [-1, 0] = none := by decide

/-- A negative raw token remains in the public prefix but receives PAD through the actual checked decoder interface.
Source: the genuine lower vocabulary bound and complete prefix-preserving learned callback. -/
theorem bindingFunction_negative (hV : 0 < V) (hsize : V ≤ 1024) (θ : BindingParameters V C) :
    bindingFunction hV hsize θ [-1] = [-1, pad] := by
  simp only [bindingFunction, extend, bindingNext, encodeTokens]
  norm_num

example : (0 : ℕ) < 2 ∧ (2 : ℕ) ≤ 1024 := by omega

end
end Transformer.GPTMini.Convex.Structured
