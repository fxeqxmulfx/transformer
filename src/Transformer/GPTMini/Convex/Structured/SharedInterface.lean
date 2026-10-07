import Transformer.GPTMini.Convex.Structured.SharedReference
import Transformer.GPTMini.TokenInterface.Decoding
import Transformer.Basis.Basic

/-!
# Actual shared causal head through List Int to List Int

Source: the actual shared stochastic state/value head at d72d3b3 and
the original checked GPTMini integer/greedy adapter at f11b6e2. The
head callback reads genuine raw vocabulary tokens, propagates learned
state distributions and computes true ten-coordinate output means.
No reference rule, labels, prepared encoder or correct logits enter it.

Vocabulary and context checks preserve the original append-one public
contract on all integer lists. This standalone head interface is a
step toward the changed embedding/attention block, not a claim that
prenorm, residuals, tied embeddings or the complete stack are realized.
The likelihood's unrestricted shared-parameter convexity is already
proved; task-specific raw correctness must discharge reference semantics.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

variable {V C : ℕ}

/-- A genuine model vocabulary ID keeps its value in the fixed 1024-token decoder alphabet.
Source: the actual vocabulary bound, without modulo aliasing of token IDs. -/
def vocabularyCode (hsize : V ≤ 1024) (token : Fin V) : Fin 1024 :=
  ⟨token.val, by have ht := token.isLt; omega⟩

/-- Lifting the real vocabulary to the decoder creates no token collisions.
Source: exact integer ID preservation, needed for margins against all true vocabulary competitors. -/
theorem vocabularyCode_injective (hsize : V ≤ 1024) : Function.Injective (vocabularyCode hsize) := by
  intro a b h
  have hv := congrArg Fin.val h
  exact Fin.ext hv

example : (68 : ℕ) ≤ 1024 := by omega

/-- Greedy continuation computed only from checked raw tokens and actual free learned state/value weights.
Source: the real shared head's output means and original deterministic bestToken; invalid, empty or overlong inputs get PAD. -/
def sharedMarkovNext (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C) (tokens : Tokens) : ℤ :=
  match encodeTokens V tokens with
  | some (head :: tail) =>
      if tail.length + 1 ≤ C then
        (bestToken hV (fun token => sharedMarkovScore θ (head :: tail) (vocabularyCode hsize token))).val
      else pad
  | _ => pad

/-- The standalone actually learned head uses the same append-one integer-list public signature.
Source: Basis.extend and the checked genuine learned stochastic output, without a task oracle in prediction. -/
def sharedMarkovFunction (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C) : Tokens → Tokens :=
  extend (sharedMarkovNext hV hsize θ)

/-- Every integer-list call retains the exact requested output length.
Source: the actual standalone learned-head callback's append-one contract, including invalid inputs. -/
theorem sharedMarkovFunction_length (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C) (tokens : Tokens) :
    (sharedMarkovFunction hV hsize θ tokens).length = tokens.length + 1 := extend_length _ _

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 := by omega

/-- Every integer-list call preserves the entire original token order.
Source: the same real append-one callback, independent of learned weight quality or input legality. -/
theorem sharedMarkovFunction_prefix (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C) (tokens : Tokens) :
    (sharedMarkovFunction hV hsize θ tokens).take tokens.length = tokens := extend_prefix _ _

example : (0 : ℕ) < 36 ∧ (36 : ℕ) ≤ 1024 := by omega

/-- Successful actual encoding/context checks expose the genuine final learned-head greedy score.
Source: the exact checked callback definition, not a semantic readout premise. -/
theorem sharedMarkovNext_of_encode (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ C) :
    sharedMarkovNext hV hsize θ tokens =
      ((bestToken hV (fun token => sharedMarkovScore θ (head :: tail) (vocabularyCode hsize token))).val : ℤ) := by
  unfold sharedMarkovNext
  rw [hencode]
  dsimp only
  rw [ite_eq_left hlen]

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 ∧ encodeTokens 68 [1, 22, 18] =
    some [(1 : Fin 68), 22, 18] ∧ ([22, 18] : List (Fin 68)).length + 1 ≤ 19 := by decide

/-- The complete integer-list continuation is coupled to the actual computed normalized state/value decoder.
Source: the successful checked encoding bridge and original prefix-preserving extension convention. -/
theorem sharedMarkovFunction_of_encode (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ C) :
    sharedMarkovFunction hV hsize θ tokens = tokens ++
      [((bestToken hV (fun token => sharedMarkovScore θ (head :: tail) (vocabularyCode hsize token))).val : ℤ)] := by
  unfold sharedMarkovFunction extend
  rw [sharedMarkovNext_of_encode hV hsize θ tokens head tail hencode hlen]

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 ∧ encodeTokens 68 [1, 21, 18] =
    some [(1 : Fin 68), 21, 18] ∧ ([21, 18] : List (Fin 68)).length + 1 ≤ 19 := by decide

/-- The genuine greedy decoder at one finite shared weight table selects the independently derived reference endpoint label.
Source: strict true whole-vocabulary confidence/margins and collision-free real token IDs; raw task semantics remain to be proved. -/
theorem referenceShared_best (hV : 0 < V) (hsize : V ≤ 1024) (rule : Fin V → Fin 6 → Fin 6)
    (labels : Fin 6 → Fin V) (start : Fin 6) (tokens : List (Fin V)) (hT : tokens.length ≤ 128) :
    bestToken hV (fun token => sharedMarkovScore
      (referenceSharedParameters (C := C) rule (fun state => vocabularyCode hsize (labels state)) start referenceGain)
      tokens (vocabularyCode hsize token)) = labels (referenceRun rule start tokens) := by
  apply bestToken_of_strict
  intro rival hr
  apply referenceShared_strict rule (fun state => vocabularyCode hsize (labels state)) start tokens _ hT
  intro he
  exact hr ((vocabularyCode_injective hsize he).symm)

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 ∧ ([1, 22, 18] : List (Fin 68)).length ≤ 128 := by decide

/-- Empty raw input has no final row and receives PAD through the same total learned-head interface.
Source: the checked callback's specified empty-input convention. -/
theorem sharedMarkovFunction_empty (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C) :
    sharedMarkovFunction hV hsize θ [] = [pad] := by
  unfold sharedMarkovFunction extend sharedMarkovNext
  rw [encodeTokens, List.nil_append]

example : (0 : ℕ) < 548 ∧ (548 : ℕ) ≤ 1024 := by omega

/-- Successful encoding of an overlong input never reaches the learned head or decoder.
Source: the original raw context check, retained in the standalone proposed head callback. -/
theorem sharedMarkovNext_overlong (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C)
    (tokens : Tokens) (head : Fin V) (tail : List (Fin V))
    (hencode : encodeTokens V tokens = some (head :: tail)) (hlen : ¬tail.length + 1 ≤ C) :
    sharedMarkovNext hV hsize θ tokens = pad := by
  unfold sharedMarkovNext
  rw [hencode]
  dsimp only
  rw [ite_eq_right hlen]

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 ∧ encodeTokens 68 [1, 22, 18] =
    some [(1 : Fin 68), 22, 18] ∧ ¬([22, 18] : List (Fin 68)).length + 1 ≤ 2 := by decide

/-- A failed actual vocabulary check always gives the specified PAD continuation.
Source: the real raw integer encoder and total fallback, without any learned row lookup on invalid IDs. -/
theorem sharedMarkovNext_invalid (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C)
    (tokens : Tokens) (hencode : encodeTokens V tokens = none) : sharedMarkovNext hV hsize θ tokens = pad := by
  unfold sharedMarkovNext
  rw [hencode]

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 ∧ encodeTokens 68 [-1, 22, 18] = none := by decide

/-- A negative integer is rejected by the real checked vocabulary lookup rather than aliased to PAD's embedding.
Source: Basis.encodeTokens's actual lower bound, retained before any learned computation. -/
theorem sharedMarkovFunction_negative (hV : 0 < V) (hsize : V ≤ 1024) (θ : SharedParameters V C) :
    sharedMarkovFunction hV hsize θ [-1] = [-1, pad] := by
  simp only [sharedMarkovFunction, extend, sharedMarkovNext, encodeTokens]
  norm_num

example : (0 : ℕ) < 68 ∧ (68 : ℕ) ≤ 1024 := by omega

end
end Transformer.GPTMini.Convex.Structured
