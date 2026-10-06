import Transformer.Basis.Basic
import Transformer.GPTMini.TokenInterface.Decoding

/-!
# The real softmax GPTMini as List Int to List Int

Source: archived gpt_mini.py at f11b6e2, its existing Lean forward, and
infrastructure/nn/transformer.py at cbafbe9. Inputs are checked against the
finite vocabulary and context bound, and use positions 0,...,T-1. The
actual final-position logits are greedily decoded and one token appended.

This is a mathematical real-arithmetic function: it does not certify
PyTorch rounding or how AdamW finds weights. Invalid, empty and overlong
inputs get PAD. The existing Lean forward has no context cap, so this
adapter adds the Python implementation's explicit bound. No task oracle
is called from the model interface.

As in the existing Lean stack, one supplied epsilon is shared by RMSNorm,
QKNorm and XSA. Python's defaults instead use 1e-5 for RMSNorm and 1e-6
for the head normalizations. This shared-epsilon family is explicit here;
this adapter does not assert equality with those unequal-default numerics.
-/

namespace Transformer.GPTMini.TokenInterface

open scoped Classical
noncomputable section

/-- Last-position logits from the complete actual model on a nonempty token list.
Source: GPTMini.forward, with the reference's integer-indexed RoPE positions. -/
def lastLogits (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) : Fin cfg.vocab_size → ℝ :=
  forward cfg params eps (fun i => (i.val : ℝ)) (head :: tail).get
    ⟨tail.length, by simp⟩

/-- Last-position output softmax, still obtained from the actual model stack.
Source: GPTMini.softmaxOutput; the token and position arrays coincide with lastLogits. -/
def lastProbabilities (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) : Fin cfg.vocab_size → ℝ :=
  softmaxOutput cfg params eps (fun i => (i.val : ℝ)) (head :: tail).get
    ⟨tail.length, by simp⟩

/-- Greedy continuation after checked raw-token encoding and a context-size check.
Source: the new integer adapter to the original causal softmax GPTMini. -/
def nextToken (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens : Transformer.Basis.Tokens) : ℤ :=
  match Transformer.Basis.encodeTokens cfg.vocab_size tokens with
  | some (head :: tail) =>
      if tail.length + 1 ≤ cfg.max_seq_len then
        (bestToken cfg.vocab_pos (lastProbabilities cfg params eps head tail)).val
      else Transformer.Basis.pad
  | _ => Transformer.Basis.pad

/-- The requested List Int to List Int model function, preserving the whole input.
Source: the new common Basis interface; its answer comes only from GPTMini logits. -/
def tokenFunction (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    Transformer.Basis.Tokens → Transformer.Basis.Tokens :=
  Transformer.Basis.extend (nextToken cfg params eps)

/-- Model continuations satisfy exactly the same length contract as the task functions.
Source: the new adapter; this statement makes no task-correctness assumption. -/
theorem tokenFunction_length (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens : Transformer.Basis.Tokens) :
    (tokenFunction cfg params eps tokens).length = tokens.length + 1 :=
  Transformer.Basis.extend_length _ _

/-- The model preserves every original integer token, rather than returning just its prediction.
Source: the answer-append convention used for every task and the model. -/
theorem tokenFunction_prefix (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens : Transformer.Basis.Tokens) :
    (tokenFunction cfg params eps tokens).take tokens.length = tokens :=
  Transformer.Basis.extend_prefix _ _

/-- The adapter's final-probability and logit decoders are exactly interchangeable.
Source: the actual softmaxOutput ordering theorem, including ties. -/
theorem bestToken_lastProbabilities (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) :
    bestToken cfg.vocab_pos (lastProbabilities cfg params eps head tail) =
      bestToken cfg.vocab_pos (lastLogits cfg params eps head tail) :=
  (bestToken_softmaxOutput cfg params eps _ _ _).symm

/-- On a legal finite-token context, the integer answer is the actual greedy final logit.
Source: the checked round-trip adapter; the context bound matches Python's forward check. -/
theorem nextToken_decode_cons (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    nextToken cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      (bestToken cfg.vocab_pos (lastLogits cfg params eps head tail)).val := by
  simp only [nextToken, Transformer.Basis.encode_decode, ite_eq_left hlen,
    bestToken_lastProbabilities]

example : ([⟨9, by decide⟩] : List (Fin Config.default.vocab_size)).length + 1 ≤
    Config.default.max_seq_len := by decide

/-- The complete raw-list result is linked to the actual final-position forward call.
Source: GPTMini.forward, not an abstract rounded-transformer recognizer or a paired-key oracle. -/
theorem tokenFunction_decode_cons (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    tokenFunction cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.decodeTokens (head :: tail) ++
        [((bestToken cfg.vocab_pos (lastLogits cfg params eps head tail)).val : ℤ)] := by
  unfold tokenFunction Transformer.Basis.extend
  rw [nextToken_decode_cons cfg params eps head tail hlen]

example : ([⟨22, by decide⟩, ⟨18, by decide⟩] : List (Fin Config.default.vocab_size)).length + 1 ≤
    Config.default.max_seq_len := by decide

/-- Successful raw encoding aligns the complete input and the model's final row exactly.
Source: the new checked adapter; both vocabulary and context checks remain explicit premises. -/
theorem tokenFunction_of_encode (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens : Transformer.Basis.Tokens)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
    (hencode : Transformer.Basis.encodeTokens cfg.vocab_size tokens = some (head :: tail))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    tokenFunction cfg params eps tokens = tokens ++
      [((bestToken cfg.vocab_pos (lastLogits cfg params eps head tail)).val : ℤ)] := by
  rw [← Transformer.Basis.decode_encode hencode]
  exact tokenFunction_decode_cons cfg params eps head tail hlen

example : Transformer.Basis.encodeTokens Config.default.vocab_size [1, 9] =
      some [⟨1, by decide⟩, ⟨9, by decide⟩] ∧
    ([⟨9, by decide⟩] : List (Fin Config.default.vocab_size)).length + 1 ≤
      Config.default.max_seq_len := by decide

/-- The empty context has no final supervised position and receives PAD.
Source: the total integer adapter's explicitly specified fallback. -/
theorem tokenFunction_empty (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    tokenFunction cfg params eps [] = [Transformer.Basis.pad] := by
  simp [tokenFunction, Transformer.Basis.extend, nextToken, Transformer.Basis.encodeTokens]

/-- A negative ID cannot reach the real model's embedding lookup.
Source: the checked adapter, which avoids Int.toNat silently converting -1 to zero. -/
theorem tokenFunction_negative (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    tokenFunction cfg params eps [-1] = [-1, Transformer.Basis.pad] := by
  simp [tokenFunction, Transformer.Basis.extend, nextToken, Transformer.Basis.encodeTokens]

/-- Overlong finite-token inputs are rejected at the adapter even though Lean forward is total.
Source: infrastructure/nn/transformer.py's explicit context limit, restored in this interface. -/
theorem nextToken_overlong (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
    (hlen : ¬tail.length + 1 ≤ cfg.max_seq_len) :
    nextToken cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.pad := by
  simp only [nextToken, Transformer.Basis.encode_decode, ite_eq_right hlen]

example : ¬(List.replicate 2050 (⟨1, by decide⟩ : Fin Config.default.vocab_size)).length + 1 ≤
    Config.default.max_seq_len := by
  simp only [List.length_replicate]
  change ¬2050 + 1 ≤ 2048
  omega

/-- Token-function equality means equality of predictions, with no change to the shared prefix.
Source: the new model/task comparison interface, valid for any raw-token answer function. -/
theorem tokenFunction_eq_extend_iff (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (answer : Transformer.Basis.Tokens → ℤ) (tokens : Transformer.Basis.Tokens) :
    tokenFunction cfg params eps tokens = Transformer.Basis.extend answer tokens ↔
      nextToken cfg params eps tokens = answer tokens :=
  Transformer.Basis.extend_eq_iff _ _ _

end
end Transformer.GPTMini.TokenInterface
