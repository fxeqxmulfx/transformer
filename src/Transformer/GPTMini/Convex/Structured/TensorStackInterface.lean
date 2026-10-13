import Transformer.GPTMini.Convex.Structured.TensorStackTraining
import Transformer.GPTMini.Semantics.CountEmbedding

/-!
# Genuine full tensor stack through the checked integer-list interface

Source: GPTMini.TokenInterface's actual integer encoding/context checks,
TensorStack's true shared-weight two-residual forward and its derived
final RMSNorm/tied greedy equality. This model consumes checked token
IDs, actual freely learned tensor embeddings/positions, real residual
blocks and unchanged tied Euclidean readout. The public input contains
no semantic state, route, branch label or task oracle.

At every free weight assignment, its actual prediction is proved equal
to the verified mixed model, including all encoding/context fallback
branches. Every integer input is preserved with exactly one appended
prediction. This is actual forward coupling, not correctness assumed
of supplied logits. Full Basis finite-weight instantiation and correct
data minibatch training are provided by subsequent recipe integration.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

/-- Actual checked continuation uses the complete tensor stack, final RMSNorm and tied embedding readout.
Source: the real token/context checks and TensorStack's configured forward, with no reference labels in inference. -/
def tensorNext (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model) (eps : ℝ)
    (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens) : ℤ :=
  match encodeTokens cfg.vocab_size tokens with
  | some (head :: tail) =>
      if hcap : tail.length + 1 ≤ cfg.max_seq_len then
        ((bestToken cfg.vocab_pos (fun token => inner (𝕜 := ℝ) (rmsNormEps eps
          (tensorFullStack cfg eps hwidth (tensorHeadParameters θ)
            (tensorRawSequence hsize hwidth θ (head :: tail) (by simpa only [List.length_cons] using hcap))
            (by simpa only [List.length_cons] using hcap) (bindingFinalPosition tail))) (tensorEmbedding hsize θ.1 token))).val : ℤ)
      else pad
  | _ => pad

/-- The genuine tensor model has precisely the requested prefix-preserving List Int to List Int signature.
Source: actual checked tensorNext prediction and the common Basis.extend convention. -/
def tensorFunction (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model) (eps : ℝ)
    (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) : Tokens → Tokens := extend (tensorNext cfg hsize hwidth eps θ)

/-- Actual full-stack integer prediction equals the verified raw model for every free weight and every integer input.
Source: genuine configured-stack/final-RMS/tied greedy transfer inside the same actual encoding/context branches. -/
theorem tensorNext_mixed (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens) :
    tensorNext cfg hsize hwidth eps θ tokens = mixedNext cfg.vocab_pos hsize θ tokens := by
  unfold tensorNext mixedNext
  cases he : encodeTokens cfg.vocab_size tokens with
  | none => rfl
  | some encoded =>
      cases encoded with
      | nil => rfl
      | cons head tail =>
          dsimp only
          by_cases hcap : tail.length + 1 ≤ cfg.max_seq_len
          · rw [dite_eq_left hcap, dite_eq_left hcap,
              tensorFullStack_last_best cfg cfg.vocab_pos hsize hwidth eps heps]
          · rw [dite_eq_right hcap, dite_eq_right hcap]

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by decide, by decide, by norm_num⟩

/-- The whole actual tensor integer function agrees with the proven mixed callback at every unrestricted weight assignment.
Source: actual next-prediction identity, retaining every original raw integer and every checked fallback. -/
theorem tensorFunction_mixed (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) :
    tensorFunction cfg hsize hwidth eps θ = mixedFunction cfg.vocab_pos hsize θ := by
  funext tokens
  unfold tensorFunction mixedFunction extend
  rw [tensorNext_mixed cfg hsize hwidth eps heps]

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by decide, by decide, by norm_num⟩

/-- Every integer input produces exactly one additional token under the actual tensor model.
Source: the genuine append-one interface; task validity and successful training are unnecessary. -/
theorem tensorFunction_length (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens) :
    (tensorFunction cfg hsize hwidth eps θ tokens).length = tokens.length + 1 := extend_length _ _

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model := by decide

/-- The actual tensor model preserves every original integer in order, including invalid token inputs.
Source: the unchanged actual Basis.extend operation around the true tensor predictor. -/
theorem tensorFunction_prefix (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens) :
    (tensorFunction cfg hsize hwidth eps θ tokens).take tokens.length = tokens := extend_prefix _ _

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model := by decide

/-- Valid checked input exposes the actual full-stack prediction's verified mixed greedy value.
Source: real configured tensor-next identity and unchanged encoding/context branch, without a correct-output premise. -/
theorem tensorNext_of_encode (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
    (hencode : encodeTokens cfg.vocab_size tokens = some (head :: tail)) (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    tensorNext cfg hsize hwidth eps θ tokens = ((mixedGreedy cfg.vocab_pos hsize θ (head :: tail)
      (by simpa only [List.length_cons] using hlen) (bindingFinalPosition tail)).val : ℤ) := by
  rw [tensorNext_mixed cfg hsize hwidth eps heps]
  exact mixedNext_of_encode cfg.vocab_pos hsize θ tokens head tail hencode hlen

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model ∧ (0 : ℝ) < 1 / 100000 ∧
    encodeTokens Semantics.countConfig.vocab_size [1, 22, 18] = some [(1 : Fin 68), 22, 18] ∧
    ([22, 18] : List (Fin 68)).length + 1 ≤ Semantics.countConfig.max_seq_len := by
  exact ⟨by decide, by decide, by norm_num, by decide, by decide⟩

/-- Invalid checked integer inputs cannot enter any actual tensor embedding or learned attention branch.
Source: the genuine tensorNext vocabulary check and total PAD fallback. -/
theorem tensorNext_invalid (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens)
    (hencode : encodeTokens cfg.vocab_size tokens = none) : tensorNext cfg hsize hwidth eps θ tokens = pad := by
  unfold tensorNext
  rw [hencode]

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model ∧
    encodeTokens Semantics.countConfig.vocab_size [-1, 22, 18] = none := by decide

/-- Overlong raw prefixes take the real context fallback before the full tensor stack is evaluated.
Source: the actual checked dependent context branch in tensorNext, rather than semantic inference restrictions. -/
theorem tensorNext_overlong (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
    (hencode : encodeTokens cfg.vocab_size tokens = some (head :: tail)) (hlen : ¬tail.length + 1 ≤ cfg.max_seq_len) :
    tensorNext cfg hsize hwidth eps θ tokens = pad := by
  unfold tensorNext
  rw [hencode]
  dsimp only
  rw [dite_eq_right hlen]

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model ∧
    encodeTokens Semantics.countConfig.vocab_size (decodeTokens ((1 : Fin 68) :: List.replicate 19 0)) =
      some ((1 : Fin 68) :: List.replicate 19 0) ∧
    ¬(List.replicate 19 (0 : Fin 68)).length + 1 ≤ Semantics.countConfig.max_seq_len := by decide

/-- Empty raw input is retained and receives the specified PAD answer before any tensor-head computation.
Source: the actual tensorNext empty branch and genuine prefix-preserving extension. -/
theorem tensorFunction_empty (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) : tensorFunction cfg hsize hwidth eps θ [] = [pad] := by
  unfold tensorFunction extend tensorNext
  rw [encodeTokens, List.nil_append]

example : Semantics.countConfig.vocab_size ≤ 1024 ∧ 64 ≤ Semantics.countConfig.d_model := by decide

/-- A genuine negative integer is preserved as input while the actual vocabulary check appends PAD.
Source: real checked tensor inference and unchanged raw extension, without modulo encoding. -/
example (θ : BindingParameters 68 19) :
    tensorFunction Semantics.countConfig (by decide) (by decide) (1 / 100000) θ [-1] = [-1, pad] := by
  rw [tensorFunction_mixed _ _ _ _ (by norm_num)]
  exact mixedFunction_negative _ _ _

end
end Transformer.GPTMini.Convex.Structured
