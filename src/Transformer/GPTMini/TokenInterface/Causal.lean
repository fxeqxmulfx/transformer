import Transformer.GPTMini.TokenInterface.Basic
import Transformer.GPTMini.TokenInterface.Correctness
import Transformer.GPTMini.Causality.Model

/-!
# Teacher-forced rows and List Int continuations have the same prediction

Source: Transformer.forward and synthetic/Answered.metrics at cbafbe9,
and the actual GPTMini stack at f11b6e2. Evaluation on a complete input
row and evaluation on its supervised prefix give exactly the same logits
and greedy answer. This proves the causal property needed by the common
List Int interface for depth, recall and parity, for arbitrary parameters.
It does not assume that the answer matches a task oracle.
-/

namespace Transformer.GPTMini.TokenInterface

open scoped Classical BigOperators
noncomputable section

/-- All-position logits on a finite-token list, with the reference's natural-number positions.
Source: the complete GPTMini.forward rather than a separate task interpreter. -/
def listLogits (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens : List (Fin cfg.vocab_size)) (i : Fin tokens.length) (v : Fin cfg.vocab_size) : ℝ :=
  forward cfg params eps (fun j => (j.val : ℝ)) tokens.get i v

/-- Arbitrary future tokens leave every original logit unchanged.
Source: the actual full-stack forward_prefix theorem, transported through List.get. -/
theorem listLogits_append (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens suffix : List (Fin cfg.vocab_size)) (i : Fin tokens.length) (v : Fin cfg.vocab_size) :
    listLogits cfg params eps (tokens ++ suffix)
      ⟨i.val, by simp only [List.length_append]; omega⟩ v =
        listLogits cfg params eps tokens i v := by
  have hlen : tokens.length ≤ (tokens ++ suffix).length := by
    simp only [List.length_append]
    omega
  apply Causality.forward_castLE cfg params eps hlen
    (fun j => (j.val : ℝ)) (fun j => (j.val : ℝ)) tokens.get (tokens ++ suffix).get
  · intro j
    rfl
  · intro j
    simpa only [List.get_eq_getElem, Fin.castLE] using List.getElem_append_left j.isLt

/-- Taking a prefix also preserves all its logits; the cutoff may be any natural number.
Source: the model's causal law and exact prefix indexing, including take past the input length. -/
theorem listLogits_take (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens : List (Fin cfg.vocab_size)) (n : ℕ) (i : Fin (tokens.take n).length)
    (v : Fin cfg.vocab_size) :
    listLogits cfg params eps tokens (i.castLE (List.length_take_le' n tokens)) v =
      listLogits cfg params eps (tokens.take n) i v := by
  apply Causality.forward_castLE cfg params eps (List.length_take_le' n tokens)
    (fun j => (j.val : ℝ)) (fun j => (j.val : ℝ)) (tokens.take n).get tokens.get
  · intro j
    rfl
  · intro j
    simp only [List.get_eq_getElem, List.getElem_take, Fin.castLE]

/-- A teacher-forced row at the final prefix position equals the prefix's own last logits.
Source: the last supervised row used by Basis metrics and the actual lastLogits adapter. -/
theorem lastLogits_fullRow (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail suffix : List (Fin cfg.vocab_size))
    (v : Fin cfg.vocab_size) :
    listLogits cfg params eps ((head :: tail) ++ suffix)
      ⟨tail.length, by simp only [List.length_append, List.length_cons]; omega⟩ v =
        lastLogits cfg params eps head tail v := by
  exact listLogits_append cfg params eps (head :: tail) suffix ⟨tail.length, by simp⟩ v

/-- Greedy readout commutes with evaluating a prefix inside a longer teacher-forced row.
Source: full-stack causality and the reference's deterministic smallest-ID tie rule. -/
theorem bestToken_fullRow (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail suffix : List (Fin cfg.vocab_size)) :
    bestToken cfg.vocab_pos (listLogits cfg params eps ((head :: tail) ++ suffix)
      ⟨tail.length, by simp only [List.length_append, List.length_cons]; omega⟩) =
        bestToken cfg.vocab_pos (lastLogits cfg params eps head tail) := by
  apply congrArg (bestToken cfg.vocab_pos)
  funext v
  exact lastLogits_fullRow cfg params eps head tail suffix v

/-- The raw List Int answer is exactly the same greedy token as the full-row evaluator predicts.
Source: checked encoding plus architectural causality; no task-correctness premise is present. -/
theorem nextToken_fullRow (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail suffix : List (Fin cfg.vocab_size))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    nextToken cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      (bestToken cfg.vocab_pos (listLogits cfg params eps ((head :: tail) ++ suffix)
        ⟨tail.length, by simp only [List.length_append, List.length_cons]; omega⟩)).val := by
  rw [bestToken_fullRow]
  exact nextToken_decode_cons cfg params eps head tail hlen

example : ([⟨9, by decide⟩, ⟨10, by decide⟩] : List (Fin Config.default.vocab_size)).length + 1 ≤
    Config.default.max_seq_len := by decide

/-- The entire integer continuation can be checked at its supervised row in a full sequence.
Source: the model's own List Int function, not just a comparison of abstract recognizers. -/
theorem tokenFunction_fullRow (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail suffix : List (Fin cfg.vocab_size))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    tokenFunction cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.decodeTokens (head :: tail) ++
        [((bestToken cfg.vocab_pos (listLogits cfg params eps ((head :: tail) ++ suffix)
          ⟨tail.length, by simp only [List.length_append, List.length_cons]; omega⟩)).val : ℤ)] := by
  unfold tokenFunction Transformer.Basis.extend
  rw [nextToken_fullRow cfg params eps head tail suffix hlen]

example : ([⟨22, by decide⟩, ⟨18, by decide⟩] : List (Fin Config.default.vocab_size)).length + 1 ≤
    Config.default.max_seq_len := by decide

/-- A genuinely legal complete teacher-forced row also gives the List Int prefix continuation.
Source: Python's context check and the previous exact model/prefix correspondence. -/
theorem tokenFunction_legal_fullRow (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail suffix : List (Fin cfg.vocab_size))
    (hlen : ((head :: tail) ++ suffix).length ≤ cfg.max_seq_len) :
    tokenFunction cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.decodeTokens (head :: tail) ++
        [((bestToken cfg.vocab_pos (listLogits cfg params eps ((head :: tail) ++ suffix)
          ⟨tail.length, by simp only [List.length_append, List.length_cons]; omega⟩)).val : ℤ)] := by
  have hp : tail.length + 1 ≤ cfg.max_seq_len := by
    simp only [List.length_append, List.length_cons] at hlen
    omega
  exact tokenFunction_fullRow cfg params eps head tail suffix hp

example : (([⟨1, by decide⟩, ⟨9, by decide⟩] : List (Fin Config.default.vocab_size)) ++
    ([⟨10, by decide⟩, ⟨11, by decide⟩] : List (Fin Config.default.vocab_size))).length ≤
      Config.default.max_seq_len := by decide

/-- Exact correctness can be verified at the full teacher-forced row, including equal-logit ties.
Source: the common List Int contract, full-stack causality and the actual greedy readout criterion. -/
theorem tokenFunction_fullRow_correct_iff (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail suffix : List (Fin cfg.vocab_size))
    (answer : Fin cfg.vocab_size) (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    let row := listLogits cfg params eps ((head :: tail) ++ suffix)
      ⟨tail.length, by simp only [List.length_append, List.length_cons]; omega⟩
    tokenFunction cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.decodeTokens (head :: tail) ++ [(answer.val : ℤ)] ↔
        (∀ v, row v ≤ row answer) ∧ (∀ v, row answer ≤ row v → answer.val ≤ v.val) := by
  dsimp only
  simp only [lastLogits_fullRow]
  exact tokenFunction_correct_iff cfg params eps _ head tail answer
    (Transformer.Basis.encode_decode _) hlen

example : ([⟨9, by decide⟩] : List (Fin Config.default.vocab_size)).length + 1 ≤
    Config.default.max_seq_len := by decide

/-- Generating a token cannot retroactively change the logits that generated it.
Source: causal autoregressive evaluation, with the actual greedy token as the added suffix. -/
theorem generatedToken_preserves_logits (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) (v : Fin cfg.vocab_size) :
    listLogits cfg params eps
      ((head :: tail) ++ [bestToken cfg.vocab_pos (lastLogits cfg params eps head tail)])
      ⟨tail.length, by simp only [List.length_append, List.length_cons]; omega⟩ v =
        lastLogits cfg params eps head tail v := by
  exact lastLogits_fullRow cfg params eps head tail _ v

end
end Transformer.GPTMini.TokenInterface
