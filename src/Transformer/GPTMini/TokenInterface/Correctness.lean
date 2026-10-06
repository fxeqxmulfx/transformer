import Transformer.GPTMini.TokenInterface.Control
import Transformer.Basis.Tasks

/-!
# Certificates connecting raw Basis semantics to the actual model

Source: the new Basis List Int adapters, GPTMini.forward from archived
gpt_mini.py at f11b6e2, and Basis recipes at cbafbe9. A checked context and
a logit certificate prove equality of two actual List Int functions. The
exact certificate includes all vocabulary competitors and the tie rule.
Strict margins are a simpler sufficient condition, not a premise claiming
that the model already solves all tasks.

The explicit small-model control supplies satisfiable examples of every
certificate hypothesis. It solves one prefix and fails another, showing
that architectural laws alone do not establish the task logit property.
Specification states that property for given parameters on all valid inputs.
Optimizer convergence and floating-point execution are not certified.
-/

namespace Transformer.GPTMini.TokenInterface

open scoped Classical
noncomputable section

/-- A strict final-logit margin verifies the full continuation on a finite-token context.
Source: the new actual-forward decoding bridge, at finite softmax temperature. -/
theorem tokenFunction_of_margin (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) (answer : Fin cfg.vocab_size)
    (hlen : tail.length + 1 ≤ cfg.max_seq_len)
    (hmargin : ∀ v, v ≠ answer → lastLogits cfg params eps head tail v <
      lastLogits cfg params eps head tail answer) :
    tokenFunction cfg params eps (Transformer.Basis.decodeTokens (head :: tail)) =
      Transformer.Basis.decodeTokens (head :: tail) ++ [(answer.val : ℤ)] := by
  rw [tokenFunction_decode_cons cfg params eps head tail hlen,
    bestToken_of_strict cfg.vocab_pos _ answer hmargin]

example : ([⟨9, by decide⟩] : List (Fin controlConfig.vocab_size)).length + 1 ≤
      controlConfig.max_seq_len ∧
    ∀ v : Fin controlConfig.vocab_size, v ≠ ⟨15, by decide⟩ →
      lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩ [⟨9, by decide⟩] v <
        lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩ [⟨9, by decide⟩]
          ⟨15, by decide⟩ := ⟨by decide, control_depth_margin⟩

/-- Exact raw-list correctness is equivalent to a maximum with the reference's tie rule.
Source: the actual final-row logits, checked encoding and deterministic greedy readout.
Unlike a strict-margin condition, this equivalence also covers equal-logit answers. -/
theorem tokenFunction_correct_iff (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (tokens : Transformer.Basis.Tokens)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) (answer : Fin cfg.vocab_size)
    (hencode : Transformer.Basis.encodeTokens cfg.vocab_size tokens = some (head :: tail))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len) :
    tokenFunction cfg params eps tokens = tokens ++ [(answer.val : ℤ)] ↔
      (∀ v, lastLogits cfg params eps head tail v ≤ lastLogits cfg params eps head tail answer) ∧
        ∀ v, lastLogits cfg params eps head tail answer ≤ lastLogits cfg params eps head tail v →
          answer.val ≤ v.val := by
  have he : tokenFunction cfg params eps tokens = tokens ++ [(answer.val : ℤ)] ↔
      bestToken cfg.vocab_pos (lastLogits cfg params eps head tail) = answer := by
    rw [tokenFunction_of_encode cfg params eps tokens head tail hencode hlen]
    simp [Fin.ext_iff]
  rw [he, bestToken_eq_iff]

example : Transformer.Basis.encodeTokens controlConfig.vocab_size [1, 9] =
      some [⟨1, by decide⟩, ⟨9, by decide⟩] ∧
    ([⟨9, by decide⟩] : List (Fin controlConfig.vocab_size)).length + 1 ≤
      controlConfig.max_seq_len := by decide

/-- At a known raw task label, correctness is equivalent to checking the actual vocabulary logits.
Source: the new Basis/model interface; exact ties follow the lowest-ID convention. -/
theorem task_correct_iff (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (mode : Transformer.Basis.Mode) (task : Transformer.Basis.Task)
    (tokens : Transformer.Basis.Tokens)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) (answer : Fin cfg.vocab_size)
    (hencode : Transformer.Basis.encodeTokens cfg.vocab_size tokens = some (head :: tail))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len)
    (hanswer : Transformer.Basis.taskNext mode task tokens = (answer.val : ℤ)) :
    tokenFunction cfg params eps tokens = Transformer.Basis.taskFunction mode task tokens ↔
      (∀ v, lastLogits cfg params eps head tail v ≤ lastLogits cfg params eps head tail answer) ∧
        ∀ v, lastLogits cfg params eps head tail answer ≤ lastLogits cfg params eps head tail v →
          answer.val ≤ v.val := by
  rw [Transformer.Basis.taskFunction, Transformer.Basis.extend, hanswer]
  exact tokenFunction_correct_iff cfg params eps tokens head tail answer hencode hlen

example : Transformer.Basis.encodeTokens controlConfig.vocab_size [1, 9] =
      some [⟨1, by decide⟩, ⟨9, by decide⟩] ∧
    ([⟨9, by decide⟩] : List (Fin controlConfig.vocab_size)).length + 1 ≤
      controlConfig.max_seq_len ∧
    Transformer.Basis.taskNext .easy .depth [1, 9] = (15 : ℤ) := by decide

/-- A checked context, the raw task label and an actual model margin imply task agreement.
Source: the common List Int functions and Basis's task-specific next-answer semantics. -/
theorem task_correct_of_margin (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (mode : Transformer.Basis.Mode) (task : Transformer.Basis.Task)
    (tokens : Transformer.Basis.Tokens)
    (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)) (answer : Fin cfg.vocab_size)
    (hencode : Transformer.Basis.encodeTokens cfg.vocab_size tokens = some (head :: tail))
    (hlen : tail.length + 1 ≤ cfg.max_seq_len)
    (hanswer : Transformer.Basis.taskNext mode task tokens = (answer.val : ℤ))
    (hmargin : ∀ v, v ≠ answer → lastLogits cfg params eps head tail v <
      lastLogits cfg params eps head tail answer) :
    tokenFunction cfg params eps tokens = Transformer.Basis.taskFunction mode task tokens := by
  rw [tokenFunction_of_encode cfg params eps tokens head tail hencode hlen,
    bestToken_of_strict cfg.vocab_pos _ answer hmargin]
  simp only [Transformer.Basis.taskFunction, Transformer.Basis.extend, hanswer]

example : Transformer.Basis.encodeTokens controlConfig.vocab_size [1, 9] =
      some [⟨1, by decide⟩, ⟨9, by decide⟩] ∧
    ([⟨9, by decide⟩] : List (Fin controlConfig.vocab_size)).length + 1 ≤
      controlConfig.max_seq_len ∧
    Transformer.Basis.taskNext .easy .depth [1, 9] = (15 : ℤ) ∧
    ∀ v : Fin controlConfig.vocab_size, v ≠ ⟨15, by decide⟩ →
      lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩ [⟨9, by decide⟩] v <
        lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩ [⟨9, by decide⟩]
          ⟨15, by decide⟩ := ⟨by decide, by decide, by decide, control_depth_margin⟩

/-- The full model solves a Basis task exactly when its predictions agree at all supervised prefixes.
Source: the actual model function and SolvesTask, without assuming or asserting weight existence. -/
theorem solvesTask_tokenFunction_iff (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (mode : Transformer.Basis.Mode) (task : Transformer.Basis.Task) :
    Transformer.Basis.SolvesTask (tokenFunction cfg params eps) mode task ↔
      ∀ tokens, Transformer.Basis.TaskPrefix mode task tokens →
        nextToken cfg params eps tokens = Transformer.Basis.taskNext mode task tokens :=
  Transformer.Basis.solvesTask_extend_iff _ _ _

/-- The model and every task produce lists of identical length before correctness is checked.
Source: the requested common List Int interface, including invalid-input continuations. -/
theorem model_task_lengths (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (mode : Transformer.Basis.Mode) (task : Transformer.Basis.Task)
    (tokens : Transformer.Basis.Tokens) :
    (tokenFunction cfg params eps tokens).length =
      (Transformer.Basis.taskFunction mode task tokens).length := by
  rw [tokenFunction_length, Transformer.Basis.taskFunction_length]

/-- The constructed passing prefix is genuinely supervised within the actual Basis context.
Source: depth E_2's raw-prefix grammar, not a fabricated PAD-only domain. -/
theorem control_prefix_supervised : Transformer.Basis.TaskPrefix .easy .depth [1, 9] := by
  exact ⟨[some false], by decide, by decide, rfl⟩

/-- This particular actual model does not solve the full task, despite a valid margin witness.
Source: the independently validated a,b counterexample and the actual model's PAD prediction. -/
theorem control_not_solves_depth :
    ¬Transformer.Basis.SolvesTask (tokenFunction controlConfig controlParams (1 / 100000))
      .easy .depth := by
  intro hsolve
  have hp : Transformer.Basis.TaskPrefix .easy .depth [1, 9, 10] := by
    exact ⟨[some false, some true], by decide, by decide, rfl⟩
  exact control_depth_failure (hsolve _ hp)

/-- Model and oracle both retain the same complete prefix on the positive control.
Source: the actual-model margin theorem and the depth integer semantics, compared in one type. -/
theorem control_task_agreement :
    tokenFunction controlConfig controlParams (1 / 100000) [1, 9] =
      Transformer.Basis.taskFunction .easy .depth [1, 9] := control_depth_one_prefix

end
end Transformer.GPTMini.TokenInterface
