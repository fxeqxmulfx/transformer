import Transformer.GPTMini.TokenInterface.Causal
import Transformer.Basis.Encoding.Tasks
import Transformer.Basis.Requirements

/-!
# The exact task property required of a given GPTMini

Source: the actual Basis raw-prefix grammars and GPTMini.forward at
cbafbe9/f11b6e2. Vocabulary/context compatibility suffices to prove that
all task prefixes enter the real model path. Correctness is then exactly
the maximum/tie property of its actual logits on every valid prefix.
This theorem characterizes the required property of given parameters;
its statement does not quantify existentially over model weights.

The comparison includes depth order, raw key/value binding, hard recall
overwrites and both parity answer/EOS positions. Architectural causality
already lets the same property be checked in full teacher-forced rows.
-/

namespace Transformer.GPTMini.TokenInterface

open scoped Classical
noncomputable section

/-- Every supervised prefix follows the actual model branch, rather than any PAD fallback.
Source: the independently proved vocabulary/context/nonempty properties of Basis inputs. -/
theorem taskPrefix_model_path (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (mode : Transformer.Basis.Mode) (task : Transformer.Basis.Task)
    (hV : Transformer.Basis.vocabularySize task ≤ cfg.vocab_size)
    (hcap : Transformer.Basis.contextSize task ≤ cfg.max_seq_len)
    (tokens : Transformer.Basis.Tokens) (ht : Transformer.Basis.TaskPrefix mode task tokens) :
    ∃ (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size)),
      Transformer.Basis.encodeTokens cfg.vocab_size tokens = some (head :: tail) ∧
      tail.length + 1 ≤ cfg.max_seq_len ∧
      nextToken cfg params eps tokens =
        (bestToken cfg.vocab_pos (lastLogits cfg params eps head tail)).val := by
  obtain ⟨head, tail, he, hlen⟩ :=
    Transformer.Basis.Encoding.model_input mode task tokens _ _ hV hcap ht
  refine ⟨head, tail, he, hlen, ?_⟩
  rw [← Transformer.Basis.decode_encode he]
  exact nextToken_decode_cons cfg params eps head tail hlen

example : Transformer.Basis.vocabularySize .depth ≤ controlConfig.vocab_size ∧
    Transformer.Basis.contextSize .depth ≤ controlConfig.max_seq_len ∧
    Transformer.Basis.TaskPrefix .easy .depth [1, 9] :=
  ⟨by decide, by decide, control_prefix_supervised⟩

/-- For given model parameters, complete task correctness is exactly a property of all its logits.
Source: the independent raw task labels and the actual greedy readout, including ties.
The only hypotheses are vocabulary/context compatibility, proved by concrete architecture examples. -/
theorem solvesTask_iff_logit_property (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (mode : Transformer.Basis.Mode) (task : Transformer.Basis.Task)
    (hV : Transformer.Basis.vocabularySize task ≤ cfg.vocab_size)
    (hcap : Transformer.Basis.contextSize task ≤ cfg.max_seq_len) :
    Transformer.Basis.SolvesTask (tokenFunction cfg params eps) mode task ↔
      ∀ tokens, Transformer.Basis.TaskPrefix mode task tokens →
        ∃ (head : Fin cfg.vocab_size) (tail : List (Fin cfg.vocab_size))
          (answer : Fin cfg.vocab_size),
          Transformer.Basis.encodeTokens cfg.vocab_size tokens = some (head :: tail) ∧
          tail.length + 1 ≤ cfg.max_seq_len ∧
          (answer.val : ℤ) = Transformer.Basis.taskNext mode task tokens ∧
          (∀ v, lastLogits cfg params eps head tail v ≤ lastLogits cfg params eps head tail answer) ∧
          (∀ v, lastLogits cfg params eps head tail answer ≤ lastLogits cfg params eps head tail v →
            answer.val ≤ v.val) := by
  constructor
  · intro hsolve tokens ht
    obtain ⟨head, tail, he, hlen, hnext⟩ :=
      taskPrefix_model_path cfg params eps mode task hV hcap tokens ht
    let answer := bestToken cfg.vocab_pos (lastLogits cfg params eps head tail)
    have hn := (solvesTask_tokenFunction_iff cfg params eps mode task).mp hsolve tokens ht
    have ha : (answer.val : ℤ) = Transformer.Basis.taskNext mode task tokens := hnext.symm.trans hn
    refine ⟨head, tail, answer, he, hlen, ha, ?_⟩
    exact (bestToken_eq_iff cfg.vocab_pos _ answer).mp rfl
  · intro hproperty tokens ht
    obtain ⟨head, tail, answer, he, hlen, ha, hmax, htie⟩ := hproperty tokens ht
    exact (task_correct_iff cfg params eps mode task tokens head tail answer he hlen ha.symm).mpr
      ⟨hmax, htie⟩

example : Transformer.Basis.vocabularySize .depth ≤ controlConfig.vocab_size ∧
    Transformer.Basis.contextSize .depth ≤ controlConfig.max_seq_len := ⟨by decide, by decide⟩

/-- A given model solves easy depth exactly when its actual prefix logits satisfy the required property.
Source: the small Basis depth input shape; no layer count, width or weights are existentially chosen. -/
theorem depth_logit_property (params : ModelParams controlConfig) (eps : ℝ) :
    Transformer.Basis.SolvesTask (tokenFunction controlConfig params eps) .easy .depth ↔
      ∀ tokens, Transformer.Basis.TaskPrefix .easy .depth tokens →
        ∃ (head : Fin controlConfig.vocab_size) (tail : List (Fin controlConfig.vocab_size))
          (answer : Fin controlConfig.vocab_size),
          Transformer.Basis.encodeTokens controlConfig.vocab_size tokens = some (head :: tail) ∧
          tail.length + 1 ≤ controlConfig.max_seq_len ∧
          (answer.val : ℤ) = Transformer.Basis.taskNext .easy .depth tokens ∧
          (∀ v, lastLogits controlConfig params eps head tail v ≤
            lastLogits controlConfig params eps head tail answer) ∧
          (∀ v, lastLogits controlConfig params eps head tail answer ≤
            lastLogits controlConfig params eps head tail v → answer.val ≤ v.val) :=
  solvesTask_iff_logit_property controlConfig params eps .easy .depth (by decide) (by decide)

/-- Any bag-invariant next-answer computation fails the actual eight-write MQAR task.
Source: the independent raw-binding pair; this applies to the real model function as an argument. -/
theorem model_bag_obstruction (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    ¬(Transformer.Basis.PermutationInvariant (nextToken cfg params eps) ∧
      Transformer.Basis.SolvesTask (tokenFunction cfg params eps) .easy .recall) := by
  rintro ⟨hbag, hsolve⟩
  exact Transformer.Basis.recall_not_bag_solver (nextToken cfg params eps) hbag hsolve

/-- A correct model must distinguish chronological overwrites in its actual next-token prediction.
Source: Basis hard MQAR's same-bag sixteen-write pair, with model parameters universally given. -/
theorem model_overwrite_obstruction (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    ¬(nextToken cfg params eps (Transformer.Basis.rewritePrefix 292 300) =
        nextToken cfg params eps (Transformer.Basis.rewritePrefix 300 292) ∧
      Transformer.Basis.SolvesTask (tokenFunction cfg params eps) .hard .recall) := by
  rintro ⟨heq, hsolve⟩
  exact Transformer.Basis.recall_requires_last_write (nextToken cfg params eps) hsolve heq

/-- Collapsing the two equal-count depth words prevents correctness of the actual model.
Source: E_2's validated aabb/abab pair, testing the next prediction rather than retained inputs. -/
theorem model_order_obstruction (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    ¬(nextToken cfg params eps [1, 9, 9, 10, 10] =
        nextToken cfg params eps [1, 9, 10, 9, 10] ∧
      Transformer.Basis.SolvesTask (tokenFunction cfg params eps) .easy .depth) := by
  rintro ⟨heq, hsolve⟩
  exact Transformer.Basis.depth_requires_order (nextToken cfg params eps) hsolve heq

/-- Ignoring one-bit content prevents the actual model from solving even the smallest parity inputs.
Source: parity's equal-length legal EVEN/ODD prompts and its real next-token adapter. -/
theorem model_bit_obstruction (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    ¬(nextToken cfg params eps (Transformer.Basis.parityPrompt [false]) =
        nextToken cfg params eps (Transformer.Basis.parityPrompt [true]) ∧
      Transformer.Basis.SolvesTask (tokenFunction cfg params eps) .easy .parity) := by
  rintro ⟨heq, hsolve⟩
  exact Transformer.Basis.parity_requires_bits (nextToken cfg params eps) hsolve heq

/-- Architectural causality and input legality alone are insufficient for the required logit property.
Source: the actual small control model's proved depth failure, with no unknown-weight premise. -/
theorem control_fails_logit_property :
    ¬(∀ tokens, Transformer.Basis.TaskPrefix .easy .depth tokens →
      ∃ (head : Fin controlConfig.vocab_size) (tail : List (Fin controlConfig.vocab_size))
        (answer : Fin controlConfig.vocab_size),
        Transformer.Basis.encodeTokens controlConfig.vocab_size tokens = some (head :: tail) ∧
        tail.length + 1 ≤ controlConfig.max_seq_len ∧
        (answer.val : ℤ) = Transformer.Basis.taskNext .easy .depth tokens ∧
        (∀ v, lastLogits controlConfig controlParams (1 / 100000) head tail v ≤
          lastLogits controlConfig controlParams (1 / 100000) head tail answer) ∧
        (∀ v, lastLogits controlConfig controlParams (1 / 100000) head tail answer ≤
          lastLogits controlConfig controlParams (1 / 100000) head tail v → answer.val ≤ v.val)) := by
  intro hproperty
  exact control_not_solves_depth ((depth_logit_property controlParams (1 / 100000)).mpr hproperty)

end
end Transformer.GPTMini.TokenInterface
