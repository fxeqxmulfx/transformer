import Transformer.Basis.Depth
import Transformer.Basis.Recall
import Transformer.Basis.Parity

/-!
# The Basis suite in one raw-token function interface

Source: domain/basis.py, experiments/basis/experiment.py and the raw
synthetic generators at cbafbe9. Each task/mode gives a total List Int to
List Int function. Correctness of a candidate function is agreement at
every supervised prefix within the actual context limit. This is stronger
than the benchmark's sampled 99 percent sequence-accuracy stopping rule.

Depth and recall append their label at a supervised input position; that
label is not inserted into the benchmark's following random input stream.
Parity really generates its answer and EOS by two successive calls.
The model is trained separately for each task and mode; the same weights
are not required to assign incompatible easy/hard depth labels.
-/

namespace Transformer.Basis

/-- The two recipes; Source: domain/basis.py's MODES. -/
inductive Mode where
  | easy
  | hard
  deriving DecidableEq

/-- The three Basis tasks; Source: domain/basis.py's TASKS, including recall = MQAR. -/
inductive Task where
  | depth
  | recall
  | parity
  deriving DecidableEq

/-- Task-specific raw vocabulary sizes, including reserved symbols.
Source: Generator.vocab, MQAR.vocab and Answered.vocab at cbafbe9. -/
def vocabularySize : Task → ℕ
  | .depth => 36
  | .recall => 548
  | .parity => 68

/-- Actual model context caps include depth OOD evaluation and parity's answer input.
Source: Synthetic.context and Parity.context(16), not just the problem-size limits. -/
def contextSize : Task → ℕ
  | .depth => 128
  | .recall => 64
  | .parity => 19

/-- Raw next answers use the exact Basis easy/hard parameters.
Source: domain/basis.py; parity is shared between modes. -/
def taskNext (mode : Mode) : Task → Tokens → ℤ
  | .depth => depthNext (if mode = .easy then 2 else 4)
  | .recall => recallNext (if mode = .easy then easyRecall else hardRecall)
  | .parity => parityNext

/-- Every task has exactly the requested List Int to List Int type.
Source: the new append-one-answer formulation of supervised Basis prefixes. -/
def taskFunction (mode : Mode) (task : Task) : Tokens → Tokens := extend (taskNext mode task)

/-- The validated supervised grammar within the task's actual model context.
Source: the three raw-input oracles; sampler restrictions are documented in their adapters. -/
def TaskPrefix (mode : Mode) : Task → Tokens → Prop
  | .depth => DepthPrefix (contextSize .depth)
  | .recall => RecallPrefix (if mode = .easy then easyRecall else hardRecall) (contextSize .recall)
  | .parity => ParityPrefix (contextSize .parity)

/-- A candidate raw-token function solves a task when it agrees on all supervised prefixes.
Source: the new correctness predicate, stronger than finite sampled benchmark accuracy.
This defines the property of its arguments; it makes no existence claim about GPTMini weights. -/
def SolvesTask (f : Tokens → Tokens) (mode : Mode) (task : Task) : Prop :=
  ∀ tokens, TaskPrefix mode task tokens → f tokens = taskFunction mode task tokens

/-- All tasks satisfy the requested output length, even outside their supervised grammar.
Source: the common total append-one-token convention. -/
theorem taskFunction_length (mode : Mode) (task : Task) (tokens : Tokens) :
    (taskFunction mode task tokens).length = tokens.length + 1 := extend_length _ _

/-- All original tokens are retained exactly by every task function.
Source: the common continuation convention; no vocabulary conversion occurs in the oracle. -/
theorem taskFunction_prefix (mode : Mode) (task : Task) (tokens : Tokens) :
    (taskFunction mode task tokens).take tokens.length = tokens := extend_prefix _ _

/-- Parity is the same function in both modes, including its second EOS prediction.
Source: domain/basis.py shares the PARITY recipe rather than constructing a harder variant. -/
theorem parity_modes_same : taskFunction .easy .parity = taskFunction .hard .parity := by
  rfl

/-- For any append-one candidate, task correctness reduces to its discrete next answers.
Source: the new interface's cancellation theorem, with the actual supervised-prefix domain. -/
theorem solvesTask_extend_iff (next : Tokens → ℤ) (mode : Mode) (task : Task) :
    SolvesTask (extend next) mode task ↔
      ∀ tokens, TaskPrefix mode task tokens → next tokens = taskNext mode task tokens := by
  constructor
  · intro h tokens ht
    exact (extend_eq_iff _ _ _).mp (h tokens ht)
  · intro h tokens ht
    exact (extend_eq_iff _ _ _).mpr (h tokens ht)

/-- Agreement at both parity positions guarantees the complete free-generated answer.
Source: Basis parity's 1..16-bit limit and Parity.solve's two-token answer, without scratchpads. -/
theorem solvesTask_parity_twice (f : Tokens → Tokens) (mode : Mode)
    (hsolve : SolvesTask f mode .parity) (bits : List Bool)
    (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    f (f (parityPrompt bits)) = parityPrompt bits ++ [parityLabel bits, eos] := by
  have hbits : bits ≠ [] := by
    intro h
    simp [h] at hmin
  have hp : ParityPrefix 19 (parityPrompt bits) := by
    refine ⟨bits, hmin, hmax, ?_, Or.inl rfl⟩
    rw [parityPrompt_length]
    omega
  have ha : ParityPrefix 19 (parityPrompt bits ++ [parityLabel bits]) := by
    refine ⟨bits, hmin, hmax, ?_, Or.inr rfl⟩
    simp only [List.length_append, List.length_singleton, parityPrompt_length]
    omega
  have hs : f (parityPrompt bits) = parityFunction (parityPrompt bits) := hsolve _ hp
  have ht : f (parityPrompt bits ++ [parityLabel bits]) =
      parityFunction (parityPrompt bits ++ [parityLabel bits]) := hsolve _ ha
  rw [hs, parityFunction_prompt bits hbits, ht, parityFunction_answer bits hbits]

/-- A satisfiable function-level certificate, supplied by the actual parity interpreter.
Source: the raw oracle; this example supplies no GPTMini parameters or capacity theorem. -/
example : SolvesTask parityFunction .easy .parity ∧
    1 ≤ ([true, false] : List Bool).length ∧ ([true, false] : List Bool).length ≤ 16 := by
  refine ⟨?_, by decide, by decide⟩
  intro tokens _
  rfl

/-- The same raw function cannot solve both depth modes, since their labels conflict.
Source: Basis E_2/E_4 recipes; this justifies keeping the task mode outside the raw token list. -/
theorem depth_modes_incompatible (f : Tokens → Tokens) :
    ¬(SolvesTask f .easy .depth ∧ SolvesTask f .hard .depth) := by
  rintro ⟨he, hh⟩
  have hp : DepthPrefix 128 [1, 9, 11, 9, 10] := by
    exact ⟨[some false, none, some false, some true], by decide, by decide, rfl⟩
  have ha : f [1, 9, 11, 9, 10] = depthFunction 2 [1, 9, 11, 9, 10] := he _ hp
  have hb : f [1, 9, 11, 9, 10] = depthFunction 4 [1, 9, 11, 9, 10] := hh _ hp
  rw [depth_easy_example] at ha
  rw [depth_hard_example] at hb
  have hc := ha.symm.trans hb
  contradiction

/-- Easy mode really selects eight raw writes and 256 keys, not a two-record control.
Source: domain/basis.py; the fixed table below witnesses the actual supervised grammar. -/
theorem easyRecall_prefix : TaskPrefix .easy .recall
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] := by
  exact ⟨by decide, 292, by decide⟩

/-- Hard mode's raw sixteen writes include eight changed-value overwrites of eight keys.
Source: domain/basis.py and MQAR.sample/check; the last value of key 36 is 300. -/
theorem hardRecall_prefix : TaskPrefix .hard .recall
    [1, 36, 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299,
      36, 300, 37, 301, 38, 302, 39, 303, 40, 304, 41, 305, 42, 306, 43, 307, 36] := by
  exact ⟨by decide, 300, by decide⟩

end Transformer.Basis
