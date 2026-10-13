import Transformer.GPTMini.Semantics.ParityCorrectness
import Transformer.GPTMini.Semantics.RecallCorrectness
import Transformer.GPTMini.Semantics.DepthCorrectness
import Transformer.GPTMini.TokenInterface.Correctness

/-!
# Complete original softmax GPTMini Basis family

Source: Basis's three raw generators/recipes at cbafbe9, the archived
GPTMini.forward at f11b6e2 and the complete original-model constructions
in ParityCorrectness, RecallCorrectness and DepthCorrectness. Each run
has its own vocabulary and shared parameter assignment, as in Basis.
The same assignment handles every validated prefix of that task/mode.

The substantive all-task theorem follows the actual embedding, complete
original hidden loop, tied normalized logits and checked integer adapter.
Neither correct logits nor a prepared task encoder is a hypothesis.
Parity includes label and EOS and actual two-call free generation.

All six grammars are covered in real arithmetic at shared epsilon in
(0,1/64]. Depth uses the original easy 64/two and hard 128/six models.
Recall and parity use 64/two even on the hard grammar: no 128/six lift
is asserted for those tasks. Original four heads, FFN expansion four,
vocabularies and context caps are retained. Python's unequal default
head/RMS epsilons and floating-point rounding are outside this theorem.
Recall gains may depend on epsilon but never on a particular raw prefix.
Given-weight correctness does not establish AdamW convergence, convex
trainability or a measured benchmark-training FLOP budget.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface Transformer.Basis

/-- The concrete original configurations for separate Basis task/mode runs.
Source: the complete original constructions; recall/parity retain small dimensions in both grammars. -/
noncomputable def basisModelConfig (mode : Mode) : Task → Config
  | .depth => depthConfig mode
  | .recall => recallConfig
  | .parity => countConfig

/-- Given shared original parameters for each task/mode, independent of the input prefix.
Source: actual complete depth, eight/sixteen-write recall and parity/EOS parameter families. -/
noncomputable def basisModelParams (mode : Mode) (eps : ℝ) :
    (task : Task) → ModelParams (basisModelConfig mode task)
  | .depth => depthModelParams mode
  | .recall => recallModelParams (if mode = .easy then 8 else 16) eps
  | .parity => decoderParams 1024 131072

/-- The genuine checked original-model callback in exactly the requested raw integer-list type.
Source: tokenFunction, with task-specific original configurations and given shared parameters. -/
noncomputable def basisModelFunction (mode : Mode) (eps : ℝ) (task : Task) : Tokens → Tokens :=
  tokenFunction (basisModelConfig mode task) (basisModelParams mode eps task) eps

/-- All three complete Basis grammars in either mode are solved by these actual original softmax models.
Source: the full raw-input task theorems, with their common shared-epsilon range made explicit. -/
theorem basisModel_solves (mode : Mode) (task : Task) (eps : ℝ)
    (heps : 0 < eps) (hsmall : eps ≤ 1 / 64) :
    SolvesTask (basisModelFunction mode eps task) mode task := by
  have hclip : eps ≤ 1 := by linarith
  cases task with
  | depth => exact depthModel_solves_depth mode eps heps hclip
  | «recall» => exact recallModel_solves_recall eps heps.le hclip mode
  | parity => exact decoder_solves_parity eps heps hsmall mode

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 := by norm_num

/-- The actual next-token predictor equals the independent data semantics on every validated supervised prefix.
Source: complete raw task correctness and the checked append-one model's exact prediction equivalence. -/
theorem basisModel_predicts (mode : Mode) (task : Task) (eps : ℝ)
    (heps : 0 < eps) (hsmall : eps ≤ 1 / 64) (tokens : Tokens)
    (hprefix : TaskPrefix mode task tokens) :
    nextToken (basisModelConfig mode task) (basisModelParams mode eps task) eps tokens =
      taskNext mode task tokens :=
  (solvesTask_tokenFunction_iff _ _ _ _ _).mp (basisModel_solves mode task eps heps hsmall) tokens hprefix

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    TaskPrefix .easy .depth [1, 9] := ⟨by norm_num, by norm_num, control_prefix_supervised⟩

/-- Every actual original callback appends exactly one integer on its entire input domain.
Source: tokenFunction's total checked interface, including empty/invalid/overlong fallback inputs. -/
theorem basisModel_length (mode : Mode) (eps : ℝ) (task : Task) (tokens : Tokens) :
    (basisModelFunction mode eps task tokens).length = tokens.length + 1 :=
  tokenFunction_length _ _ _ _

/-- The actual model preserves the full integer prefix in its original order on every call.
Source: the checked public continuation interface, independent of the supervised grammar. -/
theorem basisModel_prefix (mode : Mode) (eps : ℝ) (task : Task) (tokens : Tokens) :
    (basisModelFunction mode eps task tokens).take tokens.length = tokens :=
  tokenFunction_prefix _ _ _ _

/-- All concrete task configurations use exactly Basis's original raw vocabulary and context cap.
Source: vocabularySize/contextSize and the original complete-model configuration records. -/
theorem basisModel_domain (mode : Mode) (task : Task) :
    (basisModelConfig mode task).vocab_size = vocabularySize task ∧
      (basisModelConfig mode task).max_seq_len = contextSize task := by
  cases task <;> exact ⟨rfl, rfl⟩

/-- The complete constructions fit width 64 except hard depth, which uses original width 128.
Source: original depth small/large recipes and the explicitly small recall/parity constructions. -/
theorem basisModel_width (mode : Mode) (task : Task) :
    (basisModelConfig mode task).d_model = if mode = .hard ∧ task = .depth then 128 else 64 := by
  cases mode <;> cases task <;> rfl

/-- The given full hidden loops have two layers except hard depth's original six-layer loop.
Source: actual ModelParams and original configuration records, not a claim about an unproved lifted model. -/
theorem basisModel_layers (mode : Mode) (task : Task) :
    (basisModelConfig mode task).n_layers = if mode = .hard ∧ task = .depth then 6 else 2 := by
  cases mode <;> cases task <;> rfl

/-- The original four heads and four-times-width FFN are retained for every complete Basis construction.
Source: all three actual original Config records at cbafbe9/f11b6e2. -/
theorem basisModel_blocks (mode : Mode) (task : Task) :
    (basisModelConfig mode task).n_heads = 4 ∧
      (basisModelConfig mode task).d_ff = 4 * (basisModelConfig mode task).d_model := by
  cases mode <;> cases task <;> exact ⟨rfl, rfl⟩

/-- At reference RMS epsilon 1e-5 the actual original family simultaneously solves all six full raw grammars.
Source: basisModel_solves with no additional encoder, routing or task-correctness premise. -/
theorem basisModel_reference_all :
    ∀ mode task, SolvesTask (basisModelFunction mode (1 / 100000) task) mode task := by
  intro mode task
  exact basisModel_solves mode task _ (by norm_num) (by norm_num)

/-- Two actual calls freely produce the parity label then EOS, using the first prediction as the second input.
Source: complete supervised-prefix correctness and Basis's genuine two-token generation semantics. -/
theorem basisModel_parity_twice (mode : Mode) (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (bits : List Bool) (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    let f := basisModelFunction mode eps .parity
    f (f (parityPrompt bits)) = parityPrompt bits ++ [parityLabel bits, eos] :=
  solvesTask_parity_twice _ mode (basisModel_solves mode .parity eps heps hsmall) bits hmin hmax

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    1 ≤ ([true, false] : List Bool).length ∧ ([true, false] : List Bool).length ≤ 16 := by norm_num

/-- Real ordered depth computation distinguishes equal-multiplicity words through the same complete public family.
Source: actual original easy depth's universal order-sensitive correctness control. -/
theorem basisModel_order_control :
    basisModelFunction .easy (1 / 100000) .depth [1, 9, 9, 10, 10] = [1, 9, 9, 10, 10, 16] ∧
      basisModelFunction .easy (1 / 100000) .depth [1, 9, 10, 9, 10] = [1, 9, 10, 9, 10, 15] :=
  depthModel_easy_order_control _ (by norm_num) (by norm_num)

/-- Actual adjacent binding survives in the full family: swapping neighboring values changes the raw answer.
Source: original eight-write recall's derived encoder/retrieval and independent binding controls. -/
theorem basisModel_binding_control :
    basisModelFunction .easy (1 / 100000) .recall (bindingPrefix 292 293) = bindingPrefix 292 293 ++ [292] ∧
      basisModelFunction .easy (1 / 100000) .recall (bindingPrefix 293 292) = bindingPrefix 293 292 ++ [293] :=
  recallModel_swapped_bindings _ (by norm_num) (by norm_num)

end Transformer.GPTMini.Semantics
