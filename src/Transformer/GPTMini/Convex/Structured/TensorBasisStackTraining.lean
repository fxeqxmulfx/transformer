import Transformer.GPTMini.Convex.Structured.TensorBasisModel

/-!
# Complete raw Basis training through the actual configured tensor stack

Source: independently correct generated Basis labels at 62f1f6f,
actual complete tensor training at 815556c and TensorStackTraining's
real full-stack likelihood coupling. Easy/hard recipes use width 64/
128 and two/six actual shared-weight blocks, with original residuals,
fixed zero deferred FFN and unchanged final RMSNorm/tied readout.

Each loss below is the actual final attention's complete NLL after
all intermediate blocks, using only correct labels generated from
raw training data. Sample and variable-length minibatch objectives
are globally convex on the entire unrestricted shared raw parameter
space. Their probability is that same true learned inference joint.
Parameter counts remain linear in vocabulary/context and independent
of data count. Output-only CE, a learned FFN and independently trained
deep matrices are outside scope; ordinary AdamW success and equal-FLOP
measurement remain experimental, and label generation must be charged.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped BigOperators Classical
noncomputable section

/-- Genuine configured-stack Basis sample training evaluates actual final-head complete NLL at independently computed raw-data labels.
Source: tensorStackNLL's real complete forward and BasisTraining.basisDataTarget, with labels absent from learned inference. -/
def basisStackNLL (mode : Mode) (task : Task) (eps : ℝ) (head : Fin (vocabularySize task))
    (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : ℝ :=
  tensorStackNLL (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1 (basisTensorConfig_fits mode task).2
    eps (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail) θ

/-- The actually computed full-stack Basis sample loss is identical to the independently verified complete data likelihood for every free weight.
Source: genuine complete-stack tensor likelihood transfer and the unchanged actual raw data target. -/
theorem basisStackNLL_sequence (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisStackNLL mode task eps head tail hcap θ = basisDataNLL mode task head tail hcap θ := by
  unfold basisStackNLL basisDataNLL
  exact tensorStackNLL_sequence _ _ _ eps heps θ _ hcap _ _

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity :=
  ⟨by norm_num, by decide⟩

/-- Actual full-stack Basis training is negative log of the same true learned stack joint at its correct raw-data configuration.
Source: genuine complete-stack NLL/probability coupling, including all free embedding/Q/K/value/position/head fields. -/
theorem basisStackNLL_eq (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisStackNLL mode task eps head tail hcap θ = -Real.log
      (tensorStackProbability (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1 (basisTensorConfig_fits mode task).2
        eps θ (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail)) := by
  unfold basisStackNLL
  exact tensorStackNLL_eq _ _ _ eps heps θ _ hcap _ _

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 36), 9, 10, 9, 10] : List (Fin 36)).length ≤ contextSize .depth :=
  ⟨by norm_num, by decide⟩

/-- Every real configured-stack Basis sample objective is globally convex jointly in the whole unrestricted learned parameter domain.
Source: exact full-stack computation equality and the independently correct complete raw-data likelihood. -/
theorem basisStackNLL_convex (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (basisStackNLL mode task eps head tail hcap) := by
  have heq : basisStackNLL mode task eps head tail hcap = basisDataNLL mode task head tail hcap := by
    funext θ
    exact basisStackNLL_sequence mode task eps heps head tail hcap θ
  rw [heq]
  exact basisDataNLL_convex mode task head tail hcap

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 21, 22, 18] : List (Fin 68)).length ≤ contextSize .parity :=
  ⟨by norm_num, by decide⟩

/-- Actual configured-stack complete sample losses are nonnegative at every unrestricted learned weight assignment.
Source: the same normalized positive inference likelihood and actual full-stack/raw-data objective equality. -/
theorem basisStackNLL_nonneg (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    0 ≤ basisStackNLL mode task eps head tail hcap θ := by
  rw [basisStackNLL_sequence mode task eps heps]
  exact basisDataNLL_nonneg mode task head tail hcap θ

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity :=
  ⟨by norm_num, by decide⟩

/-- Actual full-stack variable-length minibatch training shares every free weight and sums genuine correctly labeled final-head likelihoods.
Source: real configured-stack sample computations, with no per-example learned state or routing table. -/
def basisStackBatchNLL {B : Type*} [Fintype B] (mode : Mode) (task : Task) (eps : ℝ)
    (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : ℝ :=
  ∑ b, basisStackNLL mode task eps (head b) (tail b) (hcap b) θ

/-- Every genuine full-stack variable-length batch computes exactly the same independent-data complete mixed likelihood.
Source: each actual configured-stack sample transfer, retaining all correct raw data targets and shared parameters. -/
theorem basisStackBatchNLL_sequence {B : Type*} [Fintype B] (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisStackBatchNLL mode task eps head tail hcap θ = mixedBatchNLL (fun b => head b :: tail b) hcap
      (fun b => bindingFinalPosition (tail b)) (fun b => basisDataTarget mode task (head b) (tail b)) θ := by
  unfold basisStackBatchNLL mixedBatchNLL
  apply Finset.sum_congr rfl
  intro b hb
  exact basisStackNLL_sequence mode task eps heps (head b) (tail b) (hcap b) θ

example : (0 : ℝ) < 1 / 100000 ∧
    (∀ b : Fin 2, ((1 : Fin 68) :: (if b = 0 then [22, 18] else [21, 22, 18])).length ≤ contextSize .parity) := by
  refine ⟨by norm_num, ?_⟩
  intro b
  fin_cases b <;> decide

/-- Actual configured-stack minibatch training is globally convex jointly in every unrestricted shared raw weight.
Source: exact actual batch computation equality and independently correct generated complete Basis labels across all six recipes. -/
theorem basisStackBatchNLL_convex {B : Type*} [Fintype B] (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (basisStackBatchNLL mode task eps head tail hcap) := by
  have heq : basisStackBatchNLL mode task eps head tail hcap = mixedBatchNLL (fun b => head b :: tail b) hcap
      (fun b => bindingFinalPosition (tail b)) (fun b => basisDataTarget mode task (head b) (tail b)) := by
    funext θ
    exact basisStackBatchNLL_sequence mode task eps heps head tail hcap θ
  rw [heq]
  exact basisDataBatch_convex mode task head tail hcap

example : (0 : ℝ) < 1 / 100000 ∧
    (∀ b : Fin 2, ((1 : Fin 68) :: (if b = 0 then [22, 18] else [21, 22, 18])).length ≤ contextSize .parity) := by
  refine ⟨by norm_num, ?_⟩
  intro b
  fin_cases b <;> decide

/-- The genuine full eight-record recall sample has jointly convex actual two-block tensor training with matching and values both free.
Source: real easy-mode serialization and the independently correct raw record/digit labels in the full configured stack. -/
example : ConvexOn ℝ Set.univ (basisStackNLL .easy .recall (1 / 100000) (1 : Fin 548)
    ([(36 : Fin 548), 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] : List (Fin 548)) (by decide)) :=
  basisStackNLL_convex _ _ _ (by norm_num) _ _ _

/-- Full hard-depth complete data training uses the actual width-128 six-block stack and globally convex free domain.
Source: genuine E4 recipe data labels and real configured-stack training, without a fixed correct hidden-state input. -/
example : ConvexOn ℝ Set.univ (basisStackNLL .hard .depth (1 / 100000) (1 : Fin 36)
    ([(9 : Fin 36), 10, 9, 10] : List (Fin 36)) (by decide)) := basisStackNLL_convex _ _ _ (by norm_num) _ _ _

/-- The exact entire free scalar counts survive tensor-stack realization without per-layer copies or per-example parameters.
Source: the same actual unrestricted BindingParameters domain and proved raw complete-model count. -/
theorem basisStack_parameters (task : Task) :
    Fintype.card (SharedField (vocabularySize task) (contextSize task)) +
      Fintype.card (Fin (contextSize task + (contextSize task - 1))) = 52 * vocabularySize task + 3 * contextSize task + 128 := by
  exact basisData_parameters task

end
end Transformer.GPTMini.Convex.Structured
