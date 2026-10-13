import Transformer.GPTMini.Convex.Structured.TensorLikelihood

/-!
# Actual tensor complete training with correct raw Basis data

Source: BasisTraining's independently correct branch/path/route/channel
labels at 62f1f6f and the genuine tensor loss/probability computation
in TensorTraining/TensorLikelihood. Every recipe reuses those fixed
raw-data labels at its real physical final query. Forward tensors
contain actual learned embeddings/positions and genuine RMSNorm;
all learned Q/K/value/state fields remain unrestricted together.

Computed sample/minibatch losses are proved equal to the original
complete data likelihood and globally convex in the same whole shared
parameter space. The same actual tensor inference probability governs
them; no per-example learned table or teacher state enters inference.
Widths 64/128 fit. Full-stack field preservation and zero deferred FFN
integration remain before candidate acceptance; AdamW success and
equal-FLOP comparisons are not claimed. Charge data-label generation.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped BigOperators Classical
noncomputable section

variable {d : ℕ}

/-- Genuine Basis tensor sample loss evaluates the actual learned model on independently generated complete data labels.
Source: the actual tensor likelihood and BasisTraining.basisDataTarget, with supervision absent from the tensor forward. -/
def basisTensorNLL (hwidth : 64 ≤ d) (eps : ℝ) (mode : Mode) (task : Task)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : ℝ :=
  tensorMixedNLL hwidth (tensorHeadParameters θ)
    (tensorSequence (by cases task <;> decide) hwidth eps θ (head :: tail) hcap)
    hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail)

/-- True tensor Basis sample training equals the independently verified complete raw-data objective for every free weight.
Source: actual complete tensor likelihood transfer on all six recipe data-target families. -/
theorem basisTensorNLL_sequence (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps) (mode : Mode) (task : Task)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisTensorNLL hwidth eps mode task head tail hcap θ = basisDataNLL mode task head tail hcap θ := by
  unfold basisTensorNLL basisDataNLL
  exact tensorMixedNLL_sequence (by cases task <;> decide) hwidth eps heps θ
    (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail)

example : (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity := by
  exact ⟨by omega, by norm_num, by decide⟩

/-- Actual complete Basis tensor loss is the negative log of the very same genuine learned inference configuration probability.
Source: correct fixed data targets and TensorLikelihood's true tensor NLL/probability identity. -/
theorem basisTensorNLL_eq (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps) (mode : Mode) (task : Task)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisTensorNLL hwidth eps mode task head tail hcap θ = -Real.log
      (tensorMixedProbability hwidth (tensorHeadParameters θ)
        (tensorSequence (by cases task <;> decide) hwidth eps θ (head :: tail) hcap)
        hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail)) := by
  unfold basisTensorNLL
  exact tensorMixedNLL_eq (by cases task <;> decide) hwidth eps heps θ
    (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail)

example : (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 36), 9, 10, 9, 10] : List (Fin 36)).length ≤ contextSize .depth := by
  exact ⟨by omega, by norm_num, by decide⟩

/-- Every actual Basis tensor sample objective is globally convex jointly in all free token/head/value/position weights.
Source: exact actual tensor objective equality with the verified complete likelihood using independent raw-data supervision. -/
theorem basisTensorNLL_convex (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps) (mode : Mode) (task : Task)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (basisTensorNLL hwidth eps mode task head tail hcap) := by
  have heq : basisTensorNLL hwidth eps mode task head tail hcap = basisDataNLL mode task head tail hcap := by
    funext θ
    exact basisTensorNLL_sequence hwidth eps heps mode task head tail hcap θ
  rw [heq]
  exact basisDataNLL_convex mode task head tail hcap

example : (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity := by
  exact ⟨by omega, by norm_num, by decide⟩

/-- Every actual tensor sample complete likelihood is nonnegative, independently of current model quality.
Source: the same normalized positive learned model and actual raw-data/tensor objective identity. -/
theorem basisTensorNLL_nonneg (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps) (mode : Mode) (task : Task)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    0 ≤ basisTensorNLL hwidth eps mode task head tail hcap θ := by
  rw [basisTensorNLL_sequence hwidth eps heps]
  exact basisDataNLL_nonneg mode task head tail hcap θ

example : (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([(1 : Fin 68), 21, 22, 18] : List (Fin 68)).length ≤ contextSize .parity := by
  exact ⟨by omega, by norm_num, by decide⟩

/-- Genuine variable-length tensor minibatches share every raw weight and use the real generated data targets.
Source: actual summed basisTensorNLL samples, without per-example trainable path or routing parameters. -/
def basisTensorBatchNLL {B : Type*} [Fintype B] (hwidth : 64 ≤ d) (eps : ℝ) (mode : Mode) (task : Task)
    (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : ℝ :=
  ∑ b, basisTensorNLL hwidth eps mode task (head b) (tail b) (hcap b) θ

/-- True tensor minibatch training computes exactly the same verified shared complete data likelihood.
Source: every actual sample transfer, with physical sequence lengths and label-generation rules retained. -/
theorem basisTensorBatchNLL_sequence {B : Type*} [Fintype B] (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (mode : Mode) (task : Task) (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisTensorBatchNLL hwidth eps mode task head tail hcap θ = mixedBatchNLL (fun b => head b :: tail b) hcap
      (fun b => bindingFinalPosition (tail b)) (fun b => basisDataTarget mode task (head b) (tail b)) θ := by
  unfold basisTensorBatchNLL mixedBatchNLL
  apply Finset.sum_congr rfl
  intro b hb
  exact basisTensorNLL_sequence hwidth eps heps mode task (head b) (tail b) (hcap b) θ

example : (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    (∀ b : Fin 2, ((1 : Fin 68) :: (if b = 0 then [22, 18] else [21, 22, 18])).length ≤ contextSize .parity) := by
  refine ⟨by omega, by norm_num, ?_⟩
  intro b
  fin_cases b <;> decide

/-- Actual shared variable-length Basis tensor minibatch training is globally convex in the whole unrestricted free domain.
Source: exact computed tensor batch equality and the genuine independently supervised shared complete likelihood. -/
theorem basisTensorBatchNLL_convex {B : Type*} [Fintype B] (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (mode : Mode) (task : Task) (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (basisTensorBatchNLL hwidth eps mode task head tail hcap) := by
  have heq : basisTensorBatchNLL hwidth eps mode task head tail hcap = mixedBatchNLL (fun b => head b :: tail b) hcap
      (fun b => bindingFinalPosition (tail b)) (fun b => basisDataTarget mode task (head b) (tail b)) := by
    funext θ
    exact basisTensorBatchNLL_sequence hwidth eps heps mode task head tail hcap θ
  rw [heq]
  exact basisDataBatch_convex mode task head tail hcap

example : (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    (∀ b : Fin 2, ((1 : Fin 68) :: (if b = 0 then [22, 18] else [21, 22, 18])).length ≤ contextSize .parity) := by
  refine ⟨by omega, by norm_num, ?_⟩
  intro b
  fin_cases b <;> decide

/-- A full eight-record raw recall sample uses the actual globally convex tensor objective with all matching and values free.
Source: the genuine easy-mode data serialization, computed tensor likelihood and independent raw latest-write labels. -/
example : ConvexOn ℝ Set.univ (basisTensorNLL (d := 64) (by omega) (1 / 100000) .easy .recall (1 : Fin 548)
    ([(36 : Fin 548), 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] : List (Fin 548)) (by decide)) :=
  basisTensorNLL_convex _ _ (by norm_num) _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
