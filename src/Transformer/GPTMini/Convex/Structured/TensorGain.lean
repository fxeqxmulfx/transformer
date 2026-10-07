import Transformer.GPTMini.Convex.Structured.TensorBasisStackTraining

/-!
# Fixed linear gain on every free tensor potential

Source: plan.md §4 and experiments/basis_tensor's first equal-FLOP
easy-depth failure on 2026-10-07: the real complete NLL falls but the
model still predicts only reject after 659 ordinary AdamW updates.
This candidate changes coordinates, not the model class or optimizer:
every free embedding/Q/K/value/state/position/head field is gain * θ.
The ten decoder axes, protected anchor, actual RMS/residual/tied stack
and fixed-zero deferred FFN keep their original definitions.

The actual complete sample and minibatch objectives remain globally
convex jointly in every unrestricted coordinate. For nonzero gain,
explicit inverse coordinates recover every original real model/loss;
the given finite weights still solve all six complete raw Basis recipes.
Gain is fixed architectural metadata, not a learned parameter. This
does not remove gauge directions, prove AdamW convergence, or assert
that the proposed gain eight repairs the measured failure. Its extra
arithmetic and empirical success must be checked in the lab.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- Actual full tensor integer function with a fixed gain on all free potentials.
Source: TensorStackInterface.tensorFunction, evaluated at a genuine common scaled parameter assignment. -/
def tensorGainFunction (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps gain : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) : Tokens → Tokens :=
  tensorFunction cfg hsize hwidth eps (gain • θ)

/-- Nonzero gain recovers every original true-stack integer function at explicit inverse coordinates.
Source: the actual common parameter scaling and the real field scalar-action inverse; no logits or correct route are assumed. -/
theorem tensorGainFunction_recover (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps gain : ℝ) (hgain : gain ≠ 0) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) :
    tensorGainFunction cfg hsize hwidth eps gain (gain⁻¹ • θ) = tensorFunction cfg hsize hwidth eps θ := by
  unfold tensorGainFunction
  rw [smul_inv_smul₀ hgain]

example : (basisTensorConfig .easy .depth).vocab_size ≤ 1024 ∧
    64 ≤ (basisTensorConfig .easy .depth).d_model ∧ (8 : ℝ) ≠ 0 :=
  ⟨by decide, by decide, by norm_num⟩

/-- Every actual gained true-stack prediction equals inference by that same gained raw learned joint.
Source: unrestricted TensorStackInterface.tensorFunction_mixed, including the actual encoding/context fallbacks. -/
theorem tensorGainFunction_mixed (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps : ℝ) (heps : 0 < eps) (gain : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) :
    tensorGainFunction cfg hsize hwidth eps gain θ = mixedFunction cfg.vocab_pos hsize (gain • θ) := by
  unfold tensorGainFunction
  exact tensorFunction_mixed cfg hsize hwidth eps heps (gain • θ)

example : (basisTensorConfig .hard .recall).vocab_size ≤ 1024 ∧
    64 ≤ (basisTensorConfig .hard .recall).d_model ∧ (0 : ℝ) < 1 / 100000 := by
  exact ⟨by decide, by decide, by norm_num⟩

/-- Actual gain changes no part of the required append-one integer-list interface.
Source: the genuine tensor function's length law at the scaled free assignment, on valid and invalid inputs alike. -/
theorem tensorGainFunction_length (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps gain : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens) :
    (tensorGainFunction cfg hsize hwidth eps gain θ tokens).length = tokens.length + 1 :=
  tensorFunction_length cfg hsize hwidth eps (gain • θ) tokens

example : (basisTensorConfig .hard .parity).vocab_size ≤ 1024 ∧
    64 ≤ (basisTensorConfig .hard .parity).d_model := by decide

/-- Every raw integer remains in physical order under the actual gained stack.
Source: genuine tensorFunction_prefix at the same scaled parameters, without any task-validity premise. -/
theorem tensorGainFunction_prefix (cfg : Config) (hsize : cfg.vocab_size ≤ 1024) (hwidth : 64 ≤ cfg.d_model)
    (eps gain : ℝ) (θ : BindingParameters cfg.vocab_size cfg.max_seq_len) (tokens : Tokens) :
    (tensorGainFunction cfg hsize hwidth eps gain θ tokens).take tokens.length = tokens :=
  tensorFunction_prefix cfg hsize hwidth eps (gain • θ) tokens

example : (basisTensorConfig .easy .recall).vocab_size ≤ 1024 ∧
    64 ≤ (basisTensorConfig .easy .recall).d_model := by decide

/-- Explicit finite coordinates realizing the already proved complete raw Basis weights at nonzero gain.
Source: TensorBasisModel.basisMixedParameters and genuine scalar inverse; no reference rule enters learned forward. -/
def basisGainWitness (mode : Mode) (task : Task) (gain : ℝ) :
    BindingParameters (vocabularySize task) (contextSize task) := gain⁻¹ • basisMixedParameters mode task

/-- The actual gained given-weight full stack solves every complete raw Basis recipe.
Source: exact inverse-coordinate recovery and TensorBasisModel.basisTensor_solves, including true two-call parity generation. -/
theorem basisGain_solves (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (gain : ℝ) (hgain : gain ≠ 0) :
    SolvesTask (tensorGainFunction (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1
      (basisTensorConfig_fits mode task).2 eps gain (basisGainWitness mode task gain)) mode task := by
  unfold basisGainWitness
  rw [tensorGainFunction_recover _ _ _ eps gain hgain]
  exact basisTensor_solves mode task eps heps

example : (0 : ℝ) < 1 / 100000 ∧ (8 : ℝ) ≠ 0 := ⟨by norm_num, by norm_num⟩

/-- Actual gained configured-stack complete sample loss, using unchanged correct raw-data labels.
Source: TensorBasisStackTraining.basisStackNLL at all scaled free fields, with observations absent from inference. -/
def basisGainNLL (mode : Mode) (task : Task) (eps gain : ℝ) (head : Fin (vocabularySize task))
    (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : ℝ :=
  basisStackNLL mode task eps head tail hcap (gain • θ)

/-- The actual gained loss is negative log of the same real gained inference joint at the correct data configuration.
Source: genuine full-stack NLL/probability coupling, now at the common scaled assignment throughout the actual stack. -/
theorem basisGainNLL_eq (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps) (gain : ℝ)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisGainNLL mode task eps gain head tail hcap θ = -Real.log
      (tensorStackProbability (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1 (basisTensorConfig_fits mode task).2
        eps (gain • θ) (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail)) := by
  unfold basisGainNLL
  exact basisStackNLL_eq mode task eps heps head tail hcap (gain • θ)

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 36), 9, 10] : List (Fin 36)).length ≤ contextSize .depth :=
  ⟨by norm_num, by decide⟩

/-- Fixed gain preserves global joint convexity of actual correctly labeled full-stack training in all unrestricted coordinates.
Source: true basisStackNLL_convex and composition with scalar multiplication as a real linear map. -/
theorem basisGainNLL_convex (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps) (gain : ℝ)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (basisGainNLL mode task eps gain head tail hcap) := by
  unfold basisGainNLL
  simpa only [Set.preimage_univ, Function.comp_def, LinearMap.lsmul_apply] using
    (basisStackNLL_convex mode task eps heps head tail hcap).comp_linearMap
      (LinearMap.lsmul ℝ (BindingParameters (vocabularySize task) (contextSize task)) gain)

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity :=
  ⟨by norm_num, by decide⟩

/-- Every gained actual complete objective remains nonnegative at every simultaneous free coordinate assignment.
Source: the normalized positive true-stack joint's unchanged complete likelihood at the scaled parameters. -/
theorem basisGainNLL_nonneg (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps) (gain : ℝ)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    0 ≤ basisGainNLL mode task eps gain head tail hcap θ :=
  basisStackNLL_nonneg mode task eps heps head tail hcap (gain • θ)

example : (0 : ℝ) < 1 / 100000 ∧ ([(1 : Fin 68), 21, 22, 18] : List (Fin 68)).length ≤ contextSize .parity :=
  ⟨by norm_num, by decide⟩

/-- Nonzero gain recovers every original actual sample loss, rather than a weaker objective with different targets.
Source: genuine common scalar inverse at every free parameter and the same basisStackNLL computation. -/
theorem basisGainNLL_recover (mode : Mode) (task : Task) (eps gain : ℝ) (hgain : gain ≠ 0)
    (head : Fin (vocabularySize task)) (tail : List (Fin (vocabularySize task)))
    (hcap : (head :: tail).length ≤ contextSize task) (θ : BindingParameters (vocabularySize task) (contextSize task)) :
    basisGainNLL mode task eps gain head tail hcap (gain⁻¹ • θ) = basisStackNLL mode task eps head tail hcap θ := by
  unfold basisGainNLL
  rw [smul_inv_smul₀ hgain]

example : (8 : ℝ) ≠ 0 ∧ ([(1 : Fin 36), 9, 10] : List (Fin 36)).length ≤ contextSize .depth :=
  ⟨by norm_num, by decide⟩

/-- Actual shared gained minibatch training retains all genuine full-stack sample computations and raw data targets.
Source: basisStackBatchNLL at one common scaled assignment, with no per-example or per-layer free fields. -/
def basisGainBatchNLL {B : Type*} [Fintype B] (mode : Mode) (task : Task) (eps gain : ℝ)
    (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : ℝ :=
  basisStackBatchNLL mode task eps head tail hcap (gain • θ)

/-- Every real gained minibatch objective remains globally convex jointly in all unrestricted shared coordinates.
Source: actual full-stack variable-length batch convexity under the same common real linear map. -/
theorem basisGainBatchNLL_convex {B : Type*} [Fintype B] (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps) (gain : ℝ)
    (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (basisGainBatchNLL mode task eps gain head tail hcap) := by
  unfold basisGainBatchNLL
  simpa only [Set.preimage_univ, Function.comp_def, LinearMap.lsmul_apply] using
    (basisStackBatchNLL_convex mode task eps heps head tail hcap).comp_linearMap
      (LinearMap.lsmul ℝ (BindingParameters (vocabularySize task) (contextSize task)) gain)

example : (0 : ℝ) < 1 / 100000 ∧
    (∀ b : Fin 2, ((1 : Fin 68) :: (if b = 0 then [22, 18] else [21, 22, 18])).length ≤ contextSize .parity) := by
  refine ⟨by norm_num, ?_⟩
  intro b
  fin_cases b <;> decide

end
end Transformer.GPTMini.Convex.Structured
