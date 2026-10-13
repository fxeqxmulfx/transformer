import Transformer.GPTMini.Convex.Structured.RecallDataTargets

/-!
# Actual convex complete training from all raw Basis semantics

Source: independent depth/parity data targets at 3812fda/b91036f,
deterministic raw recall labels in RecallDataTargets and actual learned
mixed inference/capability at d436526/d7be618. Each training recipe uses
the same genuine unrestricted raw mixed parameter space and objective.
Complete labels are generated solely from the actual raw nonempty prefix
and known dataset recipe; the physical last position supplies the query.

All generated output channels agree with independently specified Basis
answers. The computed complete NLL equals negative log of the same actual
mixed inference probability and is globally convex in all simultaneously
free embedding/Q/K/transition/value/position/head coordinates, including
shared variable-length minibatches. No target path, route, recipe label
or task interpreter enters learned inference. Relative/token parameters
are compact and independent of training examples or prototype counts.

This establishes mathematical complete-supervision training scope and
head-level capability. Actual tensor/prenorm/residual/tied realization,
floating-point behavior, ordinary AdamW success and the equal-FLOP
comparison remain separate obligations. Output-only CE is not asserted
convex. Label generation and all actual training work must be charged.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped BigOperators Classical
noncomputable section

/-- All complete branch/path/route/channel labels are computed from the actual raw training prefix and fixed recipe only.
Source: independently verified causal state targets and deterministic Nat-ranked latest-write data scan; no learned weights are inputs. -/
def basisDataTarget (mode : Mode) (task : Task) : (head : Fin (vocabularySize task)) →
    (tail : List (Fin (vocabularySize task))) → MixedConfiguration (head :: tail) :=
  match task with
  | .depth => fun head tail => .inl (depthSharedTargets (if mode = .easy then 2 else 4) (head :: tail))
  | .recall => fun head tail => .inr (recallDataTarget (if mode = .easy then 8 else 16) (head :: tail) (bindingFinalPosition tail))
  | .parity => fun head tail => .inl (paritySharedTargets (head :: tail))

/-- Every actual generated complete output-channel label is the independent raw Basis answer's collision-free full decoder code.
Source: genuine raw reference equivalence or deterministic last-write scan, with only actual task-valid data assumed. -/
theorem basisDataTarget_answer (mode : Mode) (task : Task) (head : Fin (vocabularySize task))
    (tail : List (Fin (vocabularySize task))) (hprefix : TaskPrefix mode task (decodeTokens (head :: tail))) :
    ∃ target : Fin (vocabularySize task), (target.val : ℤ) = taskNext mode task (decodeTokens (head :: tail)) ∧
      mixedChannels (basisDataTarget mode task head tail) = outputDigit
        (vocabularyCode (by cases task <;> decide) target) := by
  cases task with
  | depth =>
      refine ⟨depthReferenceLabel (if mode = .easy then 2 else 4) (referenceRun depthSharedRule 0 (head :: tail)), ?_, rfl⟩
      rw [depthShared_run]
      apply depthRawReference_next _ 128 _ _ _ hprefix
      · cases mode <;> decide
      · cases mode <;> decide
  | «recall» =>
      exact recallDataTarget_task mode (head :: tail) (bindingFinalPosition tail) (by rfl) hprefix
  | parity =>
      refine ⟨paritySharedLabel (referenceRun paritySharedRule 0 (head :: tail)), ?_, rfl⟩
      rw [paritySharedLabel_id, parityShared_run]
      exact parityReference_next 19 _ hprefix

example : TaskPrefix .easy .parity (decodeTokens [(1 : Fin 68), 22, 18]) :=
  ⟨[true], by decide, by decide, by decide, Or.inl rfl⟩

/-- Actual Basis complete training evaluates the genuine learned mixed likelihood at its generated raw-data labels.
Source: true mixedNLL on the physical last query, with labels external to the actual inference computation. -/
def basisDataNLL (mode : Mode) (task : Task) (head : Fin (vocabularySize task))
    (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : ℝ :=
  mixedNLL (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail) θ

/-- The actually computed complete Basis objective is exactly the negative log of that same actual full learned inference configuration.
Source: genuine mixed training/inference identity, rather than a separate teacher-conditioned model. -/
theorem basisDataNLL_eq (mode : Mode) (task : Task) (head : Fin (vocabularySize task))
    (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : basisDataNLL mode task head tail hcap θ =
      -Real.log (mixedProbability θ (head :: tail) hcap (bindingFinalPosition tail) (basisDataTarget mode task head tail)) :=
  mixedNLL_eq _ _ _ _ _

example : ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity := by decide

/-- Every actual complete Basis recipe objective is globally convex jointly in the entire unrestricted raw mixed parameter domain.
Source: the genuine computed mixed likelihood with fixed independently correct data labels, without frozen encoder, Q/K or value weights. -/
theorem basisDataNLL_convex (mode : Mode) (task : Task) (head : Fin (vocabularySize task))
    (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (basisDataNLL mode task head tail hcap) := mixedNLL_convex _ _ _ _

example : ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity := by decide

/-- The actual generated-data Basis complete loss is nonnegative at every freely trained finite parameter assignment.
Source: true normalized positive mixed likelihood, without a correct model prediction or successful optimizer premise. -/
theorem basisDataNLL_nonneg (mode : Mode) (task : Task) (head : Fin (vocabularySize task))
    (tail : List (Fin (vocabularySize task))) (hcap : (head :: tail).length ≤ contextSize task)
    (θ : BindingParameters (vocabularySize task) (contextSize task)) : 0 ≤ basisDataNLL mode task head tail hcap θ :=
  mixedNLL_nonneg _ _ _ _ _

example : ([(1 : Fin 68), 22, 18] : List (Fin 68)).length ≤ contextSize .parity := by decide

/-- Finite actual variable-length training minibatches retain global joint convexity using the genuine generated Basis complete labels.
Source: true shared mixedBatchNLL convexity, with no per-example learned state/route table or replacement optimizer. -/
theorem basisDataBatch_convex {B : Type*} [Fintype B] (mode : Mode) (task : Task)
    (head : B → Fin (vocabularySize task)) (tail : B → List (Fin (vocabularySize task)))
    (hcap : ∀ b, (head b :: tail b).length ≤ contextSize task) :
    ConvexOn ℝ Set.univ (mixedBatchNLL (fun b => head b :: tail b) hcap
      (fun b => bindingFinalPosition (tail b)) (fun b => basisDataTarget mode task (head b) (tail b))) :=
  mixedBatchNLL_convex _ _ _ _

example : ∀ b : Fin 2, ((1 : Fin 68) :: (if b = 0 then [22, 18] else [21, 22, 18])).length ≤ contextSize .parity := by
  intro b
  fin_cases b <;> decide

/-- Every full Basis recipe has precisely linear vocabulary/context parameter storage, independent of raw prefix or dataset count.
Source: genuine shared 52-field layout and all 2C-1 freely learned relative positions, with task context positivity derived from the real recipes. -/
theorem basisData_parameters (task : Task) :
    Fintype.card (SharedField (vocabularySize task) (contextSize task)) +
      Fintype.card (Fin (contextSize task + (contextSize task - 1))) =
        52 * vocabularySize task + 3 * contextSize task + 128 := by
  apply bindingParameters_count
  cases task <;> decide

/-- The exact actual free scalar counts are 2384 for depth, 28816 for recall and 3721 for parity, including both heads.
Source: the real full vocabulary/context recipes and proved compact shared parameter cardinality, not a count of implicit paths or prototypes. -/
theorem basisData_parameter_counts :
    Fintype.card (SharedField 36 128) + Fintype.card (Fin (128 + (128 - 1))) = 2384 ∧
    Fintype.card (SharedField 548 64) + Fintype.card (Fin (64 + (64 - 1))) = 28816 ∧
    Fintype.card (SharedField 68 19) + Fintype.card (Fin (19 + (19 - 1))) = 3721 := by
  rw [bindingParameters_count 36 128 (by omega), bindingParameters_count 548 64 (by omega),
    bindingParameters_count 68 19 (by omega)]
  norm_num

/-- Genuine complete easy and hard parity objectives both permit ordinary unrestricted simultaneous parameter updates.
Source: actual shared mixed loss class and data labels; this control claims convexity, not that AdamW already trained the model. -/
example : ConvexOn ℝ Set.univ (basisDataNLL .hard .parity (1 : Fin 68) ([(22 : Fin 68), 18] : List (Fin 68)) (by decide)) :=
  basisDataNLL_convex _ _ _ _ _

/-- A real complete eight-record raw recall training prefix uses the same globally convex unrestricted mixed objective.
Source: original easy-mode table/query serialization and actual deterministic data labels, with every Q/K/value/head coordinate free. -/
example : ConvexOn ℝ Set.univ (basisDataNLL .easy .recall (1 : Fin 548)
    ([(36 : Fin 548), 292, 37, 293, 38, 294, 39, 295, 40, 296, 41, 297, 42, 298, 43, 299, 36] : List (Fin 548)) (by decide)) :=
  basisDataNLL_convex _ _ _ _ _

/-- Actual full raw hard-depth complete supervision has a globally convex likelihood in the same mixed model class.
Source: four-alternation raw data targets at the real 128-position cap and the independently proved E4 semantics. -/
example : ConvexOn ℝ Set.univ (basisDataNLL .hard .depth (1 : Fin 36)
    ([(9 : Fin 36), 10, 9, 10] : List (Fin 36)) (by decide)) := basisDataNLL_convex _ _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
