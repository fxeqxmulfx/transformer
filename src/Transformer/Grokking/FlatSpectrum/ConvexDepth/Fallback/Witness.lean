module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Batch
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHeadBudget

/-!
# A bounded complex fallback preserves one input and fails on another

Source: `review/flat_spectrum.md`, CM38. The candidate maps two distinct inputs
to an observed state. The second input therefore has support but can still
have the wrong answer. A third input is unsupported and retains the parent's
actual logits. Both blocks use the proved flat Fourier code and margin24.
This is a finite model-class witness, not a learned benchmark endpoint.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback

open Matrix
open Landscape.Neural (crossEntropy)

/-- The original observed input set of the counterexample, CM38. -/
def witnessObserved : Finset (Fin 3) := {0}

/-- A nonconstant candidate state map that aliases inputs0 and1, CM38. -/
def witnessCandidateState (x : Fin 3) : Fin 4 := if x = 2 then 1 else 0

/-- The actual observed-state support test used by the route, CM38. -/
def witnessKnown (x : Fin 3) : Bool :=
  decide (witnessCandidateState x ∈ witnessObserved.image witnessCandidateState)

/-- A raw target with a different label at the unobserved alias, CM38.
It is used only to check this counterexample, not in the support predicate. -/
def witnessTarget (x : Fin 3) : Bool := decide (x ≠ 0)

/-- The declared margin labels of both actual coefficient blocks, CM38.
The parent uses distinct states; the candidate aliases two input states. -/
def witnessLabels (s : Fin 4) : Bool := decide (s ≠ 0)

/-- Actual conjugate-transform complex parameters, CM38. -/
noncomputable def witnessHead : Matrix (Fin 4) Bool ℂ :=
  complexInitializer fourier4 witnessLabels 24

/-- Fixed routed real/imaginary features from two flat component codes, CM38. -/
noncomputable def witnessFeatures : (Fin 4 × Bool) ⊕ (Fin 4 × Bool) → Fin 3 → ℝ :=
  features witnessKnown (fun k x => complexFeatures fourier4 k (witnessCandidateState x))
    (fun l x => complexFeatures fourier4 l x.castSucc)

/-- Both stored actual coefficient blocks are charged, CM38. -/
noncomputable def witnessCoefficients : (Fin 4 × Bool) ⊕ (Fin 4 × Bool) → Bool → ℝ :=
  coefficients (complexCoefficients witnessHead) (complexCoefficients witnessHead)

/-- The witness's actual combined forward pass, CM38. -/
noncomputable def witnessLogits (x : Fin 3) : Bool → ℝ :=
  headLogits witnessFeatures witnessCoefficients x

/-- The saved parent forward pass, before introducing the candidate, CM38. -/
noncomputable def witnessParentLogits (x : Fin 3) : Bool → ℝ :=
  headLogits (complexFeatures fourier4) (complexCoefficients witnessHead) x.castSucc

/-- Input1 is unseen yet supported; input2 is unseen and unsupported,
CM38. This is finite input/state bookkeeping, not a correctness predicate. -/
theorem witness_support : witnessKnown 0 = true ∧ witnessKnown 1 = true ∧
    witnessKnown 2 = false ∧ (1 : Fin 3) ∉ witnessObserved ∧ (2 : Fin 3) ∉ witnessObserved := by
  decide

/-- The actual Fourier products yield the routed margin row, CM38. -/
theorem witness_logits (x : Fin 3) :
    witnessLogits x = if witnessKnown x = true then
      (fun c => if c = witnessLabels (witnessCandidateState x) then 24 else 0) else
      (fun c => if c = witnessLabels x.castSucc then 24 else 0) := by
  unfold witnessLogits witnessFeatures witnessCoefficients
  rw [logits]
  change (if witnessKnown x = true then
    headLogits (complexFeatures fourier4) (complexCoefficients witnessHead)
      (witnessCandidateState x) else
    headLogits (complexFeatures fourier4) (complexCoefficients witnessHead) x.castSucc) = _
  simp only [witnessHead, complexInitializer_logits fourier4 fourier4_unitary]

/-- Both concrete complex heads fit the native64-per-state budget, and
their combined point freezes the parent's block, CM38. -/
theorem witness_budget : witnessCoefficients ∈
    parentDomain (complexCoefficients witnessHead) 512 := by
  have h := complexInitializer_native_budget fourier4 fourier4_unitary witnessLabels
  have hh := coefficients_mem h h
  norm_num at hh
  simpa only [witnessCoefficients, witnessHead] using hh

/-- The routed model and saved parent have the same ordinary raw-label
CE on the only observed input, CM38. This is equality of actual forward losses. -/
theorem witness_training_ce :
    crossEntropy (witnessLogits 0) (witnessTarget 0) =
      crossEntropy (witnessParentLogits 0) (witnessTarget 0) := by
  rw [witness_logits]
  unfold witnessParentLogits witnessHead
  rw [complexInitializer_logits fourier4 fourier4_unitary]
  norm_num [witnessKnown, witnessCandidateState, witnessObserved, witnessTarget, witnessLabels]

/-- Support alone cannot guarantee preservation of an already correct
parent answer: the supported unseen alias gets a strictly wrong raw prediction,
CM38, despite the equal observed CE and bounded flat component codes. -/
theorem witness_covered_regression : witnessKnown 1 = true ∧ (1 : Fin 3) ∉ witnessObserved ∧
    witnessParentLogits 1 false < witnessParentLogits 1 (witnessTarget 1) ∧
    witnessLogits 1 (witnessTarget 1) < witnessLogits 1 false := by
  refine ⟨witness_support.2.1, witness_support.2.2.2.1, ?_, ?_⟩
  · unfold witnessParentLogits witnessHead
    rw [complexInitializer_logits fourier4 fourier4_unitary]
    norm_num [witnessLabels, witnessTarget]
  · rw [witness_logits]
    norm_num [witnessKnown, witnessCandidateState, witnessObserved, witnessLabels, witnessTarget]

/-- In contrast, the unsupported input retains the exact saved parent
logits, CM38. This follows from the actual affine-domain theorem. -/
theorem witness_unsupported_preserved : witnessKnown 2 = false ∧
    witnessLogits 2 = witnessParentLogits 2 := by
  refine ⟨witness_support.2.2.1, ?_⟩
  exact parent_logits witnessKnown _ _ witness_budget witness_support.2.2.1

example := complex_features_bound witnessKnown fourier4 fourier4
  fourier4_unitary fourier4_unitary witnessCandidateState Fin.castSucc (1 : Fin 3)

example := complex_logits_bound witnessKnown fourier4 fourier4
  fourier4_unitary fourier4_unitary witnessCandidateState Fin.castSucc
  witness_budget.1 (1 : Fin 3) true

end Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback
