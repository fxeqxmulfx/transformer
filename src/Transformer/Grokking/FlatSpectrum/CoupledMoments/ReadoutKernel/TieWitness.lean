module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.LossTies

/-!
# Equal training CE with opposite unseen pooled predictions

Source: `review/flat_spectrum.md`, CM36. The original complete-kernel search
refuses the missing corner and uses the explicitly declared fallback class1.
The grouped search accepts answer0. Both actual bounded complex models retain
the same raw training logits through full, mini and zero-update schedules.
This is a model-class witness, not a claim about every native fallback order.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

open KernelCodes ConvexDepth
open Landscape.Neural (crossEntropy)

/-- Actual original search, with explicit fallback1 after refusal, CM36. -/
noncomputable def cornerRawDefault (z : CodeStates cornerRawBasis) : ZMod 2 × Bool :=
  cornerLabels (codeRepresentative cornerRawBasis [0, 1, 2] 1 z)

/-- Actual grouped search and declared raw representative, CM36. -/
noncomputable def cornerGroupedDefault (z : CodeStates cornerGroupedBasis) : ZMod 2 × Bool :=
  cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0 z)

/-- The ordinary full-vocabulary training objective cannot distinguish
these two complete-kernel heads at any common finite CE schedule, CM36. -/
theorem corner_training_loss_tied (batches : ℕ → Finset ↥cornerObserved)
    (rates : ℕ → ℝ) (n : ℕ) :
    (∑ x : ↥cornerObserved, crossEntropy
      (vectorCELogits cornerRawBasis cornerObserved cornerHeadLabels
        cornerRawDefault batches rates n x.val) (cornerHeadLabels x)) / cornerObserved.card =
    (∑ x : ↥cornerObserved, crossEntropy
      (vectorCELogits cornerGroupedBasis cornerObserved cornerHeadLabels
        cornerGroupedDefault batches rates n x.val) (cornerHeadLabels x)) / cornerObserved.card := by
  exact vectorCE_training_loss_eq cornerRawBasis cornerGroupedBasis cornerObserved
    cornerHeadLabels cornerRawDefault cornerGroupedDefault batches rates n

/-- Refusal preserves the chosen raw fallback in the actual complex
forward pass after every finite schedule, CM36. It was never certified correct. -/
theorem corner_raw_unseen_logits (batches : ℕ → Finset ↥cornerObserved)
    (rates : ℕ → ℝ) (n : ℕ) :
    vectorCELogits cornerRawBasis cornerObserved cornerHeadLabels
      cornerRawDefault batches rates n (true, true) =
        fun c : ZMod 2 × Bool => if c = (1, false) then 24 else 0 := by
  rw [vectorCE_unseen_logits cornerRawBasis cornerObserved cornerHeadLabels _ batches rates n
    (by decide : (true, true) ∉ cornerObserved)]
  change (fun c => if c = cornerLabels (codeRepresentative cornerRawBasis [0, 1, 2] 1
    (codeState cornerRawBasis (true, true))) then 24 else 0) = _
  rw [representative_of_refusal cornerRawBasis [0, 1, 2] 1 true true corner_raw_refuses]
  rfl

/-- The additional grouped match supplies a different actual unseen
margin row under the same schedule, CM36. No target is passed to the head. -/
theorem corner_grouped_unseen_logits (batches : ℕ → Finset ↥cornerObserved)
    (rates : ℕ → ℝ) (n : ℕ) :
    vectorCELogits cornerGroupedBasis cornerObserved cornerHeadLabels
      cornerGroupedDefault batches rates n (true, true) =
        fun c : ZMod 2 × Bool => if c = (0, false) then 24 else 0 := by
  rw [vectorCE_unseen_logits cornerGroupedBasis cornerObserved cornerHeadLabels _ batches rates n
    (by decide : (true, true) ∉ cornerObserved)]
  change (fun c => if c = cornerLift (codeRepresentative cornerGroupedBasis [0, 1] 0
    (codeState cornerGroupedBasis (true, true))) then 24 else 0) = _
  have hm : chooseClass cornerGroupedBasis (queryCode cornerGroupedBasis true true) [0, 1] =
      some 0 := by
    rw [corner_grouped_match]
    exact chooseClass_first _ _ _
  rw [representative_of_match cornerGroupedBasis [0, 1] 0 true true hm]
  rfl

/-- Despite their equal observed CE, the ungrouped fallback loses the
original pooled answer and the grouped head wins, for both compatible raw
completions, CM36. This is a strict actual-complex prediction difference. -/
theorem corner_tie_pooled_miss (batches : ℕ → Finset ↥cornerObserved)
    (rates : ℕ → ℝ) (n : ℕ) (which : Bool) :
    quotientScore Prod.fst (vectorCELogits cornerRawBasis cornerObserved cornerHeadLabels
      cornerRawDefault batches rates n (true, true)) (cornerTarget which true true).1 <
    quotientScore Prod.fst (vectorCELogits cornerRawBasis cornerObserved cornerHeadLabels
      cornerRawDefault batches rates n (true, true)) 1 ∧
    quotientScore Prod.fst (vectorCELogits cornerGroupedBasis cornerObserved cornerHeadLabels
      cornerGroupedDefault batches rates n (true, true)) 1 <
    quotientScore Prod.fst (vectorCELogits cornerGroupedBasis cornerObserved cornerHeadLabels
      cornerGroupedDefault batches rates n (true, true)) (cornerTarget which true true).1 := by
  rw [corner_raw_unseen_logits, corner_grouped_unseen_logits]
  have ht : (cornerTarget which true true).1 = (0 : ZMod 2) := rfl
  rw [ht]
  constructor
  · exact quotientScore_margin_strict Prod.fst (1, false) 24
      (native_margin_exceeds_vocabulary (by decide)) (by decide)
  · exact quotientScore_margin_strict Prod.fst (0, false) 24
      (native_margin_exceeds_vocabulary (by decide)) (by decide)

example := corner_training_loss_tied (fun _ => Finset.univ) (fun _ => 1) 3
example := corner_training_loss_tied cornerBatches (fun _ => 1) 4
example := corner_training_loss_tied (fun _ => Finset.univ) (fun _ => 1) 0

example := corner_tie_pooled_miss (fun _ => Finset.univ) (fun _ => 1) 3 true
example := corner_tie_pooled_miss cornerBatches (fun _ => 1) 4 false
example := corner_tie_pooled_miss (fun _ => Finset.univ) (fun _ => 1) 0 true

example := vectorCE_budget cornerRawBasis cornerObserved cornerHeadLabels cornerRawDefault
  cornerBatches (fun _ => 1) 4
example := vectorCE_budget cornerGroupedBasis cornerObserved cornerHeadLabels cornerGroupedDefault
  cornerBatches (fun _ => 1) 4

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
