module

public import Transformer.Grokking.Convex.ConvexDepth.ContextInterpolation.Loss

/-!
# The original CE batches under fixed context interpolation

Source: `review/convex/moments.md`, CM42. Exact source-state rows on observed
inputs preserve the ordinary raw-label objective for all parameters. Original
record IDs retain repeated examples and the denominator of every minibatch.
The same identity applies before any optimizer step, including forward-only.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ContextInterpolation

variable {K I X C S : Type*} [Fintype K] [Fintype I] [Fintype C]

/-- The actual normalized full-vocabulary CE is jointly convex in all
head coefficients with the same budget, CM42. Context weights are fixed. -/
theorem batchLoss_convex (w : X → I → ℝ) (source : K → I → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (R : ℝ) :
    ConvexOn ℝ (headBudget R)
      (Fallback.batchLoss (features w source) inputs labels batch) := by
  have h := headLoss_convex (features w source)
    (fun s : ↥batch => inputs s.val) (fun s : ↥batch => labels s.val) R
  have hc : 0 ≤ (batch.card : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  refine ⟨headBudget_convex R, ?_⟩
  intro W hW V hV a b ha hb hab
  have hm := mul_le_mul_of_nonneg_left (h.2 hW hV ha hb hab) hc
  change (batch.card : ℝ)⁻¹ * _ ≤ a * ((batch.card : ℝ)⁻¹ * _) +
    b * ((batch.card : ℝ)⁻¹ * _)
  calc
    _ ≤ (batch.card : ℝ)⁻¹ * (a * headLoss _ _ _ W + b * headLoss _ _ _ V) := hm
    _ = _ := by ring

/-- Exact observed rows preserve the original normalized raw CE for every
head tensor, CM42. No assertion about labels at unseen inputs is used. -/
theorem batch_loss_eq [DecidableEq I] (w : X → I → ℝ) (source : K → I → ℝ)
    (state : X → I) (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hrow : ∀ s ∈ batch, ∀ i, w (inputs s) i = if i = state (inputs s) then 1 else 0)
    (W : K → C → ℝ) :
    Fallback.batchLoss (features w source) inputs labels batch W =
      Fallback.batchLoss source (fun s => state (inputs s)) labels batch W := by
  unfold Fallback.batchLoss headLoss
  congr 1
  apply Finset.sum_congr rfl
  intro s _
  have heq : headLogits (features w source) W (inputs s.val) =
      headLogits source W (state (inputs s.val)) := by
    funext c
    unfold headLogits
    apply Finset.sum_congr rfl
    intro k _
    rw [features_at w source (inputs s.val) _ (hrow s.val s.property)]
  rw [heq]

/-- Complete Fréchet derivatives agree on the original observed batches,
CM42. At differentiability points these are the actual derivatives; elsewhere
`fderiv` has its usual library convention. The fixed weights are not trained. -/
theorem batch_fderiv_eq [DecidableEq I] (w : X → I → ℝ) (source : K → I → ℝ)
    (state : X → I) (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hrow : ∀ s ∈ batch, ∀ i, w (inputs s) i = if i = state (inputs s) then 1 else 0)
    (W : K → C → ℝ) :
    fderiv ℝ (Fallback.batchLoss (features w source) inputs labels batch) W =
      fderiv ℝ (Fallback.batchLoss source (fun s => state (inputs s)) labels batch) W := by
  have heq : Fallback.batchLoss (features w source) inputs labels batch =
      Fallback.batchLoss source (fun s => state (inputs s)) labels batch := by
    funext V
    exact batch_loss_eq w source state inputs labels batch hrow V
  rw [heq]

/-- Actual derivative existence transfers too, CM42. This does not assume
differentiability of a nearest-neighbor selection with respect to its inputs. -/
theorem batch_hasFDerivAt [DecidableEq I] (w : X → I → ℝ) (source : K → I → ℝ)
    (state : X → I) (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hrow : ∀ s ∈ batch, ∀ i, w (inputs s) i = if i = state (inputs s) then 1 else 0)
    (W : K → C → ℝ) (D : (K → C → ℝ) →L[ℝ] ℝ)
    (hd : HasFDerivAt (Fallback.batchLoss source (fun s => state (inputs s)) labels batch) D W) :
    HasFDerivAt (Fallback.batchLoss (features w source) inputs labels batch) D W := by
  have heq : Fallback.batchLoss (features w source) inputs labels batch =
      Fallback.batchLoss source (fun s => state (inputs s)) labels batch := by
    funext V
    exact batch_loss_eq w source state inputs labels batch hrow V
  rw [heq]
  exact hd

/-- Four context inputs: observed 0 and 1 are retained, while the remaining
two use a fixed uniform feature average, CM42's batch witness. -/
noncomputable def batchWeights (x i : Fin 4) : ℝ :=
  if x.val < 2 then pointWeights id x i else 1 / 4

/-- Original IDs 0 and 1 repeat input 0; the final ID contains input 1. -/
def batchInputs (s : Fin 3) : Fin 4 := if s = 2 then 1 else 0

/-- The concrete repeated-input records satisfy the exact-row premise. -/
theorem batchWeights_observed (s : Fin 3) (i : Fin 4) :
    batchWeights (batchInputs s) i = if i = batchInputs s then 1 else 0 := by
  fin_cases s <;> simp [batchWeights, batchInputs, pointWeights]

example := batchLoss_convex batchWeights (complexFeatures fourier4) batchInputs batchInputs
  (Finset.univ : Finset (Fin 3)) 64
example := batch_loss_eq batchWeights (complexFeatures fourier4) id batchInputs batchInputs
  (Finset.univ : Finset (Fin 3)) (fun s _ => batchWeights_observed s)
  (complexCoefficients (complexInitializer fourier4 id 4))
-- The short final batch has denominator one, rather than the full batch's three.
example := batch_loss_eq batchWeights (complexFeatures fourier4) id batchInputs batchInputs
  ({2} : Finset (Fin 3)) (fun s _ => batchWeights_observed s)
  (complexCoefficients (complexInitializer fourier4 id 4))
example := batch_fderiv_eq batchWeights (complexFeatures fourier4) id batchInputs batchInputs
  (Finset.univ : Finset (Fin 3)) (fun s _ => batchWeights_observed s)
  (complexCoefficients (complexInitializer fourier4 id 4))
-- A nonempty repeated-record batch with zero source features has a genuine
-- zero derivative. The preceding examples also cover nonzero complex logits.
example := batch_hasFDerivAt batchWeights (fun (_ : Fin 2) (_ : Fin 4) => 0)
  id batchInputs batchInputs (Finset.univ : Finset (Fin 3))
  (fun s _ => batchWeights_observed s) (fun _ _ => 1) 0 (by
    have hz (V : Fin 2 → Fin 4 → ℝ) (x : Fin 4) :
        headLogits (fun (_ : Fin 2) (_ : Fin 4) => 0) V x = (fun _ : Fin 4 => 0) := by
      funext c
      simp [headLogits]
    have heq : Fallback.batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
        (fun s => id (batchInputs s)) batchInputs (Finset.univ : Finset (Fin 3)) =
        fun _ : Fin 2 → Fin 4 → ℝ => Fallback.batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
          batchInputs batchInputs Finset.univ 0 := by
      funext V
      simp only [Fallback.batchLoss, headLoss, hz]
    rw [heq]
    exact hasFDerivAt_const _ _)

end Transformer.Grokking.Convex.ConvexDepth.ContextInterpolation
