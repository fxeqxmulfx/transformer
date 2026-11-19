module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale.Basic

/-!
# Original training batches under fixed confidence scaling

Source: `review/convex/moments.md`, CM43. Scale one on observed inputs preserves
the original normalized CE and all coefficient derivatives. Record IDs keep
repeated observations and the actual denominator of a short minibatch tail.
The same feature transformation applies to zero-update forward-only heads.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale

variable {K X C S : Type*} [Fintype K] [Fintype C]

/-- Ordinary CE in every head coefficient is convex with a frozen scale,
CM43. This does not assert joint convexity in a trainable scale and tensor. -/
theorem batchLoss_convex (scale : X → ℝ) (source : K → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (R : ℝ) :
    ConvexOn ℝ (headBudget R)
      (Fallback.batchLoss (features scale source) inputs labels batch) := by
  have h := headLoss_convex (features scale source)
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

/-- Scale one at every original record leaves its actual logits unchanged,
so its complete batch CE agrees for every tensor, CM43. -/
theorem batch_loss_eq (scale : X → ℝ) (source : K → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hscale : ∀ s ∈ batch, scale (inputs s) = 1) (W : K → C → ℝ) :
    Fallback.batchLoss (features scale source) inputs labels batch W =
      Fallback.batchLoss source inputs labels batch W := by
  unfold Fallback.batchLoss headLoss
  congr 1
  apply Finset.sum_congr rfl
  intro s _
  rw [logits, hscale s.val s.property, one_smul]

/-- Full coefficient derivatives are unchanged on observed batches, CM43.
The usual `fderiv` convention applies away from differentiability points. -/
theorem batch_fderiv_eq (scale : X → ℝ) (source : K → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hscale : ∀ s ∈ batch, scale (inputs s) = 1) (W : K → C → ℝ) :
    fderiv ℝ (Fallback.batchLoss (features scale source) inputs labels batch) W =
      fderiv ℝ (Fallback.batchLoss source inputs labels batch) W := by
  have heq : Fallback.batchLoss (features scale source) inputs labels batch =
      Fallback.batchLoss source inputs labels batch := by
    funext V
    exact batch_loss_eq scale source inputs labels batch hscale V
  rw [heq]

/-- Actual derivative existence transfers from the original objective,
CM43. No derivative of a training-data selection procedure is assumed. -/
theorem batch_hasFDerivAt (scale : X → ℝ) (source : K → X → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hscale : ∀ s ∈ batch, scale (inputs s) = 1) (W : K → C → ℝ)
    (D : (K → C → ℝ) →L[ℝ] ℝ)
    (hd : HasFDerivAt (Fallback.batchLoss source inputs labels batch) D W) :
    HasFDerivAt (Fallback.batchLoss (features scale source) inputs labels batch) D W := by
  have heq : Fallback.batchLoss (features scale source) inputs labels batch =
      Fallback.batchLoss source inputs labels batch := by
    funext V
    exact batch_loss_eq scale source inputs labels batch hscale V
  rw [heq]
  exact hd

/-- Composing CM42's exact observed context rows with CM43's observed scale
one preserves the original source-state batch, including repeated IDs. -/
theorem context_batch_loss_eq {I : Type*} [Fintype I] [DecidableEq I]
    (w : X → I → ℝ) (source : K → I → ℝ) (scale : X → ℝ) (state : X → I)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hrow : ∀ s ∈ batch, ∀ i, w (inputs s) i = if i = state (inputs s) then 1 else 0)
    (hscale : ∀ s ∈ batch, scale (inputs s) = 1) (W : K → C → ℝ) :
    Fallback.batchLoss (features scale (ContextInterpolation.features w source))
        inputs labels batch W =
      Fallback.batchLoss source (fun s => state (inputs s)) labels batch W := by
  rw [batch_loss_eq scale _ inputs labels batch hscale]
  exact ContextInterpolation.batch_loss_eq w source state inputs labels batch hrow W

/-- A concrete fixed scale leaves observed inputs 0 and 1 intact and halves
the remaining context features, CM43. -/
noncomputable def observedScale (x : Fin 4) : ℝ := if x.val < 2 then 1 else 1 / 2

/-- The repeated original observations satisfy the exact-scale premise,
CM43. IDs 0 and 1 repeat input 0, while ID 2 has input 1. -/
theorem observedScale_at (s : Fin 3) :
    observedScale (ContextInterpolation.batchInputs s) = 1 := by
  fin_cases s <;> norm_num [observedScale, ContextInterpolation.batchInputs]

example := batchLoss_convex observedScale (complexFeatures fourier4)
  ContextInterpolation.batchInputs ContextInterpolation.batchInputs
  (Finset.univ : Finset (Fin 3)) 64
example := batch_loss_eq observedScale (complexFeatures fourier4)
  ContextInterpolation.batchInputs ContextInterpolation.batchInputs
  (Finset.univ : Finset (Fin 3)) (fun s _ => observedScale_at s)
  (complexCoefficients (complexInitializer fourier4 id 4))
-- The denominator of the final one-record minibatch stays one.
example := batch_loss_eq observedScale (complexFeatures fourier4)
  ContextInterpolation.batchInputs ContextInterpolation.batchInputs
  ({2} : Finset (Fin 3)) (fun s _ => observedScale_at s)
  (complexCoefficients (complexInitializer fourier4 id 4))
example := batch_fderiv_eq observedScale (complexFeatures fourier4)
  ContextInterpolation.batchInputs ContextInterpolation.batchInputs
  (Finset.univ : Finset (Fin 3)) (fun s _ => observedScale_at s)
  (complexCoefficients (complexInitializer fourier4 id 4))
example := context_batch_loss_eq ContextInterpolation.batchWeights (complexFeatures fourier4)
  observedScale id ContextInterpolation.batchInputs ContextInterpolation.batchInputs
  (Finset.univ : Finset (Fin 3))
  (fun s _ => ContextInterpolation.batchWeights_observed s) (fun s _ => observedScale_at s)
  (complexCoefficients (complexInitializer fourier4 id 4))
example := context_batch_loss_eq ContextInterpolation.batchWeights (complexFeatures fourier4)
  observedScale id ContextInterpolation.batchInputs ContextInterpolation.batchInputs
  ({2} : Finset (Fin 3))
  (fun s _ => ContextInterpolation.batchWeights_observed s) (fun s _ => observedScale_at s)
  (complexCoefficients (complexInitializer fourier4 id 4))
-- Derivative transfer has an inhabited nonempty batch. Nonzero complex
-- features and a nonzero head are used by the loss and gradient identities above.
example := batch_hasFDerivAt observedScale (fun (_ : Fin 2) (_ : Fin 4) => 0)
  ContextInterpolation.batchInputs ContextInterpolation.batchInputs
  (Finset.univ : Finset (Fin 3)) (fun s _ => observedScale_at s)
  (fun _ _ => 1) 0 (by
    have hz (V : Fin 2 → Fin 4 → ℝ) (x : Fin 4) :
        headLogits (fun (_ : Fin 2) (_ : Fin 4) => 0) V x = (fun _ : Fin 4 => 0) := by
      funext c
      simp [headLogits]
    have heq : Fallback.batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
        ContextInterpolation.batchInputs ContextInterpolation.batchInputs
        (Finset.univ : Finset (Fin 3)) = fun _ : Fin 2 → Fin 4 → ℝ =>
          Fallback.batchLoss (fun (_ : Fin 2) (_ : Fin 4) => 0)
            ContextInterpolation.batchInputs ContextInterpolation.batchInputs Finset.univ 0 := by
      funext V
      simp only [Fallback.batchLoss, headLoss, hz]
    rw [heq]
    exact hasFDerivAt_const _ _)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextScale
