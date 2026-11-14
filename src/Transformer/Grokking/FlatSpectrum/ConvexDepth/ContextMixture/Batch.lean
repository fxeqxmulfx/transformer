module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture.Loss

/-!
# Original records and convex mixture diagnostics

Source: `review/flat_spectrum.md`, CM45. Ordinary diagnostic CE is convex in
the mixture and retains original record multiplicities and tail denominators.
When every bank row preserves each observation, the actual observed objective
is constant in the mixture and cannot supply a preference between contexts.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {J K X C S : Type*} [Fintype J] [Fintype K] [Fintype C]

/-- Original-record mean CE is convex in all mixture coefficients, CM45.
The bank can be a fixed training-only diagnostic; the head is not trained here. -/
theorem batchLoss_convex (bank : J → K → X → ℝ) (W : K → C → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) :
    ConvexOn ℝ simplex
      (fun a => Fallback.batchLoss (features bank a) inputs labels batch W) := by
  refine ⟨simplex_convex, ?_⟩
  intro a ha b hb t u ht hu htu
  have hsum : headLoss (features bank (t • a + u • b))
      (fun s : ↥batch => inputs s.val) (fun s : ↥batch => labels s.val) W ≤
      t * headLoss (features bank a) (fun s : ↥batch => inputs s.val)
        (fun s : ↥batch => labels s.val) W +
      u * headLoss (features bank b) (fun s : ↥batch => inputs s.val)
        (fun s : ↥batch => labels s.val) W := by
    unfold headLoss
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_le_sum
    intro s _
    exact (ce_convex bank W (inputs s.val) (labels s.val)).2 ha hb ht hu htu
  have hm := mul_le_mul_of_nonneg_left hsum
    (inv_nonneg.mpr (Nat.cast_nonneg batch.card) : 0 ≤ (batch.card : ℝ)⁻¹)
  unfold Fallback.batchLoss
  calc
    _ ≤ (batch.card : ℝ)⁻¹ * (t * headLoss _ _ _ W + u * headLoss _ _ _ W) := hm
    _ = _ := by ring

/-- The mean coefficient slope uses all original record IDs, CM45.
Repeated inputs still contribute separately; a tail uses its own cardinality. -/
noncomputable def batchSlope (bank : J → K → X → ℝ) (W : K → C → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (a : J → ℝ) (j : J) : ℝ :=
  (batch.card : ℝ)⁻¹ * ∑ s : ↥batch, slope bank W a (inputs s.val) (labels s.val) j

/-- The explicit mean slopes support the complete original-record CE,
CM45. This applies equally to full, repeated-input and short final batches. -/
theorem batch_first_order (bank : J → K → X → ℝ) (W : K → C → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (a b : J → ℝ) :
    Fallback.batchLoss (features bank a) inputs labels batch W +
        ∑ j, (b j - a j) * batchSlope bank W inputs labels batch a j ≤
      Fallback.batchLoss (features bank b) inputs labels batch W := by
  have heq : (∑ j, (b j - a j) * batchSlope bank W inputs labels batch a j) =
      (batch.card : ℝ)⁻¹ * ∑ s : ↥batch,
        ∑ j, (b j - a j) * slope bank W a (inputs s.val) (labels s.val) j := by
    simp only [batchSlope, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro s _
    apply Finset.sum_congr rfl
    intro j _
    ring
  rw [heq]
  unfold Fallback.batchLoss headLoss
  rw [← mul_add, ← Finset.sum_add_distrib]
  apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr (Nat.cast_nonneg batch.card))
  apply Finset.sum_le_sum
  intro s _
  exact first_order bank W a b (inputs s.val) (labels s.val)

/-- All observed rows agree with the source, so any mass-one mixture has
the same original mean CE for every complete head tensor, CM45. -/
theorem batch_loss_eq (bank : J → K → X → ℝ) (source : K → X → ℝ) (a : J → ℝ)
    (hs : ∑ j, a j = 1) (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hrow : ∀ s ∈ batch, ∀ j k, bank j k (inputs s) = source k (inputs s))
    (W : K → C → ℝ) :
    Fallback.batchLoss (features bank a) inputs labels batch W =
      Fallback.batchLoss source inputs labels batch W := by
  unfold Fallback.batchLoss headLoss
  congr 1
  apply Finset.sum_congr rfl
  intro s _
  have heq : headLogits (features bank a) W (inputs s.val) =
      headLogits source W (inputs s.val) := by
    funext c
    unfold headLogits
    apply Finset.sum_congr rfl
    intro k _
    rw [features_at bank source a hs (inputs s.val) (hrow s.val s.property)]
  rw [heq]

/-- Exact observed-row preservation cannot select a mixture through the
original observed CE, CM45. A separate training-only diagnostic is necessary. -/
theorem observed_loss_constant (bank : J → K → X → ℝ) (source : K → X → ℝ)
    {a b : J → ℝ} (ha : a ∈ simplex) (hb : b ∈ simplex)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hrow : ∀ s ∈ batch, ∀ j k, bank j k (inputs s) = source k (inputs s))
    (W : K → C → ℝ) :
    Fallback.batchLoss (features bank a) inputs labels batch W =
      Fallback.batchLoss (features bank b) inputs labels batch W := by
  rw [batch_loss_eq bank source a ha.2 inputs labels batch hrow,
    batch_loss_eq bank source b hb.2 inputs labels batch hrow]

/-- The original complete-head Fréchet derivatives also agree, CM45.
This does not claim a derivative of a discrete bank-discovery procedure. -/
theorem batch_fderiv_eq (bank : J → K → X → ℝ) (source : K → X → ℝ) (a : J → ℝ)
    (hs : ∑ j, a j = 1) (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hrow : ∀ s ∈ batch, ∀ j k, bank j k (inputs s) = source k (inputs s))
    (W : K → C → ℝ) :
    fderiv ℝ (Fallback.batchLoss (features bank a) inputs labels batch) W =
      fderiv ℝ (Fallback.batchLoss source inputs labels batch) W := by
  have heq : Fallback.batchLoss (features bank a) inputs labels batch =
      Fallback.batchLoss source inputs labels batch := by
    funext V
    exact batch_loss_eq bank source a hs inputs labels batch hrow V
  rw [heq]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ContextMixture
