module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.GatedContext.Loss

/-!
# Original batches for anchored local contexts

Source: `review/flat_spectrum.md`, CM46. Original record multiplicities and
tail denominators remain explicit. Fixed gates permit a convex diagnostic;
observed-row preservation still leaves the original observed loss unchanged.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.GatedContext

open scoped BigOperators

variable {G J K X C S : Type*} [Fintype G] [Fintype J] [Fintype K] [Fintype C]

/-- Mean ordinary CE of original records is convex in all local rows,
CM46. Repeated inputs retain their separate record identities. -/
theorem batchLoss_convex (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) :
    ConvexOn ℝ rows
      (fun a => Fallback.batchLoss (features gate bank base radius a) inputs labels batch W) := by
  refine ⟨rows_convex, ?_⟩
  intro a ha b hb t u ht hu htu
  have hsum : headLoss (features gate bank base radius (t • a + u • b))
      (fun s : ↥batch => inputs s.val) (fun s : ↥batch => labels s.val) W ≤
      t * headLoss (features gate bank base radius a) (fun s : ↥batch => inputs s.val)
        (fun s : ↥batch => labels s.val) W +
      u * headLoss (features gate bank base radius b) (fun s : ↥batch => inputs s.val)
        (fun s : ↥batch => labels s.val) W := by
    unfold headLoss
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_le_sum
    intro s _
    exact (ce_convex gate bank base radius W (inputs s.val) (labels s.val)).2 ha hb ht hu htu
  have hm := mul_le_mul_of_nonneg_left hsum
    (inv_nonneg.mpr (Nat.cast_nonneg batch.card) : 0 ≤ (batch.card : ℝ)⁻¹)
  unfold Fallback.batchLoss
  calc
    _ ≤ (batch.card : ℝ)⁻¹ * (t * headLoss _ _ _ W + u * headLoss _ _ _ W) := hm
    _ = _ := by ring

/-- Averaging the explicit local supporting terms uses the actual batch
cardinality, CM46, including repeated inputs and short final batches. -/
theorem batch_first_order (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (base : J → ℝ) (radius : ℝ) (W : K → C → ℝ)
    (inputs : S → X) (labels : S → C) (batch : Finset S) (a b : G → J → ℝ) :
    Fallback.batchLoss (features gate bank base radius a) inputs labels batch W +
        (batch.card : ℝ)⁻¹ * ∑ s : ↥batch, ∑ g, ∑ j,
          (b g j - a g j) * slope gate bank base radius W a (inputs s.val) (labels s.val) g j ≤
      Fallback.batchLoss (features gate bank base radius b) inputs labels batch W := by
  unfold Fallback.batchLoss headLoss
  rw [← mul_add, ← Finset.sum_add_distrib]
  apply mul_le_mul_of_nonneg_left _ (inv_nonneg.mpr (Nat.cast_nonneg batch.card))
  apply Finset.sum_le_sum
  intro s _
  exact first_order gate bank base radius W a b (inputs s.val) (labels s.val)

/-- Mass-one gates and rows preserve every original observed loss for
every complete head, CM46. Positivity and a radius constraint are unnecessary. -/
theorem batch_loss_eq (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (source : K → X → ℝ) (base : J → ℝ) (radius : ℝ) (a : G → J → ℝ)
    (hbase : ∑ j, base j = 1) (ha : ∀ g, ∑ j, a g j = 1)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hgate : ∀ s ∈ batch, ∑ g, gate (inputs s) g = 1)
    (hrow : ∀ s ∈ batch, ∀ j k, bank j k (inputs s) = source k (inputs s))
    (W : K → C → ℝ) :
    Fallback.batchLoss (features gate bank base radius a) inputs labels batch W =
      Fallback.batchLoss source inputs labels batch W := by
  unfold Fallback.batchLoss headLoss
  congr 1
  apply Finset.sum_congr rfl
  intro s _
  have heq : headLogits (features gate bank base radius a) W (inputs s.val) =
      headLogits source W (inputs s.val) := by
    funext c
    unfold headLogits
    apply Finset.sum_congr rfl
    intro k _
    rw [features_at gate bank source base radius a (inputs s.val)
      hbase (hgate s.val s.property) ha (hrow s.val s.property)]
  rw [heq]

/-- The complete-head Frechet derivatives agree on the original observed
batches, CM46. This supplies no derivative of gate discovery. -/
theorem batch_fderiv_eq (gate : X → G → ℝ) (bank : J → K → X → ℝ)
    (source : K → X → ℝ) (base : J → ℝ) (radius : ℝ) (a : G → J → ℝ)
    (hbase : ∑ j, base j = 1) (ha : ∀ g, ∑ j, a g j = 1)
    (inputs : S → X) (labels : S → C) (batch : Finset S)
    (hgate : ∀ s ∈ batch, ∑ g, gate (inputs s) g = 1)
    (hrow : ∀ s ∈ batch, ∀ j k, bank j k (inputs s) = source k (inputs s))
    (W : K → C → ℝ) :
    fderiv ℝ (Fallback.batchLoss (features gate bank base radius a) inputs labels batch) W =
      fderiv ℝ (Fallback.batchLoss source inputs labels batch) W := by
  have heq : Fallback.batchLoss (features gate bank base radius a) inputs labels batch =
      Fallback.batchLoss source inputs labels batch := by
    funext V
    exact batch_loss_eq gate bank source base radius a hbase ha inputs labels batch hgate hrow V
  rw [heq]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.GatedContext
