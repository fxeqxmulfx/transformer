module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning.Paired

/-!
# Ordinary mean CE and complete indexed minibatch updates

Source: `review/convex/moments.md`, CM82. A nonempty batch is indexed by
`Fin (N + 1)`: repeated records and short final batches enter exactly once per
index. Both analytic work and feasible-query derivatives are those of the
actual mean CE. These results do not assert optimizer convergence or accuracy.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning

open scoped BigOperators
open Renaming

variable {O J : Type*} [Fintype O] [Fintype J] [DecidableEq O] {T N : ℕ}

/-- Mean ordinary CE on every indexed nonempty raw batch, CM82. -/
noncomputable def meanBatchLoss (data : Fin (N + 1) → Prefix O T)
    (labels : Fin (N + 1) → O) (p : Parameters J O (ObjectWeights O T)) : ℝ :=
  ((N + 1 : ℕ) : ℝ)⁻¹ * ∑ i, sequenceLoss (data i) (labels i) p

/-- Every record supplies all raw coordinates before mean normalization, CM82. -/
noncomputable def meanBatchGradient (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    (p : Parameters J O (ObjectWeights O T)) : GradientArray J O T :=
  ((N + 1 : ℕ) : ℝ)⁻¹ • ∑ i,
    rawGradient (data i) (fun j k => mask j (data i, k)) (labels i) p

/-- The full minibatch array retains each indexed record's slope, CM82. -/
theorem meanBatchGradient_work (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    (p v : Parameters J O (ObjectWeights O T)) :
    rawWork (meanBatchGradient mask data labels p) v =
      ((N + 1 : ℕ) : ℝ)⁻¹ * ∑ i,
        rawWork (rawGradient (data i) (fun j k => mask j (data i, k)) (labels i) p) v := by
  rw [meanBatchGradient, rawWork_smul, rawWork_sum]

/-- The complete minibatch proposal supports the actual mean ordinary CE
throughout the unchanged full-prefix cone, CM82. -/
theorem meanBatchLoss_first_order (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    {p q : Parameters J O (ObjectWeights O T)}
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hq : q ∈ complexDomain rawInput Set.univ mask) :
    meanBatchLoss data labels p + rawWork (meanBatchGradient mask data labels p) (q - p) ≤
      meanBatchLoss data labels q := by
  rw [meanBatchGradient_work]
  simp only [meanBatchLoss, ← mul_add, ← Finset.sum_add_distrib]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Finset.sum_le_sum
  intro i hi
  exact sequenceLoss_first_order mask hp hq (data i) (labels i)

/-- Feasible full-array queries differentiate exactly this indexed mean
loss, CM82. No record is dropped to form a nominal batch size. -/
theorem meanBatchLoss_hasDerivAt (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    {p v : Parameters J O (ObjectWeights O T)} {radius : ℝ} (hr : 0 < radius)
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hplus : p + radius • v ∈ complexDomain rawInput Set.univ mask)
    (hminus : p + (-radius) • v ∈ complexDomain rawInput Set.univ mask) :
    HasDerivAt (fun amount : ℝ => meanBatchLoss data labels (p + amount • v))
      (rawWork (meanBatchGradient mask data labels p) v) 0 := by
  rw [meanBatchGradient_work]
  exact (HasDerivAt.fun_sum (fun i hi =>
    sequenceLoss_hasDerivAt mask hr hp hplus hminus (data i) (labels i))).const_mul _

/-- Mean CE of the actual shared prediction for every indexed record, CM82. -/
noncomputable def meanPairedBatchLoss (σ : Equiv.Perm O)
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    (p : Parameters J O (ObjectWeights O T)) : ℝ :=
  ((N + 1 : ℕ) : ℝ)⁻¹ * ∑ i, pairedLoss σ (data i) (labels i) p

/-- Both shared passes contribute for every indexed batch record, CM82. -/
noncomputable def meanPairedBatchGradient (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    (p : Parameters J O (ObjectWeights O T)) : GradientArray J O T :=
  ((N + 1 : ℕ) : ℝ)⁻¹ • ∑ i, pairedGradient σ mask (data i) (labels i) p

/-- Full shared minibatch work is the indexed mean of complete record work,
CM82. Repetition remains repetition, including in a short tail. -/
theorem meanPairedBatchGradient_work (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    (p v : Parameters J O (ObjectWeights O T)) :
    rawWork (meanPairedBatchGradient σ mask data labels p) v =
      ((N + 1 : ℕ) : ℝ)⁻¹ * ∑ i, rawWork (pairedGradient σ mask (data i) (labels i) p) v := by
  rw [meanPairedBatchGradient, rawWork_smul, rawWork_sum]

/-- The shared full-array minibatch proposal supports the actual paired
mean CE over the entire original parameter domain, CM82. -/
theorem meanPairedBatchLoss_first_order (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    {p q : Parameters J O (ObjectWeights O T)}
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hq : q ∈ complexDomain rawInput Set.univ mask) :
    meanPairedBatchLoss σ data labels p +
      rawWork (meanPairedBatchGradient σ mask data labels p) (q - p) ≤
        meanPairedBatchLoss σ data labels q := by
  rw [meanPairedBatchGradient_work]
  simp only [meanPairedBatchLoss, ← mul_add, ← Finset.sum_add_distrib]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  apply Finset.sum_le_sum
  intro i hi
  exact pairedLoss_first_order σ mask hp hq (data i) (labels i)

/-- The forward-only queries measure this actual shared minibatch loss,
whose derivative is the complete coordinate pairing, CM82. Finite-difference
error and optimizer convergence remain separate obligations. -/
theorem meanPairedBatchLoss_hasDerivAt (σ : Equiv.Perm O) (mask : J → Pattern (Prefix O T))
    (data : Fin (N + 1) → Prefix O T) (labels : Fin (N + 1) → O)
    {p v : Parameters J O (ObjectWeights O T)} {radius : ℝ} (hr : 0 < radius)
    (hp : p ∈ complexDomain rawInput Set.univ mask)
    (hplus : p + radius • v ∈ complexDomain rawInput Set.univ mask)
    (hminus : p + (-radius) • v ∈ complexDomain rawInput Set.univ mask) :
    HasDerivAt (fun amount : ℝ => meanPairedBatchLoss σ data labels (p + amount • v))
      (rawWork (meanPairedBatchGradient σ mask data labels p) v) 0 := by
  rw [meanPairedBatchGradient_work]
  exact (HasDerivAt.fun_sum (fun i hi =>
    pairedLoss_hasDerivAt σ mask hr hp hplus hminus (data i) (labels i))).const_mul _

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Learning
