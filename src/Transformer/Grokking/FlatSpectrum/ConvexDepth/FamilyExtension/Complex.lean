module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension.Basic

/-!
# Complete complex outputs and ordinary CE under fixed-family extension

Source: `review/flat_spectrum.md`, CM67. The embedding preserves every actual
complex logit and each original indexed batch loss. It embeds an old feasible
model at the same capacity bound, without identifying any new target labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension

open scoped BigOperators
open LatentPrograms (Parameters features head)

variable {J K X S C : Type*} [Fintype J] [Fintype K] [Fintype S] [DecidableEq S]

/-- Actual complete complex products preserve every raw logit at every
input when new blocks are zero, CM67. The statement does not use training
coverage or a target-realizability premise. -/
theorem logits_embed (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (old : J → X → S) (added : K → X → S) (q : Parameters J S C) (x : X) :
    headLogits (features U (programs old added)) (head U 1 (embed q)) x =
      headLogits (features U old) (head U 1 q) x := by
  funext c
  rw [LatentPrograms.logits U hU, LatentPrograms.logits U hU]
  simp [Fintype.sum_sum_type, programs, embed]

variable [Fintype C] [DecidableEq C]

omit [DecidableEq C] in
/-- Every original record keeps its ordinary full-vocabulary CE, CM67.
Repeated inputs keep their distinct record IDs and every batch its denominator. -/
theorem batchLoss_embed {I : Type*} (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (old : J → X → S) (added : K → X → S) (inputs : I → X) (labels : I → C)
    (batch : Finset I) (q : Parameters J S C) :
    BoundedLatent.batchLoss U (programs old added) inputs labels batch (embed q) =
      BoundedLatent.batchLoss U old inputs labels batch q := by
  unfold BoundedLatent.batchLoss
  simp_rw [logits_embed U hU]

/-- Every embedded old feasible head has the same full-column capacity
bound, CM67; both real and imaginary coefficients of every block are charged. -/
theorem head_budget_embed (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (old : J → X → S) (added : K → X → S) (oldOrder : List J) (addedOrder : List K)
    (samples : List (X × C)) {δ R : ℝ} {q : Parameters J S C}
    (hq : q ∈ BoundedLatent.domain old oldOrder samples δ R) :
    head U 1 (embed (K := K) q) ∈ headBudget (2 * R * Fintype.card S) := by
  exact BoundedLatent.head_budget U hU (programs old added) (order oldOrder addedOrder)
    samples (embed_mem old added oldOrder addedOrder samples hq)

/-- A larger full family can attain every old feasible batch CE value,
CM67. This is representation inclusion; it does not promise that a finite
numerical training policy selects the embedding or improves heldout accuracy. -/
theorem attainable_loss {I : Type*} (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (old : J → X → S) (added : K → X → S) (oldOrder : List J) (addedOrder : List K)
    (samples : List (X × C)) {δ R : ℝ} {q : Parameters J S C}
    (hq : q ∈ BoundedLatent.domain old oldOrder samples δ R)
    (inputs : I → X) (labels : I → C) (batch : Finset I) :
    ∃ r ∈ BoundedLatent.domain (programs old added) (order oldOrder addedOrder) samples δ R,
      BoundedLatent.batchLoss U (programs old added) inputs labels batch r =
        BoundedLatent.batchLoss U old inputs labels batch q := by
  exact ⟨embed q, embed_mem old added oldOrder addedOrder samples hq,
    batchLoss_embed U hU old added inputs labels batch q⟩

end Transformer.Grokking.FlatSpectrum.ConvexDepth.FamilyExtension
