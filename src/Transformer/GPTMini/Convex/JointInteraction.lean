import Mathlib.Analysis.Convex.Function
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Prod
import Mathlib.LinearAlgebra.Pi
import Mathlib.Tactic

/-!
# Unconstrained joint coefficients for an embedding/attention replacement

New route following the user's drop-in and unchanged-optimizer requirements,
2026-10-06. The reference is GPTMini's embedding and `Attention.forward` at
cbafbe9, and arXiv:2211.11052v1, §3's formula `eq:attention_only`.
Learn a residual embedding and a complete query/predecessor/value interaction
tensor. The forward is affine in their joint coordinates, on all of their
Euclidean parameter space. No PSD constraint, projection, or head search is
needed to define a feasible parameter update.

This is a candidate replacement of the combined token-to-stream prefix,
not an implemented replacement of each separately callable Python module.
It changes normalized softmax attention to a causal linear interaction sum;
it does not retain QKNorm, RoPE, XSA, or the old parameter regularizer.
All token interactions are trainable, but storage is cubic in vocabulary
size. Affinity does not survive arbitrary downstream FFNs or tied readouts.
-/

noncomputable section

namespace Transformer.GPTMini.Convex

open scoped BigOperators

/-- Joint coefficients of all query/key/value-token interactions.
Source: the new unfactorized route for the §3 attention-only formula. -/
abbrev InteractionTensor (V D : ℕ) := Fin V → Fin V → Fin V → Fin D → ℝ

/-- A trainable residual embedding together with the absorbed interaction coefficients.
Source: GPTMini's embedding/residual prefix at cbafbe9; the tensor absorbs
the learned embedding/projection/value products rather than multiplying them. -/
abbrev JointInteraction (V D : ℕ) := (Fin V → Fin D → ℝ) × InteractionTensor V D

/-- The predecessor convention of the repaired binding encoder, repeated at position zero.
Source: `pairedPrevious` at cbafbe9; no sparsemax operator is used here. -/
def interactionPrevious {T : ℕ} (j : Fin T) : Fin T :=
  ⟨j.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.isLt⟩

/-- The predecessor remains inside a visible causal prefix.
Source: the neighboring-token binding convention at cbafbe9. -/
theorem interactionPrevious_le {T : ℕ} (j : Fin T) : interactionPrevious j ≤ j :=
  Nat.sub_le _ _

/-- A token-to-stream prefix with jointly learned embedding and interaction coefficients.
Source: the new linear replacement of the §3 attention/value product.
The sum is signed and unnormalized; these are explicit model deviations. -/
def jointInteractionForward {V D T : ℕ} (tokens : Fin T → Fin V)
    (p : JointInteraction V D) : Fin T → Fin D → ℝ :=
  fun i c => p.1 (tokens i) c + ∑ j, if j ≤ i then
    p.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c else 0

/-- Joint addition is respected by the actual candidate forward, including its embedding.
Source: direct expansion of the new unfactorized causal operator. -/
theorem jointInteractionForward_add {V D T : ℕ} (tokens : Fin T → Fin V)
    (p q : JointInteraction V D) :
    jointInteractionForward tokens (p + q) =
      jointInteractionForward tokens p + jointInteractionForward tokens q := by
  funext i c
  change p.1 (tokens i) c + q.1 (tokens i) c + (∑ j, if j ≤ i then
    p.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c +
      q.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c else 0) = _
  have hs : (∑ j : Fin T, if j ≤ i then
      p.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c +
        q.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c else 0) =
      ∑ j : Fin T, ((if j ≤ i then p.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c else 0) +
        (if j ≤ i then q.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c else 0)) := by
    apply Finset.sum_congr rfl
    intro j hj
    by_cases h : j ≤ i <;> simp only [h, ite_true, ite_false, zero_add]
  rw [hs, Finset.sum_add_distrib]
  change _ = (p.1 (tokens i) c + _) + (q.1 (tokens i) c + _)
  ring

/-- Scaling every joint coefficient scales the genuine forward.
Source: direct expansion of the new unfactorized causal operator. -/
theorem jointInteractionForward_smul {V D T : ℕ} (tokens : Fin T → Fin V)
    (a : ℝ) (p : JointInteraction V D) :
    jointInteractionForward tokens (a • p) = a • jointInteractionForward tokens p := by
  funext i c
  change a * p.1 (tokens i) c + (∑ j, if j ≤ i then
    a * p.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c else 0) =
      a * (p.1 (tokens i) c + ∑ j, if j ≤ i then
        p.2 (tokens i) (tokens (interactionPrevious j)) (tokens j) c else 0)
  rw [mul_add, Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  by_cases h : j ≤ i <;> simp only [h, ite_true, ite_false, mul_zero]

/-- The entire jointly trained prefix is one linear map of free parameters.
Source: the new direct-coefficient replacement, not a tangent approximation. -/
def jointInteractionMap {V D T : ℕ} (tokens : Fin T → Fin V) :
    JointInteraction V D →ₗ[ℝ] (Fin T → Fin D → ℝ) where
  toFun := jointInteractionForward tokens
  map_add' := jointInteractionForward_add tokens
  map_smul' := jointInteractionForward_smul tokens

/-- Every convex output criterion remains convex in all joint coefficients.
Source: the new parameter-space guarantee for the §3 replacement. This
does not assert that a subsequent trainable FFN/readout is affine. -/
theorem jointInteraction_criterion_convex {V D T : ℕ} (tokens : Fin T → Fin V)
    (criterion : (Fin T → Fin D → ℝ) → ℝ) (hc : ConvexOn ℝ Set.univ criterion) :
    ConvexOn ℝ Set.univ (criterion ∘ jointInteractionForward tokens) := by
  refine ⟨convex_univ, ?_⟩
  intro p hp q hq a b ha hb hab
  change criterion (jointInteractionForward tokens (a • p + b • q)) ≤ _
  rw [jointInteractionForward_add, jointInteractionForward_smul, jointInteractionForward_smul]
  exact hc.2 (Set.mem_univ _) (Set.mem_univ _) ha hb hab

/-- A nonconstant linear criterion inhabits the output-convexity premise. -/
example : ConvexOn ℝ Set.univ ((fun y : Fin 2 → Fin 1 → ℝ => y 1 0) ∘
    jointInteractionForward (V := 2) (fun j : Fin 2 => j)) := by
  let L : (Fin 2 → Fin 1 → ℝ) →ₗ[ℝ] ℝ :=
    (LinearMap.proj (R := ℝ) (φ := fun _ : Fin 1 => ℝ) 0).comp
      (LinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => Fin 1 → ℝ) 1)
  exact jointInteraction_criterion_convex _ _ (L.convexOn convex_univ)

/-- Changing future tokens preserves the actual forward at an earlier row.
Source: GPTMini's causal masking and the predecessor convention at cbafbe9. -/
theorem jointInteractionForward_causal {V D T : ℕ} (tokens other : Fin T → Fin V)
    (p : JointInteraction V D) (row : Fin T) (c : Fin D)
    (ht : ∀ j, j ≤ row → tokens j = other j) :
    jointInteractionForward tokens p row c = jointInteractionForward other p row c := by
  unfold jointInteractionForward
  rw [ht row le_rfl]
  congr 1
  apply Finset.sum_congr rfl
  intro j hj
  by_cases h : j ≤ row
  · have hp : interactionPrevious j ≤ row := le_trans (interactionPrevious_le j) h
    simp only [h, ite_true, ht j h, ht (interactionPrevious j) hp]
  · simp only [h, ite_false]

/-- Distinct continuations satisfy the causal-prefix premise for arbitrary learned coefficients. -/
example (p : JointInteraction 2 1) :
    jointInteractionForward (fun _ : Fin 2 => 0) p 0 0 =
      jointInteractionForward (fun j : Fin 2 => if j = 0 then 0 else 1) p 0 0 := by
  apply jointInteractionForward_causal
  intro j hj
  have he : j = 0 := by omega
  subst j
  rfl

/-- The complete interaction tensor has V³D independent scalar coordinates.
Source: the new unconstrained route; no compactness is hidden in factor coordinates. -/
theorem interactionTensor_coordinate_count (V D : ℕ) :
    Fintype.card (Fin V × Fin V × Fin V × Fin D) = V ^ 3 * D := by
  simp only [Fintype.card_prod, Fintype.card_fin]
  ring

/-- Including the residual embedding adds VD scalar coordinates.
Source: the explicit joint parameter storage of the new candidate prefix. -/
theorem jointInteraction_coordinate_count (V D : ℕ) :
    Fintype.card (Fin V × Fin D) + Fintype.card (Fin V × Fin V × Fin V × Fin D) =
      (V + V ^ 3) * D := by
  rw [interactionTensor_coordinate_count]
  simp only [Fintype.card_prod, Fintype.card_fin]
  ring

end Transformer.GPTMini.Convex
