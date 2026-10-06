import Transformer.GPTMini.Convex.InteractionObservations
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.AffineSpace.AffineMap

/-!
# A dimension obstruction to exact affine interaction compression

Derived from arXiv:2211.11052v1, §3's attention-only product, after the
signed unnormalized predecessor-key replacement at f9749c7. The coverage
hypothesis concerns every width-one physical linear head on all contexts
of lengths two and three. It is not a finite-data fitting hypothesis.

An affine family that includes every such head must have at least V³D
real parameters: those heads span the tensor space, and that space is
recoverable from actual short-context outputs. Thus changing how the
complete linear class is stored cannot by itself give compactness.
The result does not rule out a nonlinear family, a restricted function
class, normalized softmax, or convexity for one specific loss.
-/

noncomputable section

namespace Transformer.GPTMini.Convex

open scoped BigOperators

/-- The independent interaction coordinates have dimension V³D.
Source: the complete tensor in `JointInteraction`, counted as a real vector space. -/
theorem interactionTensor_finrank (V D : ℕ) :
    Module.finrank ℝ (InteractionTensor V D) = V ^ 3 * D := by
  simp only [InteractionTensor, Module.finrank_pi_fintype, Module.finrank_self,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id]
  ring

/-- Covering every original width-one head forces a linear decoder to cover every tensor.
Source: the physical selector-head recovery in `JointInteractionFactorization`;
the decoder may use any parameter vector space and need not store tensor entries. -/
theorem headCover_linear_surjective {V D : ℕ} {E : Type*}
    [AddCommGroup E] [Module ℝ E] (L : E →ₗ[ℝ] InteractionTensor V D)
    (hcover : ∀ h : LinearTokenHead V 1 D, ∃ p : E, L p = headInteractionTensor h) :
    Function.Surjective L := by
  classical
  intro C
  choose p hp using fun h : Fin V × Fin V => hcover (interactionBasisHead C h)
  refine ⟨∑ h, p h, ?_⟩
  rw [map_sum]
  simp_rw [hp]
  funext q k v c
  simp only [Finset.sum_apply]
  exact interactionTensor_recovery C q k v c

example (V D : ℕ) : ∀ h : LinearTokenHead V 1 D,
    ∃ p : InteractionTensor V D,
      (LinearMap.id : InteractionTensor V D →ₗ[ℝ] InteractionTensor V D) p =
        headInteractionTensor h := by
  intro h
  exact ⟨headInteractionTensor h, rfl⟩

/-- Exact linear coverage of the physical head class needs at least V³D parameters.
Source: the spanning argument above and finite-dimensional surjection monotonicity.
This is a class-coverage bound, not a lower bound for one dataset or task. -/
theorem headCover_linear_finrank {V D : ℕ} {E : Type*}
    [AddCommGroup E] [Module ℝ E] [Module.Finite ℝ E]
    (L : E →ₗ[ℝ] InteractionTensor V D)
    (hcover : ∀ h : LinearTokenHead V 1 D, ∃ p : E, L p = headInteractionTensor h) :
    V ^ 3 * D ≤ Module.finrank ℝ E := by
  rw [← interactionTensor_finrank]
  exact LinearMap.finrank_le_finrank_of_surjective (headCover_linear_surjective L hcover)

example (V D : ℕ) : ∀ h : LinearTokenHead V 1 D,
    ∃ p : InteractionTensor V D,
      (LinearMap.id : InteractionTensor V D →ₗ[ℝ] InteractionTensor V D) p =
        headInteractionTensor h := by
  intro h
  exact ⟨headInteractionTensor h, rfl⟩

/-- The bound applies to actual outputs, even if the decoder has no tensor representation.
Source: `InteractionObservations` recovers the coefficients from true final-row
outputs. Coverage includes all short contexts jointly, not separate fits per context. -/
theorem headCover_observations_finrank {V D : ℕ} {E : Type*}
    [AddCommGroup E] [Module ℝ E] [Module.Finite ℝ E]
    (F : E →ₗ[ℝ] InteractionObservations V D)
    (hcover : ∀ h : LinearTokenHead V 1 D,
      ∃ p : E, F p = tensorObservationsMap V D (headInteractionTensor h)) :
    V ^ 3 * D ≤ Module.finrank ℝ E := by
  apply headCover_linear_finrank ((recoverInteractionMap V D).comp F)
  intro h
  obtain ⟨p, hp⟩ := hcover h
  refine ⟨p, ?_⟩
  rw [LinearMap.comp_apply, hp, recoverInteraction_tensorObservations]

example (V D : ℕ) : ∀ h : LinearTokenHead V 1 D,
    ∃ p : InteractionTensor V D,
      tensorObservationsMap V D p = tensorObservationsMap V D (headInteractionTensor h) := by
  intro h
  exact ⟨headInteractionTensor h, rfl⟩

/-- A zero physical head has zero coefficients, giving an actual zero-output anchor.
Source: §3's value product specialized to the zero head, with the declared linear operator. -/
theorem headInteractionTensor_zero (V W D : ℕ) :
    headInteractionTensor (0 : LinearTokenHead V W D) = 0 := by
  funext q k v c
  unfold headInteractionTensor
  change (∑ d : Fin W, (0 : ℝ) * 0) * 0 = 0
  exact mul_zero _

/-- Allowing an affine offset does not remove the exact-coverage dimension bound.
Source: the new short-context coverage result. Subtracting the covered zero
head converts the affine predictions to a linear family with the same head coverage. -/
theorem headCover_affine_finrank {V D : ℕ} {E : Type*}
    [AddCommGroup E] [Module ℝ E] [Module.Finite ℝ E]
    (F : E →ᵃ[ℝ] InteractionObservations V D)
    (hcover : ∀ h : LinearTokenHead V 1 D,
      ∃ p : E, F p = tensorObservationsMap V D (headInteractionTensor h)) :
    V ^ 3 * D ≤ Module.finrank ℝ E := by
  obtain ⟨p₀, hp₀⟩ := hcover 0
  have hzero : F p₀ = 0 := by
    simpa only [headInteractionTensor_zero, LinearMap.map_zero] using hp₀
  apply headCover_observations_finrank F.linear
  intro h
  obtain ⟨p, hp⟩ := hcover h
  refine ⟨p - p₀, ?_⟩
  simpa only [vsub_eq_sub, hp, hzero, sub_zero] using F.linearMap_vsub p p₀

example (V D : ℕ) : ∀ h : LinearTokenHead V 1 D,
    ∃ p : InteractionTensor V D,
      (tensorObservationsMap V D).toAffineMap p =
        tensorObservationsMap V D (headInteractionTensor h) := by
  intro h
  exact ⟨headInteractionTensor h, rfl⟩

/-- The free residual embedding contributes a further VD identifiable coordinates.
Source: `InteractionObservations` also reconstructs the independent residual
table. The count concerns the complete jointly linear operator. -/
theorem jointInteraction_finrank (V D : ℕ) :
    Module.finrank ℝ (JointInteraction V D) = (V + V ^ 3) * D := by
  change Module.finrank ℝ ((Fin V → Fin D → ℝ) × InteractionTensor V D) = _
  rw [Module.finrank_prod, interactionTensor_finrank]
  simp only [Module.finrank_pi_fintype, Module.finrank_self,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id]
  ring

/-- Basis's full tensor dimension is the number appearing in the conditional class bound.
Source: Basis's 256 keys, 256 values and 36 reserved IDs at cbafbe9.
It is not a universal lower bound for compact nonlinear GPTMini replacements. -/
theorem basis_interactionTensor_finrank :
    Module.finrank ℝ (InteractionTensor 548 64) = 10532261888 := by
  rw [interactionTensor_finrank]
  norm_num

end Transformer.GPTMini.Convex
