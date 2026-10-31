module

public import Transformer.SequenceModel.Transfer.Basic

/-!
# Identical self spectra do not determine cross-split transfer

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, T4.
Negating an evaluation Jacobian preserves its complete self Gram operator,
not merely its eigenvalues. It reverses cross-split transfer. Singular
values alone therefore discard information needed to predict learning on
held-out examples. The full preconditioner stays in both identities.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Transfer

open scoped RealInnerProductSpace

variable {I A B : Type*} [Fintype I] [Fintype A] [Fintype B]

/-- T4: self Gram operators agree exactly after reversing all output tangents. -/
theorem self_cross_neg (J : Weights I →L[ℝ] Weights A) (P : Weights I →L[ℝ] Weights I) :
    cross (-J) (-J) P = cross J J P := by
  ext x
  simp [cross]

/-- T4: the same operation reverses all cross-split transfer predictions. -/
theorem cross_neg_eval (J : Weights I →L[ℝ] Weights A)
    (K : Weights I →L[ℝ] Weights B) (P : Weights I →L[ℝ] Weights I) :
    cross (-J) K P = -(cross J K P) := by
  ext x
  simp [cross]

/-- T4: positive and negative residual alignment coexist with identical
evaluation self Gram matrices; this is not captured by flatness or rank. -/
theorem opposite_transfer (J : Weights I →L[ℝ] Weights A)
    (K : Weights I →L[ℝ] Weights B) (P : Weights I →L[ℝ] Weights I)
    (u : Weights A) (r : Weights B) (h : 0 < ⟪u, cross J K P r⟫) :
    cross (-J) (-J) P = cross J J P ∧ ⟪u, cross (-J) K P r⟫ < 0 := by
  refine ⟨self_cross_neg J P, ?_⟩
  rw [cross_neg_eval]
  simpa using neg_neg_of_pos h

example := opposite_transfer (I := Fin 1) (A := Fin 1) (B := Fin 1)
  (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
  (WithLp.toLp 2 ![1]) (WithLp.toLp 2 ![1])
  (by norm_num [cross, ContinuousLinearMap.adjoint_id, EuclideanSpace.inner_eq_star_dotProduct,
    dotProduct, Fin.sum_univ_one])

end Transformer.SequenceModel.Transfer
