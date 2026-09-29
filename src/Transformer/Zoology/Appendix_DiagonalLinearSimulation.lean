/-
# Exact cancellation of Coyote's quadratic terms

Arora et al., arXiv:2312.04927v1, Appendix `lmm:primitives` and
`prop: butterfly-hyena`. A three-layer construction computes
a ⊙ u + b ⊙ (uW), with arbitrary position-dependent a and b and a shared W.
The intermediate quadratic terms cancel identically over the reals.
-/

import Transformer.Zoology.Appendix_DiagonalLinearLayout

open scoped BigOperators

namespace Transformer.Zoology

/-- The linear map needed for feature-axis butterfly factors.
Source: Appendix `eq: butterfly-split`, diagonal and off-diagonal terms. -/
def diagonalLinearApply {n d : ℕ} (W : Fin d → Fin d → ℝ)
    (a b u : RealSequence n d) : RealSequence n d :=
  fun i q => a i q * u i q + b i q * linearProjection u W i q

/-- Produce the three terms in the cancellation identity.
Source: Appendix `lmm:primitives`, shared projection and identity filter. -/
def diagonalLinearBranchParameters {n d : ℕ} [NeZero n]
    (W : Fin d → Fin d → ℝ) (a b : RealSequence n d) :
    CoyoteParameters (9 * n) d := {
  weight := W
  filter := impulse 0
  bias₁ := diagonalLinearBias₁ a b
  bias₂ := diagonalLinearBias₂ b
}

/-- The shared projection reads the same input at both copies and zero
at the third buffer. Source: Appendix `lmm:primitives`, buffer construction. -/
theorem diagonalLinearProjection_slot {n d : ℕ} [NeZero n]
    (W : Fin d → Fin d → ℝ) (u : RealSequence n d)
    (r : Fin 3) (i : Fin n) (q : Fin d) :
    linearProjection (shiftSumCopyState u) W (diagonalLinearSlot r i) q =
      if r = 2 then 0 else linearProjection u W i q := by
  fin_cases r <;> simp [linearProjection, diagonalLinearCopy_slot]

/-- The branch buffers hold (Wu+a)(u+b), (Wu)u, and ab respectively.
Source: Appendix `prop: butterfly-hyena`, corrected cancellation construction. -/
theorem diagonalLinearBranch_slot {n d : ℕ} [NeZero n]
    (W : Fin d → Fin d → ℝ) (a b u : RealSequence n d)
    (r : Fin 3) (i : Fin n) (q : Fin d) :
    coyoteLayerCyclic (diagonalLinearBranchParameters W a b)
      (shiftSumCopyState u) (diagonalLinearSlot r i) q =
      if r = 0 then (linearProjection u W i q + a i q) * (u i q + b i q)
      else if r = 1 then linearProjection u W i q * u i q
      else a i q * b i q := by
  simp only [coyoteLayerCyclic, diagonalLinearBranchParameters,
    cyclicConvolution_impulse, sub_zero, diagonalLinearProjection_slot,
    diagonalLinearBias₁_slot, diagonalLinearBias₂_slot, diagonalLinearCopy_slot]
  fin_cases r <;> simp

/-- The signed sum keeps the first branch and subtracts the other two.
Source: Appendix `prop: prim-add`, corrected cancellation construction. -/
def diagonalLinearSign (r : Fin 3) : ℝ := if r = 0 then 1 else -1

/-- Collect the signed buffer sum at the first n rows and clear the rest.
Source: Appendix `prop: prim-add` and `lem: stacking-layers`. -/
def diagonalLinearCollectParameters {n d : ℕ} [NeZero n] :
    CoyoteParameters (9 * n) d := {
  weight := fun _ _ => 0
  filter := fun k _ => ∑ r : Fin 3,
    if k = diagonalLinearTap n r then diagonalLinearSign r else 0
  bias₁ := fun j _ => if j.val < n then 1 else 0
  bias₂ := fun _ _ => 0
}

/-- The collecting layer reads the prescribed signed sum.
Source: Appendix `prop: prim-add`, cancellation buffers. -/
theorem diagonalLinearCollect_input {n d : ℕ} [NeZero n]
    (v : RealSequence (9 * n) d) (i : Fin n) (q : Fin d) :
    coyoteLayerCyclic diagonalLinearCollectParameters v (shiftSumInputSlot i) q =
      ∑ r : Fin 3, diagonalLinearSign r * v (diagonalLinearSlot r i) q := by
  simp only [coyoteLayerCyclic, diagonalLinearCollectParameters,
    linearProjection, mul_zero, Finset.sum_const_zero, zero_add, add_zero]
  simp only [shiftSumInputSlot, i.isLt, ite_true, one_mul]
  rw [cyclicConvolution_weightedThreeImpulses]
  apply Finset.sum_congr rfl
  intro r _
  rw [show (⟨i.val, by omega⟩ : Fin (9 * n)) = shiftSumInputSlot i from rfl,
    diagonalLinearTap_read]

/-- Three layers compute the diagonal-plus-projection map.
Source: Appendix `prop: butterfly-hyena`, corrected feature-axis construction. -/
def diagonalLinearNetwork {n d : ℕ} [NeZero n]
    (W : Fin d → Fin d → ℝ) (a b : RealSequence n d) :
    CyclicCoyoteNetwork (9 * n) d := {
  layers := [shiftSumCopyParameters, diagonalLinearBranchParameters W a b,
    diagonalLinearCollectParameters]
}

/-- The quadratic branch terms cancel exactly; the resulting workspace
is again zero-padded. Source: Appendix `prop: butterfly-hyena` and
`lem: stacking-layers`, constructive correction of the omitted-shift proof. -/
theorem diagonalLinearNetwork_correct {n d : ℕ} [NeZero n]
    (W : Fin d → Fin d → ℝ) (a b u : RealSequence n d) :
    (diagonalLinearNetwork W a b).run (shiftSumPad u) =
      shiftSumPad (diagonalLinearApply W a b u) := by
  change coyoteLayerCyclic diagonalLinearCollectParameters
    (coyoteLayerCyclic (diagonalLinearBranchParameters W a b)
      (coyoteLayerCyclic shiftSumCopyParameters (shiftSumPad u))) = _
  rw [shiftSumCopyParameters_apply]
  funext j q
  by_cases hj : j.val < n
  · let i : Fin n := ⟨j.val, hj⟩
    have hslot : shiftSumInputSlot i = j := Fin.ext rfl
    rw [← hslot, diagonalLinearCollect_input]
    simp only [diagonalLinearBranch_slot]
    simp [Fin.sum_univ_three, diagonalLinearSign, shiftSumPad,
      shiftSumInputSlot, diagonalLinearApply]
    ring
  · simp [coyoteLayerCyclic, diagonalLinearCollectParameters,
      linearProjection, shiftSumPad, hj]

end Transformer.Zoology
