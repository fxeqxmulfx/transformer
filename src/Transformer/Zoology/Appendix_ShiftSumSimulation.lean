/-
# Three Coyote layers for a sum of shifted diagonal maps

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`.
This corrects the omitted-shift display constructively. Copies are stored
along the sequence axis, so all three layers retain the external feature
dimension and use zero projection weights.
-/

import Transformer.Zoology.Appendix_ShiftSumCopy

open scoped BigOperators

namespace Transformer.Zoology

/-- The sum of three position-dependent diagonal multipliers applied to
cyclic shifts. Source: Appendix `eq: butterfly-split`, including shifts. -/
def shiftedDiagonalSum {m d : ℕ}
    (coeff : Fin 3 → RealSequence m d) (shift : Fin 3 → Fin m)
    (u : RealSequence m d) : RealSequence m d :=
  fun i q => ∑ a : Fin 3, coeff a i q * u (i - shift a) q

/-- Convolve the two input copies into three buffers and apply each
branch's diagonal multiplier. Source: Appendix `prop: butterfly-hyena`,
corrected three-branch construction. -/
def shiftSumBranchParameters {m d : ℕ}
    (coeff : Fin 3 → RealSequence m d) (shift : Fin 3 → Fin m) :
    CoyoteParameters (9 * m) d := {
  weight := fun _ _ => 0
  filter := fun k _ => ∑ a : Fin 3,
    if k = shiftSumTermTap a (shift a) then 1 else 0
  bias₁ := shiftSumCoefficient coeff
  bias₂ := fun _ _ => 0
}

/-- Each branch buffer holds precisely its own weighted shifted input.
Source: Appendix `prop: butterfly-hyena`, corrected diagonal/shift stages. -/
theorem shiftSumBranchParameters_apply {m d : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m d) (shift : Fin 3 → Fin m)
    (u : RealSequence m d) (a : Fin 3) (i : Fin m) (q : Fin d) :
    coyoteLayerCyclic (shiftSumBranchParameters coeff shift)
      (shiftSumCopyState u) (shiftSumTermSlot a i) q =
        coeff a i q * u (i - shift a) q := by
  simp only [coyoteLayerCyclic, shiftSumBranchParameters,
    linearProjection, mul_zero, Finset.sum_const_zero, zero_add, add_zero]
  rw [shiftSumCoefficient_slot, cyclicConvolution_threeImpulses]
  congr 1
  rw [Finset.sum_eq_single a]
  · exact shiftSumCopyState_read u i (shift a) _
      (shiftSumTermTap_self_val a i (shift a)) q
  · intro b _ hb
    exact shiftSumCopyState_outside u _ q
      (shiftSumTermTap_other_val a b i (shift b) (Ne.symm hb))
  · intro h
    exact (h (Finset.mem_univ a)).elim

/-- Sum the three branch buffers at the first block and clear all
workspace rows. Source: Appendix `prop: prim-add`, block summation. -/
def shiftSumCollectParameters {m d : ℕ} [NeZero m] :
    CoyoteParameters (9 * m) d := {
  weight := fun _ _ => 0
  filter := fun k _ => ∑ a : Fin 3,
    if k = -shiftSumTermBase (m := m) a then 1 else 0
  bias₁ := fun j _ => if j.val < m then 1 else 0
  bias₂ := fun _ _ => 0
}

/-- At a first-block token, the collecting layer adds the corresponding
tokens of all three buffers. Source: Appendix `prop: prim-add`. -/
theorem shiftSumCollectParameters_input {m d : ℕ} [NeZero m]
    (state : RealSequence (9 * m) d) (i : Fin m) (q : Fin d) :
    coyoteLayerCyclic shiftSumCollectParameters state
      (shiftSumInputSlot i) q =
        ∑ a : Fin 3, state (shiftSumTermSlot a i) q := by
  simp only [coyoteLayerCyclic, shiftSumCollectParameters,
    linearProjection, mul_zero, Finset.sum_const_zero, zero_add, add_zero]
  simp only [shiftSumInputSlot, i.isLt, ite_true, one_mul]
  rw [cyclicConvolution_threeImpulses]
  apply Finset.sum_congr rfl
  intro a _
  rw [show (⟨i.val, by omega⟩ : Fin (9 * m)) = shiftSumInputSlot i from rfl,
    shiftSumTermSlot_read]

/-- All collecting-layer rows outside the output block are zero.
Source: Appendix `prop: prim-add`, workspace clearing. -/
theorem shiftSumCollectParameters_outside {m d : ℕ} [NeZero m]
    (state : RealSequence (9 * m) d) (j : Fin (9 * m)) (q : Fin d)
    (hj : ¬ j.val < m) :
    coyoteLayerCyclic shiftSumCollectParameters state j q = 0 := by
  simp [coyoteLayerCyclic, shiftSumCollectParameters, linearProjection, hj]

/-- A stack of three cyclic Coyote layers for the shifted-diagonal sum.
Source: Appendix `prop: butterfly-hyena`, corrected constant-depth model. -/
def shiftSumNetwork {m d : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m d) (shift : Fin 3 → Fin m) :
    CyclicCoyoteNetwork (9 * m) d := {
  layers := [shiftSumCopyParameters, shiftSumBranchParameters coeff shift,
    shiftSumCollectParameters]
}

/-- The three-layer stack computes the shifted-diagonal sum and returns
the workspace to a zero-padded state, allowing exact subsequent stacking.
Source: Appendix `prop: butterfly-hyena` and `lem: stacking-layers`. -/
theorem shiftSumNetwork_correct {m d : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m d) (shift : Fin 3 → Fin m)
    (u : RealSequence m d) :
    (shiftSumNetwork coeff shift).run (shiftSumPad u) =
      shiftSumPad (shiftedDiagonalSum coeff shift u) := by
  change coyoteLayerCyclic shiftSumCollectParameters
    (coyoteLayerCyclic (shiftSumBranchParameters coeff shift)
      (coyoteLayerCyclic shiftSumCopyParameters (shiftSumPad u))) = _
  rw [shiftSumCopyParameters_apply]
  funext j q
  by_cases hj : j.val < m
  · let i : Fin m := ⟨j.val, hj⟩
    have hslot : shiftSumInputSlot i = j := Fin.ext rfl
    rw [← hslot, shiftSumCollectParameters_input]
    simp only [shiftSumBranchParameters_apply]
    simp [shiftSumPad, shiftSumInputSlot, shiftedDiagonalSum]
  · rw [shiftSumCollectParameters_outside _ j q hj]
    simp [shiftSumPad, hj]

/-- Exactly three layers are used, independently of sequence length and
coefficients. Source: Appendix `prop: butterfly-hyena`, constant depth. -/
theorem shiftSumNetwork_layerCount {m d : ℕ} [NeZero m]
    (coeff : Fin 3 → RealSequence m d) (shift : Fin 3 → Fin m) :
    (shiftSumNetwork coeff shift).layerCount = 3 := rfl

/-- The outside-block hypothesis is realized by an actual workspace row.
Source: Appendix `prop: prim-add`, finite workspace. -/
example : ¬ (⟨1, by decide⟩ : Fin 9).val < 1 := by decide

end Transformer.Zoology
