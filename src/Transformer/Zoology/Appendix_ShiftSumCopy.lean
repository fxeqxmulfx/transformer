/-
# The duplicated-input stage of the shifted-diagonal compiler

Arora et al., arXiv:2312.04927v1, Appendix `prop: prim-add`,
`prop: prim-shift`, and `prop: butterfly-hyena`. A two-tap cyclic
convolution creates two adjacent copies. Every branch then reads a
cyclically shifted input from those copies without wraparound leakage.
-/

import Transformer.Zoology.Appendix_ShiftSumLayout

open scoped BigOperators

namespace Transformer.Zoology

/-- The first stage keeps two adjacent copies of the zero-padded input.
Source: Appendix `prop: prim-add`, two-tap filter. -/
def shiftSumCopyState {m d : ℕ} [NeZero m]
    (u : RealSequence m d) : RealSequence (9 * m) d :=
  fun j q => shiftSumPad u j q +
    shiftSumPad u (j - shiftSumCopyOffset m) q

/-- Parameters of the input-duplication layer.
Source: Appendix `prop: prim-add`, corrected cyclic construction. -/
def shiftSumCopyParameters {m d : ℕ} [NeZero m] :
    CoyoteParameters (9 * m) d :=
  convolutionCoyoteParameters
    (fun j q => impulse 0 j q + impulse (shiftSumCopyOffset m) j q)

/-- One cyclic Coyote layer creates the two adjacent copies.
Source: Appendix `prop: prim-add`, duplication component. -/
theorem shiftSumCopyParameters_apply {m d : ℕ} [NeZero m]
    (u : RealSequence m d) :
    coyoteLayerCyclic shiftSumCopyParameters (shiftSumPad u) =
      shiftSumCopyState u := by
  unfold shiftSumCopyParameters
  rw [coyote_cyclic_realizes_convolution]
  funext j q
  unfold cyclicConvolution
  simp only [add_mul, Finset.sum_add_distrib]
  change cyclicConvolution (shiftSumPad u) (impulse 0) j q +
    cyclicConvolution (shiftSumPad u) (impulse (shiftSumCopyOffset m)) j q = _
  rw [cyclicConvolution_impulse, cyclicConvolution_impulse]
  simp [shiftSumCopyState]

/-- The duplicated input vanishes beyond its first two blocks.
Source: Appendix `prop: butterfly-hyena`, separation of branches. -/
theorem shiftSumCopyState_outside {m d : ℕ} [NeZero m]
    (u : RealSequence m d) (j : Fin (9 * m)) (q : Fin d)
    (hj : 2 * m ≤ j.val) : shiftSumCopyState u j q = 0 := by
  have hle : shiftSumCopyOffset m ≤ j := by
    apply Fin.le_def.mpr
    dsimp [shiftSumCopyOffset]
    omega
  have hv : (j - shiftSumCopyOffset m).val = j.val - m := by
    rw [Fin.sub_val_of_le hle]
    rfl
  have hj0 : ¬ j.val < m := by omega
  have hj1 : ¬ (j - shiftSumCopyOffset m).val < m := by omega
  simp [shiftSumCopyState, shiftSumPad, hj0, hj1]

/-- The first copy remains at its original row indices.
Source: Appendix `prop: prim-add`, duplication without wraparound. -/
theorem shiftSumCopyState_first {m d : ℕ} [NeZero m]
    (u : RealSequence m d) (i : Fin m) (q : Fin d) :
    shiftSumCopyState u (shiftSumInputSlot i) q = u i q := by
  have hnot : ¬ shiftSumCopyOffset m ≤ shiftSumInputSlot i := by
    simpa only [Fin.le_def, shiftSumCopyOffset, shiftSumInputSlot, not_le]
      using i.isLt
  have hv := Fin.intCast_val_sub_eq_sub_add_ite (shiftSumInputSlot i)
    (shiftSumCopyOffset m)
  simp only [hnot, ite_false] at hv
  have hv' : (((shiftSumInputSlot i - shiftSumCopyOffset m).val) : ℤ) =
      i.val - m + 9 * m := hv
  have hout : ¬ (shiftSumInputSlot i - shiftSumCopyOffset m).val < m := by
    omega
  simp only [shiftSumCopyState, shiftSumPad, hout, dite_false, add_zero]
  simp [shiftSumInputSlot, i.isLt]

/-- At an offset `m+i-s` in the duplicated input, the value is exactly
the cyclically shifted original token `i-s`.
Source: Appendix `prop: prim-shift`, modular shift semantics. -/
theorem shiftSumCopyState_read {m d : ℕ} [NeZero m]
    (u : RealSequence m d) (i s : Fin m) (j : Fin (9 * m))
    (hj : j.val = m + i.val - s.val) (q : Fin d) :
    shiftSumCopyState u j q = u (i - s) q := by
  by_cases hsi : s ≤ i
  · have hval : (i - s).val = i.val - s.val := Fin.sub_val_of_le hsi
    have hle : shiftSumCopyOffset m ≤ j := by
      apply Fin.le_def.mpr
      have hsi' := Fin.le_def.mp hsi
      dsimp [shiftSumCopyOffset]
      omega
    have hv : (j - shiftSumCopyOffset m).val = i.val - s.val := by
      rw [Fin.sub_val_of_le hle]
      dsimp [shiftSumCopyOffset]
      have hsi' := Fin.le_def.mp hsi
      omega
    have hj0 : ¬ j.val < m := by
      have hsi' := Fin.le_def.mp hsi
      omega
    have hj1 : (j - shiftSumCopyOffset m).val < m := by omega
    simp only [shiftSumCopyState, shiftSumPad, hj0, dite_false,
      hj1, dite_true, zero_add]
    congr 1
    exact Fin.ext (hv.trans hval.symm)
  · have hval := Fin.intCast_val_sub_eq_sub_add_ite i s
    simp only [hsi, ite_false] at hval
    have hj0 : j.val < m := by
      have hsi' : i.val < s.val := by
        simpa only [Fin.le_def, not_le] using hsi
      omega
    have hnot : ¬ shiftSumCopyOffset m ≤ j := by
      simp only [Fin.le_def, shiftSumCopyOffset]
      omega
    have hv := Fin.intCast_val_sub_eq_sub_add_ite j (shiftSumCopyOffset m)
    simp only [hnot, ite_false] at hv
    have hv' : ((j - shiftSumCopyOffset m).val : ℤ) =
        j.val - m + 9 * m := hv
    have hj1 : ¬ (j - shiftSumCopyOffset m).val < m := by omega
    simp only [shiftSumCopyState, shiftSumPad, hj0, dite_true,
      hj1, dite_false, add_zero]
    congr 1
    apply Fin.ext
    change j.val = (i - s).val
    omega

/-- A sum of cyclic impulses reads the sum of the corresponding shifted
values. Source: Appendix `prop: prim-add` and `prop: butterfly-hyena`,
addition of convolution branches. -/
theorem cyclicConvolution_threeImpulses {n d : ℕ}
    (u : RealSequence n d) (tap : Fin 3 → Fin n) (i : Fin n) (q : Fin d) :
    cyclicConvolution u (fun k _ => ∑ a : Fin 3,
      if k = tap a then 1 else 0) i q =
        ∑ a : Fin 3, u (i - tap a) q := by
  classical
  unfold cyclicConvolution
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  simp [ite_mul]

/-- Weighted impulses form a prescribed linear combination of shifted
rows. Source: Appendix `prop: prim-add`, signed branch cancellation. -/
theorem cyclicConvolution_weightedThreeImpulses {n d : ℕ}
    (u : RealSequence n d) (tap : Fin 3 → Fin n) (weight : Fin 3 → ℝ)
    (i : Fin n) (q : Fin d) :
    cyclicConvolution u (fun k _ => ∑ a : Fin 3,
      if k = tap a then weight a else 0) i q =
        ∑ a : Fin 3, weight a * u (i - tap a) q := by
  classical
  unfold cyclicConvolution
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  simp [ite_mul]

/-- The support hypothesis of the duplication lemma is satisfiable.
Source: Appendix `prop: butterfly-hyena`, concrete workspace. -/
example : 2 * 1 ≤ (⟨2, by decide⟩ : Fin (9 * 1)).val := by decide

/-- The modular read hypothesis is satisfiable even at a wrapped input
shift. Source: Appendix `prop: prim-shift`, cyclic option. -/
example : (⟨1, by decide⟩ : Fin 18).val =
    2 + (0 : Fin 2).val - (1 : Fin 2).val := by decide

end Transformer.Zoology
