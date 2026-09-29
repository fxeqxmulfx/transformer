/-
# Shift primitives and the identity-filter error

Arora et al., arXiv:2312.04927v1, Appendix Proposition `prop: prim-shift`.
The proof describes a down-shift kernel as being at `s+1`, but its polynomial
calculation uses `X^s`; the latter is the correct offset.  It also says
cyclic convolution with the identity filter reverses a sequence.  That
intermediate claim is false: the identity filter leaves the sequence fixed.
-/

import Transformer.Zoology.Section4_Coyote

open scoped BigOperators

namespace Transformer.Zoology

/-- A unit impulse at shift `s`, in every feature column.
Source: Appendix Proposition `prop: prim-shift`, corrected from the printed
`s+1` kernel to the `X^s` kernel used by its subsequent calculation. -/
def impulse {n d : ℕ} (s : Fin n) : RealSequence n d :=
  fun k _ => if k = s then 1 else 0

/-- A causal down-shift by `s` positions, with zeros above the shifted data.
Source: Appendix Proposition `prop: prim-shift`. -/
def shiftDown {n d : ℕ} (u : RealSequence n d)
    (s : Fin n) : RealSequence n d :=
  fun i q => if s ≤ i then u (i - s) q else 0

/-- Convolution with a unit impulse implements a causal down-shift.
Source: Appendix Proposition `prop: prim-shift`, corrected kernel offset. -/
theorem causalConvolution_impulse {n d : ℕ} (u : RealSequence n d)
    (s : Fin n) : causalConvolution u (impulse s) = shiftDown u s := by
  classical
  funext i q
  unfold causalConvolution shiftDown
  rw [Finset.sum_eq_single s]
  · by_cases h : s ≤ i
    · simp [h, impulse]
    · simp [h]
  · intro k _ hks
    simp [impulse, hks]
  · intro hs
    exact (hs (Finset.mem_univ s)).elim

/-- One Coyote layer implements the corrected down-shift primitive.
Source: Appendix Proposition `prop: prim-shift`, down-shift half. -/
theorem coyote_realizes_shiftDown {n d : ℕ} (u : RealSequence n d)
    (s : Fin n) :
    coyoteLayer (convolutionCoyoteParameters (impulse s)) u =
      shiftDown u s := by
  rw [coyote_realizes_convolution, causalConvolution_impulse]

/-- Zero shift is the identity for nonempty sequences.
Source: Appendix `lmm:primitives`, identity-filter construction. -/
theorem shiftDown_zero {n d : ℕ} [NeZero n] (u : RealSequence n d) :
    shiftDown u (0 : Fin n) = u := by
  funext i q
  simp [shiftDown]

/-- Parameters for gating an input by a fixed position-dependent factor.
Source: Appendix `lmm:primitives`, third construction. -/
def fixedGateParameters {n d : ℕ} [NeZero n]
    (factor : RealSequence n d) : CoyoteParameters n d := {
  weight := fun _ _ => 0
  filter := impulse 0
  bias₁ := factor
  bias₂ := fun _ _ => 0
}

/-- One Coyote layer gates an input by a fixed matrix.  Together with
`coyote_realizes_linear` and `coyote_realizes_convolution`, this proves the
three functional constructions in Appendix `lmm:primitives`. -/
theorem coyote_realizes_fixed_gate {n d : ℕ} [NeZero n]
    (u factor : RealSequence n d) :
    coyoteLayer (fixedGateParameters factor) u =
      fun i q => factor i q * u i q := by
  funext i q
  simp [coyoteLayer, fixedGateParameters, linearProjection,
    causalConvolution_impulse, shiftDown_zero]

/-- Cyclic convolution by the impulse at zero is the identity map.
Source: Appendix Proposition `prop: prim-shift`, second construction;
this refutes its assertion that this operation reverses the input. -/
theorem cyclicConvolution_impulse_zero {n d : ℕ} [NeZero n]
    (u : RealSequence n d) :
    cyclicConvolution u (impulse (0 : Fin n)) = u := by
  classical
  funext i q
  unfold cyclicConvolution
  rw [Finset.sum_eq_single (0 : Fin n)]
  · simp [impulse]
  · intro k _ hk
    simp [impulse, hk]
  · intro h
    exact (h (Finset.mem_univ (0 : Fin n))).elim

/-- An explicit input on which the paper's asserted reversal differs from
cyclic convolution by the identity filter: position zero remains zero,
whereas reversal would return the last entry, two.
Source: Appendix Proposition `prop: prim-shift`, second construction. -/
theorem identity_filter_does_not_reverse :
    cyclicConvolution (fun i : Fin 3 => fun _ : Fin 1 => (i : ℝ))
        (impulse (0 : Fin 3)) 0 0 ≠
      (fun i : Fin 3 => fun _ : Fin 1 => (i : ℝ)) 2 0 := by
  rw [cyclicConvolution_impulse_zero]
  norm_num

end Transformer.Zoology
