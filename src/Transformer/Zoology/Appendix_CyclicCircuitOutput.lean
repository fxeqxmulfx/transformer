/-
# Returning circuit results to the original sequence positions

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`.
Four cyclic Coyote layers retain the compiler's first row, duplicate its
designated outputs into position-indexed columns, shift those columns to
their destination rows, and sum them into the original feature columns.
-/

import Transformer.Zoology.Appendix_CyclicCircuitInput

namespace Transformer.Zoology

/-- Keep the first row and clear all other rows using a fixed gate.
Source: Appendix `lmm:primitives`, position-dependent gating. -/
def circuitRowZeroMaskParameters {n width : ℕ} [NeZero n] :
    CoyoteParameters n width := {
  weight := fun _ _ => 0
  filter := impulse 0
  bias₁ := fun i _ => if i = 0 then 1 else 0
  bias₂ := fun _ _ => 0
}

/-- The row-zero gate keeps precisely the first row.
Source: Appendix `lmm:primitives`, fixed gate. -/
theorem circuitRowZeroMask_apply {n width : ℕ} [NeZero n]
    (u : RealSequence n width) (i : Fin n) (q : Fin width) :
    coyoteLayerCyclic circuitRowZeroMaskParameters u i q =
      if i = 0 then u i q else 0 := by
  simp [coyoteLayerCyclic, circuitRowZeroMaskParameters,
    linearProjection, cyclicConvolution_impulse_zero]

/-- The circuit output gate whose value should be copied into a flattened
input column. Source: Appendix Theorem `thm: gen-ac`, output wiring. -/
def circuitOutputCopySource {n d : ℕ}
    (c : ArithmeticCircuit (n * d) (n * d))
    (target : Fin (n * d + c.size)) :
    Option (Fin (n * d + c.size)) :=
  (circuitColumnSource target).map fun p =>
    circuitNodeSlot (c.output (circuitFlatIndex p.1 p.2))

/-- An input column receives its designated circuit output gate.
Source: Appendix Theorem `thm: gen-ac`, output wiring. -/
theorem circuitOutputCopySource_input {n d : ℕ}
    (c : ArithmeticCircuit (n * d) (n * d))
    (i : Fin n) (q : Fin d) :
    circuitOutputCopySource c
      (circuitInputSlot (nodes := c.size) (circuitFlatIndex i q)) =
        some (circuitNodeSlot (c.output (circuitFlatIndex i q))) := by
  simp [circuitOutputCopySource, circuitColumnSource_input]

/-- A shared feature projection copies each designated output gate to
its corresponding position-indexed column. Source: Appendix Theorem
`thm: gen-ac`, final linear wiring. -/
def circuitOutputCopyWeight {n d : ℕ}
    (c : ArithmeticCircuit (n * d) (n * d)) :
    Fin (n * d + c.size) → Fin (n * d + c.size) → ℝ :=
  fun source target =>
    match circuitOutputCopySource c target with
    | some gate => if source = gate then 1 else 0
    | none => 0

/-- One linear Coyote layer realizes the output-copy projection.
Source: Appendix `lmm:primitives`, linear projection. -/
def circuitOutputCopyParameters {n d : ℕ}
    (c : ArithmeticCircuit (n * d) (n * d)) :
    CoyoteParameters n (n * d + c.size) :=
  linearCoyoteParameters (circuitOutputCopyWeight c)

/-- After masking and copying, an input column has its designated
output gate's value only at row zero. Source: Appendix Theorem
`thm: gen-ac`, output selection. -/
theorem circuitOutputCopy_input {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (u : RealSequence n (n * d + c.size))
    (r i : Fin n) (q : Fin d) :
    coyoteLayerCyclic (circuitOutputCopyParameters c)
        (coyoteLayerCyclic circuitRowZeroMaskParameters u) r
        (circuitInputSlot (nodes := c.size) (circuitFlatIndex i q)) =
      if r = 0 then
        u 0 (circuitNodeSlot (c.output (circuitFlatIndex i q)))
      else 0 := by
  unfold circuitOutputCopyParameters
  rw [coyote_cyclic_realizes_linear]
  simp only [linearProjection, circuitOutputCopyWeight,
    circuitOutputCopySource_input]
  rw [Finset.sum_eq_single
    (circuitNodeSlot (c.output (circuitFlatIndex i q)))]
  · by_cases hr : r = 0 <;>
      simp [circuitRowZeroMask_apply, hr]
  · intro k _ hk
    simp [hk]
  · intro h
    exact (h (Finset.mem_univ
      (circuitNodeSlot (c.output (circuitFlatIndex i q))))).elim

/-- A flattened column shifts its row-zero value to its destination
token. Source: Appendix Proposition `prop: prim-shift`, cyclic variant. -/
def circuitOutputScatterShift {n d size : ℕ} [NeZero n]
    (target : Fin (n * d + size)) : Fin n :=
  match circuitColumnSource target with
  | some (i, _) => i
  | none => 0

/-- The columnwise cyclic convolution used for output scattering.
Source: Appendix Theorem `thm: gen-ac`, output wiring. -/
def circuitOutputScatterParameters {n d size : ℕ} [NeZero n] :
    CoyoteParameters n (n * d + size) :=
  convolutionCoyoteParameters
    (fun k q => if k = circuitOutputScatterShift q then 1 else 0)

/-- After scattering, an output column is nonzero only in its own row.
Source: Appendix Theorem `thm: gen-ac`, output layout. -/
theorem circuitOutputScatter_input {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (u : RealSequence n (n * d + c.size))
    (r i : Fin n) (q : Fin d) :
    coyoteLayerCyclic circuitOutputScatterParameters
      (coyoteLayerCyclic (circuitOutputCopyParameters c)
        (coyoteLayerCyclic circuitRowZeroMaskParameters u)) r
      (circuitInputSlot (nodes := c.size) (circuitFlatIndex i q)) =
        if r = i then
          u 0 (circuitNodeSlot (c.output (circuitFlatIndex i q)))
        else 0 := by
  unfold circuitOutputScatterParameters
  rw [coyote_cyclic_realizes_convolution,
    cyclicConvolution_columnImpulse]
  dsimp only
  rw [show circuitOutputScatterShift
      (circuitInputSlot (nodes := c.size) (circuitFlatIndex i q)) = i by
        simp [circuitOutputScatterShift, circuitColumnSource_input]]
  rw [circuitOutputCopy_input]
  simp only [sub_eq_zero]

/-- Final shared projection sums all position-indexed copies of each
feature into that feature's original output column. Source: Appendix
Theorem `thm: gen-ac`, output wiring. -/
def circuitOutputProjectWeight {n d size : ℕ} [NeZero n] :
    Fin (n * d + size) → Fin (n * d + size) → ℝ :=
  fun source target =>
    if h : target.val < d then
      ∑ i : Fin n,
        if source = circuitInputSlot (nodes := size)
            (circuitFlatIndex i ⟨target.val, h⟩) then 1 else 0
    else 0

/-- A linear Coyote layer realizes the final output projection.
Source: Appendix `lmm:primitives`, linear projection. -/
def circuitOutputProjectParameters {n d size : ℕ} [NeZero n] :
    CoyoteParameters n (n * d + size) :=
  linearCoyoteParameters circuitOutputProjectWeight

/-- The final projected original feature is the designated output-gate
value in the first row. Source: Appendix Theorem `thm: gen-ac`, exact
functional equivalence for the output-layout stages. -/
theorem circuitOutputProject_correct {n d : ℕ} [NeZero n]
    (c : ArithmeticCircuit (n * d) (n * d))
    (u : RealSequence n (n * d + c.size))
    (r : Fin n) (q : Fin d) :
    coyoteLayerCyclic circuitOutputProjectParameters
      (coyoteLayerCyclic circuitOutputScatterParameters
        (coyoteLayerCyclic (circuitOutputCopyParameters c)
          (coyoteLayerCyclic circuitRowZeroMaskParameters u))) r
      (circuitOriginalFeature (size := c.size) q) =
        u 0 (circuitNodeSlot (c.output (circuitFlatIndex r q))) := by
  unfold circuitOutputProjectParameters
  rw [coyote_cyclic_realizes_linear]
  simp only [linearProjection, circuitOutputProjectWeight]
  simp only [show (circuitOriginalFeature (size := c.size) q).val < d from
    q.isLt, dite_true]
  have hq : (⟨(circuitOriginalFeature (n := n) (d := d)
      (size := c.size) q).val,
      q.isLt⟩ : Fin d) = q := Fin.ext rfl
  rw [hq]
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  simp only [mul_ite, mul_one, mul_zero, Finset.sum_ite_eq',
    Finset.mem_univ, ↓reduceIte]
  simp only [circuitOutputScatter_input]
  simp

end Transformer.Zoology
