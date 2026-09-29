/-
# Packing a sequence into circuit feature memory

Arora et al., arXiv:2312.04927v1, Appendix Theorem `thm: gen-ac`.
Two permitted cyclic Coyote layers copy each original feature into its
own memory column and rotate that column so the first row contains the
entire flattened input. This supplies the input-layout bridge omitted by
the one-token circuit compiler.
-/

import Transformer.Zoology.Appendix_CyclicCircuitNetwork
import Transformer.Zoology.Appendix_PaddedModel

namespace Transformer.Zoology

/-- Flatten the row and feature indices of an external sequence.
Source: Appendix Theorem `thm: gen-ac`, `u ∈ ℝ^(nd)`. -/
def circuitFlatIndex {n d : ℕ} (i : Fin n) (q : Fin d) : Fin (n * d) :=
  finProdFinEquiv (i, q)

/-- View an external sequence as the arithmetic circuit's input vector.
Source: Appendix Theorem `thm: gen-ac`, flattened input `u`. -/
def circuitFlatInput {n d : ℕ} (u : RealSequence n d) :
    Fin (n * d) → ℝ :=
  fun a => u (finProdFinEquiv.symm a).1 (finProdFinEquiv.symm a).2

/-- Flattening and then reading a pair of indices returns the original
entry. Source: Appendix Theorem `thm: gen-ac`, input convention. -/
theorem circuitFlatInput_index {n d : ℕ} (u : RealSequence n d)
    (i : Fin n) (q : Fin d) :
    circuitFlatInput u (circuitFlatIndex i q) = u i q := by
  simp [circuitFlatInput, circuitFlatIndex]

/-- The original feature width fits in the circuit's persistent memory
when the external sequence has at least one row. Source: Appendix
Theorem `thm: gen-ac`, zero-padded embedding. -/
theorem circuitInput_widthBound {n d size : ℕ} [NeZero n] :
    d ≤ n * d + size := by
  have hn : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr (NeZero.ne n)
  have h := Nat.mul_le_mul_right d hn
  omega

/-- Embed an original feature coordinate in the larger circuit memory.
Source: Appendix Theorem `thm: gen-ac`, zero-padded embedding. -/
def circuitOriginalFeature {n d size : ℕ} [NeZero n] (q : Fin d) :
    Fin (n * d + size) :=
  ⟨q.val, lt_of_lt_of_le q.isLt circuitInput_widthBound⟩

/-- Identify the external input represented by an inner memory column.
Node columns do not represent external inputs. Source: Appendix Theorem
`thm: gen-ac`, initial wire layout. -/
def circuitColumnSource {n d size : ℕ}
    (q : Fin (n * d + size)) : Option (Fin n × Fin d) :=
  if h : q.val < n * d then some (finProdFinEquiv.symm ⟨q.val, h⟩)
  else none

/-- An input column carries its own row and feature indices.
Source: Appendix Theorem `thm: gen-ac`, flattened input. -/
theorem circuitColumnSource_input {n d size : ℕ}
    (i : Fin n) (q : Fin d) :
    circuitColumnSource
      (circuitInputSlot (nodes := size) (circuitFlatIndex i q)) =
        some (i, q) := by
  have h : (circuitInputSlot (nodes := size)
      (circuitFlatIndex i q)).val < n * d :=
    (circuitFlatIndex i q).isLt
  simp only [circuitColumnSource, dite_eq_left h]
  congr 1
  exact Equiv.symm_apply_apply finProdFinEquiv (i, q)

/-- A gate-output column has no external-input source.
Source: Appendix Theorem `thm: gen-ac`, reserved gate slots. -/
theorem circuitColumnSource_node {n d size : ℕ} (j : Fin size) :
    circuitColumnSource
      (circuitNodeSlot (inputs := n * d) j) = none := by
  simp [circuitColumnSource, circuitNodeSlot]

/-- A shared linear projection duplicates feature `q` into every input
column whose source feature is `q`. Source: Appendix Theorem
`thm: gen-ac`, initial rearrangement. -/
def circuitInputCopyWeight {n d size : ℕ} [NeZero n] :
    Fin (n * d + size) → Fin (n * d + size) → ℝ :=
  fun k target =>
    match circuitColumnSource target with
    | some (_, q) => if k = circuitOriginalFeature q then 1 else 0
    | none => 0

/-- The copying layer uses the ordinary linear-projection primitive.
Source: Appendix `lmm:primitives`, linear projection. -/
def circuitInputCopyParameters {n d size : ℕ} [NeZero n] :
    CoyoteParameters n (n * d + size) :=
  linearCoyoteParameters circuitInputCopyWeight

/-- The copying layer fills each flattened input column with the
corresponding feature from the same row. Source: Appendix Theorem
`thm: gen-ac`, input rearrangement. -/
theorem circuitInputCopy_input {n d size : ℕ} [NeZero n]
    (u : RealSequence n d) (r i : Fin n) (q : Fin d) :
    coyoteLayerCyclic (circuitInputCopyParameters (size := size))
        (padSequence u) r
        (circuitInputSlot (nodes := size) (circuitFlatIndex i q)) =
      u r q := by
  rw [show coyoteLayerCyclic
      (circuitInputCopyParameters (size := size)) (padSequence u) r
      (circuitInputSlot (nodes := size) (circuitFlatIndex i q)) =
        linearProjection (padSequence u) circuitInputCopyWeight r
        (circuitInputSlot (nodes := size) (circuitFlatIndex i q)) by
      simp [coyoteLayerCyclic, circuitInputCopyParameters,
        linearCoyoteParameters, cyclicConvolution]]
  simp only [linearProjection, circuitInputCopyWeight,
    circuitColumnSource_input]
  rw [Finset.sum_eq_single (circuitOriginalFeature q)]
  · simp [padSequence, circuitOriginalFeature]
  · intro k _ hk
    simp [hk]
  · intro h
    exact (h (Finset.mem_univ (circuitOriginalFeature q))).elim

/-- The copying layer leaves every reserved gate slot zero.
Source: Appendix Theorem `thm: gen-ac`, initial memory. -/
theorem circuitInputCopy_node {n d size : ℕ} [NeZero n]
    (u : RealSequence n d) (r : Fin n) (j : Fin size) :
    coyoteLayerCyclic (circuitInputCopyParameters (size := size))
        (padSequence u) r (circuitNodeSlot (inputs := n * d) j) = 0 := by
  simp [coyoteLayerCyclic, circuitInputCopyParameters,
    linearCoyoteParameters, linearProjection, circuitInputCopyWeight,
    circuitColumnSource_node, cyclicConvolution]

/-- Rotate every copied input column so its source row reaches row zero.
The shift is zero on reserved gate columns. Source: Appendix Proposition
`prop: prim-shift`, cyclic variant. -/
def circuitGatherShift {n d size : ℕ} [NeZero n]
    (target : Fin (n * d + size)) : Fin n :=
  match circuitColumnSource target with
  | some (i, _) => -i
  | none => 0

/-- A columnwise cyclic convolution gathers the flattened input at row
zero. Source: Appendix Theorem `thm: gen-ac`, input rearrangement. -/
def circuitGatherParameters {n d size : ℕ} [NeZero n] :
    CoyoteParameters n (n * d + size) :=
  convolutionCoyoteParameters
    (fun k q => if k = circuitGatherShift q then 1 else 0)

/-- The gathered row-zero input column equals the corresponding original
sequence entry. Source: Appendix Theorem `thm: gen-ac`, input layout. -/
theorem circuitGather_input {n d size : ℕ} [NeZero n]
    (u : RealSequence n d) (i : Fin n) (q : Fin d) :
    coyoteLayerCyclic (circuitGatherParameters (size := size))
      (coyoteLayerCyclic (circuitInputCopyParameters (size := size))
        (padSequence u)) 0
      (circuitInputSlot (nodes := size) (circuitFlatIndex i q)) =
        u i q := by
  unfold circuitGatherParameters
  rw [coyote_cyclic_realizes_convolution,
    cyclicConvolution_columnImpulse]
  have hs : (0 : Fin n) - (-i) = i := by
    simp
  simpa [circuitGatherShift, circuitColumnSource_input, hs] using
    circuitInputCopy_input (size := size) u i i q

/-- The gathered gate slots are still zero at row zero.
Source: Appendix Theorem `thm: gen-ac`, initial memory. -/
theorem circuitGather_node {n d size : ℕ} [NeZero n]
    (u : RealSequence n d) (j : Fin size) :
    coyoteLayerCyclic (circuitGatherParameters (size := size))
      (coyoteLayerCyclic (circuitInputCopyParameters (size := size))
        (padSequence u)) 0 (circuitNodeSlot (inputs := n * d) j) = 0 := by
  unfold circuitGatherParameters
  rw [coyote_cyclic_realizes_convolution,
    cyclicConvolution_columnImpulse]
  simpa [circuitGatherShift, circuitColumnSource_node] using
    circuitInputCopy_node (size := size) u (0 : Fin n) j

/-- The first row after two layout layers is exactly the compiler's
initial one-token memory. Source: Appendix Theorem `thm: gen-ac`, input
layout and persistent wires. -/
theorem circuitGather_rowZero {n d size : ℕ} [NeZero n]
    (u : RealSequence n d) :
    (fun q => coyoteLayerCyclic (circuitGatherParameters (size := size))
      (coyoteLayerCyclic (circuitInputCopyParameters (size := size))
        (padSequence u)) 0 q) =
      initialCircuitMemory (nodes := size) (circuitFlatInput u) := by
  funext target
  refine Fin.addCases (motive := fun q =>
    coyoteLayerCyclic (circuitGatherParameters (size := size))
      (coyoteLayerCyclic (circuitInputCopyParameters (size := size))
        (padSequence u)) 0 q =
      initialCircuitMemory (nodes := size) (circuitFlatInput u) q)
    ?_ ?_ target
  · intro a
    let p := finProdFinEquiv.symm a
    have ha : circuitFlatIndex p.1 p.2 = a := by
      exact Equiv.apply_symm_apply finProdFinEquiv a
    have hslot : circuitInputSlot (nodes := size) a =
        Fin.castAdd size a := Fin.ext rfl
    have h := circuitGather_input (size := size) u p.1 p.2
    rw [ha, hslot] at h
    simpa [initialCircuitMemory, circuitFlatInput, p] using h
  · intro j
    have hslot : circuitNodeSlot (inputs := n * d) j =
        Fin.natAdd (n * d) j := Fin.ext rfl
    have h := circuitGather_node (size := size) u j
    rw [hslot] at h
    simpa [initialCircuitMemory] using h

end Transformer.Zoology
