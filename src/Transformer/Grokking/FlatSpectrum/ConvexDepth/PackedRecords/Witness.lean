module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords.Complex

/-!
# Nonconstant packed-state and complete complex witnesses

Source: `review/convex/moments.md`, CM68. Five record IDs include duplicates,
nonconstant labels and a singleton tail. A different witness reaches the native
state boundary and maximal 16-bit label. Invalid stored states are rejected.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords

open LatentPrograms (Parameters features head)

/-- Two input-only coordinate programs, CM68. -/
def witnessProgram (j : Bool) (x : Fin 4 × Fin 4) : Fin 4 := if j then x.2 else x.1

/-- Original IDs include repeated observations and two distinct raw labels,
CM68. Record four supplies the genuine singleton minibatch tail. -/
def witnessRows (i : Fin 5) : (Fin 4 × Fin 4) × Fin 3 :=
  if i.val < 2 then ((0, 0), 0) else ((2, 2), 2)

/-- A noninjective raw decoder used only in this formal witness, CM68. -/
def witnessLabel (s : Fin 4) : Fin 3 := if s = 0 then 0 else if s = 1 then 1 else 2

/-- Unequal masses and all complete raw decoder entries, CM68. -/
def witnessPoint : Parameters Bool (Fin 4) (Fin 3) :=
  (fun j => if j then 2 else 1,
   fun j s c => if c = witnessLabel s then if j then 2 else 1 else 0)

/-- The five records have actual duplicate IDs and nonconstant states/classes,
CM68. Packing must preserve these occurrences rather than merge them. -/
theorem witness_nonconstant :
    witnessRows 0 = witnessRows 1 ∧ witnessRows 0 ≠ witnessRows 4 ∧
    witnessProgram false (witnessRows 0).1 = 0 ∧
    witnessProgram false (witnessRows 4).1 = 2 ∧
    (witnessRows 0).2 = 0 ∧ (witnessRows 4).2 = 2 := by
  decide

/-- The largest declared native state and largest UInt16 raw label survive
actual word conversion, CM68. This is independent of the small CE witness. -/
theorem witness_boundary_words :
    ((encode boundaryRow).1 true).toNat = 4095 ∧
    ((encode boundaryRow).1 false).toNat = 1 ∧ (encode boundaryRow).2.toNat = 65535 := by
  rw [encoded_state (by decide : 4096 ≤ UInt16.size),
    encoded_state (by decide : 4096 ≤ UInt16.size),
    encoded_label (by decide : 65536 ≤ UInt16.size)]
  norm_num [boundaryRow]

/-- A representable word outside the declared state domain is rejected,
CM68; fitting it into 16 bits does not make it a valid latent state. -/
theorem witness_reject_state :
    checkedDecode 4 3 ((fun _ : Bool => UInt16.ofNat 4), UInt16.ofNat 2) = none := by
  apply (checkedDecode_none_iff _).mpr
  intro h
  have hx := h.1 false
  norm_num at hx

/-- The witness's actual readout has nonzero imaginary source and head
coordinates, CM68. All of these coefficients remain unchanged by input packing. -/
theorem witness_imaginary :
    features fourier4 witnessProgram (true, (1, true)) (3, 1) = -(1 / 2 : ℝ) ∧
    head fourier4 1 witnessPoint (true, (1, true)) (1 : Fin 3) = -1 := by
  norm_num [features, head, tableHead, complexTable, complexCoefficients, complexFeatures,
    Matrix.mul_apply, Matrix.conjTranspose_apply, Fin.sum_univ_four, fourier4,
    witnessProgram, witnessPoint, witnessLabel]

example := indexed_roundtrip (by decide : 4 ≤ UInt16.size) (by decide : 3 ≤ UInt16.size)
  (fun i => states witnessProgram (witnessRows i))
example := ordered_roundtrip (by decide : 4 ≤ UInt16.size) (by decide : 3 ≤ UInt16.size)
  (fun i => states witnessProgram (witnessRows i)) [0, 1, 2, 3, 4]
example := ordered_roundtrip (by decide : 4 ≤ UInt16.size) (by decide : 3 ≤ UInt16.size)
  (fun i => states witnessProgram (witnessRows i)) [4]
example := lookup_states witnessProgram ([0, 1, 2, 3, 4].map witnessRows) true 2
example := retained_states witnessProgram [false, true] ([0, 1, 2, 3, 4].map witnessRows)
example := domain_states witnessProgram [false, true] ([0, 1, 2, 3, 4].map witnessRows) 1 32
example := domain_roundtrip (by decide : 4 ≤ UInt16.size) (by decide : 3 ≤ UInt16.size)
  witnessProgram [false, true] witnessRows [0, 1, 2, 3, 4] 1 32
example := logits_states fourier4 witnessProgram witnessPoint (3, 1)
example := logits_roundtrip (by decide : 4 ≤ UInt16.size) (by decide : 3 ≤ UInt16.size)
  fourier4 witnessProgram witnessPoint ((3, 1), 1)
example := batchLoss_roundtrip (by decide : 4 ≤ UInt16.size) (by decide : 3 ≤ UInt16.size)
  fourier4 witnessProgram witnessRows Finset.univ witnessPoint
example := batchLoss_roundtrip (by decide : 4 ≤ UInt16.size) (by decide : 3 ≤ UInt16.size)
  fourier4 witnessProgram witnessRows {4} witnessPoint

end Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords
