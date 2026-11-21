module

public import Transformer.Grokking.FlatSpectrum.TablePart.E6FloorPrimal

/-!
# PT6: M misses the table of `floor-product`

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The kernel of
`⌊a b / 97⌋` has rank 96 and no structure we could use exactly, so the proof is by certified
numbers. A contraction next to its polar factor shows `‖K‖_S1 ≥ 153.7` (`TablePart.E6FloorDual`).
A completion from the training half, equal to a matrix of rank 4 on the unseen cells, has norm at
most 152.8 (`TablePart.E6FloorPrimal`). So `K` is not a minimum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

attribute [local instance] fact_prime_97

/-- **M does not return the table of `floor-product`** from the training half. -/
theorem not_returned_floorProduct : NotReturned floorTable := by
  intro hmin
  have hX : floorFill ∈ {X | ∀ ij ∈ (half : Set ((Fin 1 → ZMod 97) × (Fin 1 → ZMod 97))),
      X ij.1 ij.2 = tableKernel floorTable ij.1 ij.2} := by
    intro ij hij
    have hb := (mem_half_line ij.1 ij.2).1 (Finset.mem_coe.1 hij)
    simp [floorFill, hb]
  have hle := isMinOn_iff.1 hmin _ hX
  have h1 := nuclearNorm_floor_ge
  have h2 := nuclearNorm_floorFill_le
  linarith

example : NotReturned floorTable := not_returned_floorProduct

end Transformer.Grokking.FlatSpectrum.TablePart.E6
