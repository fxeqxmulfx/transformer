/-
# Genuine matrix transpose and multiplication for factor products

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope` and
`lmm: kaleido-coyote`. The transpose reverses the factor order and
transposes each factor. Matrix entries and matrix-vector multiplication
are proved from the adjoint identity, not assumed in a representation.
-/

import Transformer.Zoology.Appendix_BinaryButterflyTranspose

open scoped BigOperators

namespace Transformer.Zoology

/-- Transpose every factor and reverse its multiplication order.
Source: Appendix `def: kaleidoscope`, the B* factor. -/
def transposeBinaryButterflyStages {p k : ℕ}
    (stages : List (BinaryButterflyStage p k)) : List (BinaryButterflyStage p k) :=
  stages.reverse.map BinaryButterflyStage.transpose

/-- Executing concatenated products is ordinary function composition.
Source: Appendix `lem: stacking-layers` and `def: kaleidoscope`. -/
theorem applyBinaryButterflyStages_append {p k : ℕ}
    (first second : List (BinaryButterflyStage p k))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    applyBinaryButterflyStages (first ++ second) u =
      applyBinaryButterflyStages second (applyBinaryButterflyStages first u) := by
  simp [applyBinaryButterflyStages, List.foldl_append]

/-- The reversed product is the adjoint of the original complete product.
Source: Appendix `def: kaleidoscope`, transpose of a butterfly matrix. -/
theorem transposeBinaryButterflyStages_adjoint {p k : ℕ}
    (stages : List (BinaryButterflyStage p k))
    (u v : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    butterflyGridDot (applyBinaryButterflyStages stages u) v =
      butterflyGridDot u (applyBinaryButterflyStages (transposeBinaryButterflyStages stages) v) := by
  induction stages generalizing u with
  | nil => rfl
  | cons s ss ih =>
      change butterflyGridDot (applyBinaryButterflyStages ss (s.apply u)) v = _
      rw [ih, s.transpose_adjoint]
      simp [transposeBinaryButterflyStages, List.reverse_cons,
        applyBinaryButterflyStages]

/-- Coordinate basis vector on the original row-major layout.
Source: Appendix `def: kaleidoscope`, scalar matrix coefficients. -/
def butterflyGridBasis {p k : ℕ} (c : ButterflyGrid p k) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  fun i q => if (i, q) = c then 1 else 0

/-- Inner product against a coordinate basis extracts that coordinate.
Source: Appendix `def: kaleidoscope`, basis-vector matrix convention. -/
theorem butterflyGridDot_basis_right {p k : ℕ}
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) (c : ButterflyGrid p k) :
    butterflyGridDot u (butterflyGridBasis c) = u c.1 c.2 := by
  simp [butterflyGridDot, butterflyGridBasis, mul_ite]

/-- A coordinate basis in the first argument also extracts a coordinate.
Source: Appendix `def: kaleidoscope`, real Euclidean inner product. -/
theorem butterflyGridDot_basis_left {p k : ℕ}
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) (c : ButterflyGrid p k) :
    butterflyGridDot (butterflyGridBasis c) u = u c.1 c.2 := by
  simp [butterflyGridDot, butterflyGridBasis, ite_mul]

/-- Matrix of a complete product, by its action on coordinate vectors.
Source: Appendix `def: butterfly` and `def: kaleidoscope`. -/
def binaryButterflyStagesMatrix {p k : ℕ} (stages : List (BinaryButterflyStage p k))
    (out input : ButterflyGrid p k) : ℝ :=
  applyBinaryButterflyStages stages (butterflyGridBasis input) out.1 out.2

/-- The reversed factor product has exactly the transposed matrix entries.
Source: Appendix `def: kaleidoscope`, genuine matrix transpose. -/
theorem binaryButterflyStagesMatrix_transpose {p k : ℕ}
    (stages : List (BinaryButterflyStage p k)) (out input : ButterflyGrid p k) :
    binaryButterflyStagesMatrix (transposeBinaryButterflyStages stages) out input =
      binaryButterflyStagesMatrix stages input out := by
  have h := transposeBinaryButterflyStages_adjoint stages
    (butterflyGridBasis out) (butterflyGridBasis input)
  simp only [butterflyGridDot_basis_left, butterflyGridDot_basis_right] at h
  exact h.symm

/-- The product acts by its scalar matrix on every input, including every
feature coordinate. Source: Appendix `lmm: kaleido-coyote`, matrix semantics. -/
theorem applyBinaryButterflyStages_eq_matrix {p k : ℕ}
    (stages : List (BinaryButterflyStage p k))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) (out : ButterflyGrid p k) :
    applyBinaryButterflyStages stages u out.1 out.2 =
      ∑ input : ButterflyGrid p k, binaryButterflyStagesMatrix stages out input *
        u input.1 input.2 := by
  have h := transposeBinaryButterflyStages_adjoint stages u (butterflyGridBasis out)
  rw [butterflyGridDot_basis_right] at h
  rw [h]
  unfold butterflyGridDot
  apply Finset.sum_congr rfl
  intro input _
  change u input.1 input.2 *
    binaryButterflyStagesMatrix (transposeBinaryButterflyStages stages) input out = _
  rw [binaryButterflyStagesMatrix_transpose]
  ring

end Transformer.Zoology
