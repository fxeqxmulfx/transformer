module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction.Basic
public import Mathlib.Analysis.Complex.Order
public import Mathlib.Analysis.Convex.Combination
public import Mathlib.LinearAlgebra.Matrix.PosDef
public import Mathlib.LinearAlgebra.Matrix.Trace

/-!
# A convex matrix lift of constant amplitude

Source: `review/flat_spectrum.md`, CM21. The variable is a Hermitian positive
semidefinite matrix of trace one. There is no rank constraint. A supplied
feature matrix has columns normalized to squared norm `q`, so the affine
constant-diagonal constraint is one rather than `1/q`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix
open scoped ComplexOrder

variable {ι X : Type*} [Fintype ι]

/-- Positive semidefinite, trace-one coefficient matrices, CM21. -/
def Density (M : Matrix ι ι ℂ) : Prop := M.PosSemidef ∧ M.trace = 1

/-- An observable entry of the lifted feature kernel, CM21. -/
noncomputable def reading (B : Matrix X ι ℂ) (M : Matrix ι ι ℂ) (x y : X) : ℂ :=
  (B * M * B.conjTranspose) x y

/-- The convex lift uses affine diagonal readings, not vector amplitudes, CM21. -/
def Feasible (B : Matrix X ι ℂ) (M : Matrix ι ι ℂ) : Prop :=
  Density M ∧ ∀ x, reading B M x x = 1

/-- Unnormalized Fourier features with a possibly unknown row permutation, CM21. -/
noncomputable def basis {q : ℕ} [NeZero q] (r : X → ZMod q)
    (k : ι → ZMod q) : Matrix X ι ℂ := fun x i => ZMod.stdAddChar (r x * k i)

/-- A pure coefficient projector; membership is derived, never imposed, CM21. -/
noncomputable def projector [DecidableEq ι] (i : ι) : Matrix ι ι ℂ :=
  diagonal (Pi.single i 1)

/-- Real probability weights for a convex combination, CM21. -/
def Weights (w : ι → ℝ) : Prop := (∀ i, 0 ≤ w i) ∧ ∑ i, w i = 1

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
