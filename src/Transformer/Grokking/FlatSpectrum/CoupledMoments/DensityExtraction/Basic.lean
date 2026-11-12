module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Defs

/-!
# Convexity and observable readings

Source: `review/flat_spectrum.md`, CM21, convex lift. Positive semidefiniteness,
trace and diagonal readings are checked on the matrix variable itself.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix
open scoped ComplexOrder

variable {ι X : Type*} [Fintype ι]

/-- The lifted kernel is linear in its coefficient matrix, CM21. -/
theorem reading_mix (B : Matrix X ι ℂ) (M N : Matrix ι ι ℂ)
    (a b : ℝ) (x y : X) :
    reading B (a • M + b • N) x y = a • reading B M x y + b • reading B N x y := by
  simp [reading, Matrix.mul_add, Matrix.add_mul, Matrix.mul_smul, Matrix.smul_mul]

/-- All PSD trace-one matrices form a convex set, CM21. -/
theorem density_convex : Convex ℝ {M : Matrix ι ι ℂ | Density M} := by
  intro M hM N hN a b ha hb hab
  refine ⟨(hM.1.smul ha).add (hN.1.smul hb), ?_⟩
  simp only [Matrix.trace_add, Matrix.trace_smul, hM.2, hN.2]
  rw [← add_smul, hab, one_smul]

/-- Affine constant-amplitude readings preserve convexity, CM21. -/
theorem feasible_convex (B : Matrix X ι ℂ) : Convex ℝ {M | Feasible B M} := by
  intro M hM N hN a b ha hb hab
  refine ⟨density_convex hM.1 hN.1 ha hb hab, fun x => ?_⟩
  rw [reading_mix, hM.2, hN.2, ← add_smul, hab, one_smul]

/-- Expand the actual matrix entry into its coefficient readings, CM21. -/
theorem reading_eq (B : Matrix X ι ℂ) (M : Matrix ι ι ℂ) (x y : X) :
    reading B M x y = ∑ i, ∑ j, M i j * (B x i * star (B y j)) := by
  simp only [reading, Matrix.mul_apply, Matrix.conjTranspose_apply, sum_mul]
  rw [sum_comm]
  apply sum_congr rfl
  intro i hi
  apply sum_congr rfl
  intro j hj
  ring

/-- A diagonal density reads a weighted average of feature products, CM21. -/
theorem reading_diagonal [DecidableEq ι] (B : Matrix X ι ℂ) (d : ι → ℂ) (x y : X) :
    reading B (diagonal d) x y = ∑ i, d i * (B x i * star (B y i)) := by
  simp [reading_eq, Matrix.diagonal_apply]

/-- A PSD trace-one diagonal is precisely a probability vector, CM21. -/
theorem diagonal_density [DecidableEq ι] (w : ι → ℝ) :
    Density (diagonal fun i => (w i : ℂ)) ↔ Weights w := by
  constructor
  · rintro ⟨hp, ht⟩
    refine ⟨fun i => ?_, ?_⟩
    · simpa using (Complex.nonneg_iff.mp (hp.diag_nonneg (i := i))).1
    simpa using congrArg Complex.re ht
  · rintro ⟨hw, ht⟩
    refine ⟨Matrix.PosSemidef.diagonal (fun i => ?_), ?_⟩
    · exact Complex.nonneg_iff.mpr ⟨hw i, rfl⟩
    · simpa using congrArg (fun x : ℝ => (x : ℂ)) ht

/-- The full diagonal decomposition is an actual convex combination of pure
projectors, CM21. This algebraic identity has no positivity hypothesis. -/
theorem diagonal_decomposition [DecidableEq ι] (w : ι → ℝ) :
    diagonal (fun i => (w i : ℂ)) = ∑ i, w i • projector i := by
  ext u v
  simp only [Matrix.sum_apply, Matrix.smul_apply]
  by_cases huv : u = v
  · subst v
    simp [projector, Complex.real_smul, Pi.single_apply]
  · simp [projector, huv]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
