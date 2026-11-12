module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Hull

/-!
# A supplied subspace basis may be arbitrarily rotated

Source: `review/flat_spectrum.md`, CM21. The exact-subspace assumption is a
unitary change of coordinates from permuted Fourier features. The unitary
matrix and permutation are witnesses in the proof, not algorithm inputs.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction
open scoped ComplexOrder

variable {ι X : Type*} [Fintype ι] [DecidableEq ι]

/-- Unitary conjugation preserves the actual PSD and trace conditions, CM21. -/
theorem density_conjugate (Q M : Matrix ι ι ℂ) (hQ : Q.conjTranspose * Q = 1)
    (hM : Density M) : Density (Q * M * Q.conjTranspose) := by
  refine ⟨hM.1.mul_mul_conjTranspose_same Q, ?_⟩
  rw [Matrix.trace_mul_cycle, hQ, Matrix.one_mul, hM.2]

omit [DecidableEq ι] in
/-- Coefficient conjugation exactly accounts for a rotated feature basis,
CM21. This concerns every kernel entry, not just the diagonal. -/
theorem reading_change (B : Matrix X ι ℂ) (Q M : Matrix ι ι ℂ) (x y : X) :
    reading (B * Q) M x y = reading B (Q * M * Q.conjTranspose) x y := by
  simp only [reading, Matrix.conjTranspose_mul, Matrix.mul_assoc]

/-- Conjugation has an explicit inverse under the stated unitary identity,
CM21. No rank-one factorization is assumed. -/
theorem undo_conjugate (Q M : Matrix ι ι ℂ) (hQ : Q.conjTranspose * Q = 1) :
    Q.conjTranspose * (Q * M * Q.conjTranspose) * Q = M := by
  calc
    _ = (Q.conjTranspose * Q) * M * (Q.conjTranspose * Q) := by
      simp only [Matrix.mul_assoc]
    _ = M := by rw [hQ, Matrix.one_mul, Matrix.mul_one]

/-- The known-basis convex problem is equivalent to the hidden Fourier-basis
problem, CM21. The hypothesis is exact equality of feature bases. -/
theorem feasible_change (B : Matrix X ι ℂ) (Q M : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1) :
    Feasible (B * Q) M ↔ Feasible B (Q * M * Q.conjTranspose) := by
  constructor
  · intro hM
    exact ⟨density_conjugate Q M hQ hM.1, fun x => by
      rw [← reading_change]; exact hM.2 x⟩
  · intro hM
    refine ⟨?_, fun x => by rw [reading_change]; exact hM.2 x⟩
    have hd := density_conjugate Q.conjTranspose _ (by simpa using hQ') hM.1
    simpa only [Matrix.conjTranspose_conjTranspose, undo_conjugate Q M hQ] using hd

/-- A rotated pure projector is feasible in the supplied basis, CM21. -/
theorem rotated_projector_feasible {q : ℕ} [NeZero q] (r : X → ZMod q)
    (k : ι → ZMod q) (Q : Matrix ι ι ℂ) (hQ : Q.conjTranspose * Q = 1)
    (hQ' : Q * Q.conjTranspose = 1) (i : ι) :
    Feasible (basis r k * Q) (Q.conjTranspose * projector i * Q) := by
  apply (feasible_change _ Q _ hQ hQ').mpr
  have hu := undo_conjugate Q.conjTranspose (projector i) (by simpa using hQ')
  have he : Q * (Q.conjTranspose * projector i * Q) * Q.conjTranspose = projector i := by
    simpa using hu
  rw [he]
  exact projector_feasible r k i

/-- Every feasible matrix in a supplied exact subspace is a convex combination
of the rotated mode projectors, CM21. Both directions retain all mixtures. -/
theorem rotated_hull {q : ℕ} [NeZero q] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (Q : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1) :
    {M | Feasible (basis r k * Q) M} =
      convexHull ℝ (Set.range fun i => Q.conjTranspose * projector i * Q) := by
  apply Set.Subset.antisymm
  · intro M hM
    obtain ⟨w, hw, he⟩ := (feasible_iff_diagonal r hk _).mp
      ((feasible_change _ Q M hQ hQ').mp hM)
    have hm : M = ∑ i, w i • (Q.conjTranspose * projector i * Q) := by
      rw [← undo_conjugate Q M hQ, he, diagonal_decomposition]
      simp only [Matrix.mul_sum, Matrix.sum_mul, Matrix.mul_smul, Matrix.smul_mul]
    rw [hm]
    exact (convex_convexHull ℝ _).sum_mem (fun i hi => hw.1 i) hw.2
      (fun i hi => subset_convexHull ℝ _ (Set.mem_range_self i))
  · exact convexHull_min (by
      rintro M ⟨i, rfl⟩
      exact rotated_projector_feasible r k Q hQ hQ' i) (feasible_convex _)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
