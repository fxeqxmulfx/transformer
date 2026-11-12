module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Fourier

/-!
# The lifted feasible set is the hull of pure Fourier projectors

Source: `review/flat_spectrum.md`, CM21. Mixed matrices are feasible; purity will
be a consequence of linear optimization, not a nonconvex domain constraint.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction
open scoped ComplexOrder

variable {q : ℕ} [NeZero q] {ι X : Type*} [Fintype ι] [DecidableEq ι]

/-- A single mode is an admitted PSD trace-one density, CM21. -/
theorem projector_feasible (r : X → ZMod q) (k : ι → ZMod q) (i : ι) :
    Feasible (basis r k) (projector i) := by
  have hw : Weights (Pi.single i (1 : ℝ)) := by
    constructor
    · intro j
      by_cases hj : j = i <;> simp [hj]
    · simp
  convert diagonal_feasible r k hw using 1
  unfold projector
  congr 1
  funext j
  by_cases hj : j = i <;> simp [hj]

/-- Exact separated Fourier geometry identifies the entire feasible convex
set, not only a chosen rank-one point, CM21. -/
theorem feasible_hull (r : X ≃ ZMod q) {k : ι → ZMod q} (hk : Separated k) :
    {M | Feasible (basis r k) M} = convexHull ℝ (Set.range (projector (ι := ι))) := by
  apply Set.Subset.antisymm
  · intro M hM
    obtain ⟨w, hw, rfl⟩ := (feasible_iff_diagonal r hk M).mp hM
    rw [diagonal_decomposition]
    exact (convex_convexHull ℝ _).sum_mem (fun i hi => hw.1 i) hw.2
      (fun i hi => subset_convexHull ℝ _ (Set.mem_range_self i))
  · exact convexHull_min (by rintro M ⟨i, rfl⟩; exact projector_feasible r k i)
      (feasible_convex _)

/-- The off-diagonal observable of a pure mode gives a relative phase, CM21. -/
theorem projector_reading (r : X → ZMod q) (k : ι → ZMod q) (i : ι) (x y : X) :
    reading (basis r k) (projector i) x y =
      ZMod.stdAddChar ((r x - r y) * k i) := by
  rw [projector, reading_diagonal]
  simp only [Pi.single_apply, ite_mul, one_mul, zero_mul, sum_ite_eq', mem_univ,
    ite_true, basis]
  rw [mul_comm, star_stdAddChar_mul, ← sub_mul]

/-- A mixed density has the corresponding average relative-phase reading,
CM21. Both optimization objectives are real linear functionals of this entry. -/
theorem diagonal_reading (r : X → ZMod q) (k : ι → ZMod q) (w : ι → ℝ) (x y : X) :
    reading (basis r k) (diagonal fun i => (w i : ℂ)) x y =
      ∑ i, (w i : ℂ) * ZMod.stdAddChar ((r x - r y) * k i) := by
  rw [reading_diagonal]
  apply sum_congr rfl
  intro i hi
  simp only [basis]
  rw [mul_comm (ZMod.stdAddChar (r x * k i)), star_stdAddChar_mul, ← sub_mul]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
