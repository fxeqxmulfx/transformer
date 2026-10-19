module

public import Transformer.Perspective.UniformAtomless
public import Mathlib.MeasureTheory.Integral.Marginal

/-!
# Sets of tuples with thin slices are null for the uniform law

`Perspective.UniformTuple` is the product `σ^{⊗ n}` of a rotation-invariant
law, which has no atoms once `d ≥ 2` (`Perspective.SphereInvariant`). By
Fubini, a measurable set of tuples is then null as soon as its slices along
one fixed coordinate — the positions of that particle with all the others
frozen — have at most one point each (`measure_eq_zero_of_slices`).

This is how the null sets of `l:nosaddleconv` (arXiv:2312.10794v5,
Appendix A) are recognized: the tuples with vanishing barycentre
(`measure_sum_eq_zero`) here, and the local stable set of a saddle in
`Perspective.AppendixA_Cone`.
-/

@[expose] public section

open scoped BigOperators ENNReal
open MeasureTheory

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- A point of `𝕊^{d-1}` once `d ≥ 1`: the first basis vector. -/
noncomputable def firstBasisPoint (hd : 1 ≤ d) : SSphere d :=
  ⟨EuclideanSpace.single (⟨0, hd⟩ : Fin d) (1 : ℝ), by simp⟩

/-- **Thin slices make a null set.**  If along the coordinate `i₀` every slice
`{x | X[i₀ ↦ x] ∈ A}` of a measurable set `A` of tuples has at most one point,
then `A` is null for the uniform law (`d ≥ 2`).

Fubini peels off the integral over the `i₀`-th particle first; it is the mass
of a slice, which is `0` since `σ` has no atoms. -/
theorem measure_eq_zero_of_slices (hd : 2 ≤ d) (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) {A : Set (SphereTuple d n)} (hA : MeasurableSet A) (i₀ : Idx n)
    (hslice : ∀ X : SphereTuple d n,
      {x : SSphere d | Function.update X i₀ x ∈ A}.Subsingleton) :
    P A = 0 := by
  classical
  obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
  have := hσprob
  have hf : Measurable (A.indicator (1 : SphereTuple d n → ℝ≥0∞)) :=
    measurable_one.indicator hA
  rw [← lintegral_indicator_one hA,
    lintegral_eq_lmarginal_univ (fun _ => firstBasisPoint (by omega))]
  rw [← Finset.insert_erase (Finset.mem_univ i₀),
    lmarginal_insert' _ hf (Finset.notMem_erase i₀ _)]
  have hzero : (fun X : SphereTuple d n =>
      ∫⁻ x, A.indicator 1 (Function.update X i₀ x) ∂σ) = fun _ => 0 := by
    funext X
    refine le_antisymm ?_ zero_le
    have hind : (fun x : SSphere d => A.indicator (1 : SphereTuple d n → ℝ≥0∞)
        (Function.update X i₀ x))
        = {x : SSphere d | Function.update X i₀ x ∈ A}.indicator 1 := by
      funext x
      simp only [Set.indicator, Set.mem_ofPred_eq, Pi.one_apply]
    rw [hind]
    refine (lintegral_indicator_one_le _).trans (le_of_eq ?_)
    rcases (hslice X).eq_empty_or_singleton with h | ⟨x, h⟩
    · rw [h, measure_empty]
    · rw [h]
      exact measure_singleton_eq_zero_of_invariant d hd σ hσinv x
  rw [hzero]
  simp [lmarginal]

/-- The hypotheses of `measure_eq_zero_of_slices` are satisfiable: `d = 2` and
the empty set, whose slices are empty; the uniform law is quantified over. -/
example : 2 ≤ 2 ∧ ∀ X : SphereTuple 2 1,
    {x : SSphere 2 | Function.update X 0 x ∈ (∅ : Set (SphereTuple 2 1))}.Subsingleton :=
  ⟨le_rfl, fun _ => by simp⟩

/-- The sum of a tuple with one particle replaced, split off at that particle. -/
theorem sum_update (X : SphereTuple d n) (i₀ : Idx n) (z : SSphere d) :
    ∑ j : Idx n, (Function.update X i₀ z j : EucSpace d)
      = (z : EucSpace d) + ∑ j ∈ Finset.univ.erase i₀, (X j : EucSpace d) := by
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ i₀), Function.update_self]
  congr 1
  exact Finset.sum_congr rfl fun j hj => by
    rw [Function.update_of_ne (Finset.ne_of_mem_erase hj)]

/-- **The tuples with vanishing barycentre are null** for the uniform law: the
slice along the first particle is `{-Σ_{j ≠ 0} x_j}` or empty.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv` (the case `s = 0`). -/
theorem measure_sum_eq_zero (hd : 2 ≤ d) (hn : 1 ≤ n) (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) :
    P {X : SphereTuple d n | ∑ j : Idx n, (X j : EucSpace d) = 0} = 0 := by
  have hcont : Continuous fun X : SphereTuple d n => ∑ j : Idx n, (X j : EucSpace d) :=
    continuous_finsetSum _ fun j _ => continuous_subtype_val.comp (continuous_apply j)
  refine measure_eq_zero_of_slices hd P hP
    (isClosed_eq hcont continuous_const).measurableSet ⟨0, hn⟩ fun X x hx y hy => ?_
  simp only [Set.mem_ofPred_eq, sum_update] at hx hy
  exact Subtype.ext (add_right_cancel (hx.trans hy.symm))

/-- The hypotheses of `measure_sum_eq_zero` are satisfiable: `d = 2`, `n = 1`;
the uniform law is quantified over. -/
example : 2 ≤ 2 ∧ 1 ≤ 1 := ⟨le_rfl, le_rfl⟩

/-- **Two coincident particles form a null set** for the uniform law: a finite
union of the null diagonals of `measure_coords_eq_eq_zero`.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv` (two particles
opposite to the barycentre). -/
theorem measure_exists_coords_eq (hd : 2 ≤ d) (P : Measure (SphereTuple d n))
    (hP : UniformTuple d n P) :
    P {X : SphereTuple d n | ∃ i j : Idx n, i ≠ j ∧ X i = X j} = 0 := by
  have hset : {X : SphereTuple d n | ∃ i j : Idx n, i ≠ j ∧ X i = X j}
      = ⋃ i : Idx n, ⋃ j : Idx n, ⋃ (_ : i ≠ j), {X : SphereTuple d n | X i = X j} := by
    ext X
    simp
  rw [hset, measure_iUnion_null_iff]
  intro i
  rw [measure_iUnion_null_iff]
  intro j
  rw [measure_iUnion_null_iff]
  exact fun hij => measure_coords_eq_eq_zero d n hd P hP hij

/-- The hypothesis of `measure_exists_coords_eq` is satisfiable: `d = 2`; the
uniform law is quantified over. -/
example : 2 ≤ 2 := le_rfl

end Perspective
end Transformer
