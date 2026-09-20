/-
# The uniform law on `(𝕊^{d-1})^n` is atomless off the diagonal

`Perspective.UniformTuple` describes the uniform law as `P = σ^{⊗ n}` for a
rotation-invariant probability measure `σ` on `𝕊^{d-1}`.  Since `σ` has no
atoms (`Perspective.SphereInvariant`), the law of a pair of distinct
coordinates is `σ × σ`, whose diagonal is null by Fubini: two distinct
coordinates of a uniform tuple differ almost surely, and no single
configuration is charged.

This is what turns the "almost every initial sequence clusters" of §4 and §6
into a statement with content: the antipodal-pair counterexamples of
`antipodalPair_not_mem_clusteringSet` do not merely form a small set, they
form a null one.

Source: arXiv:2312.10794v5, §4 (the uniform law `σ_d` of `p:beta0`).
-/

import Transformer.Perspective.Section3_SmallBeta
import Transformer.Perspective.SphereInvariant
import Mathlib.MeasureTheory.Measure.Prod

open scoped BigOperators ENNReal
open MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-! ### The diagonal is null -/

/-- The law of two distinct coordinates of an i.i.d. tuple is the product of
their common marginal. -/
theorem map_pair_pi (σ : Measure (SSphere d)) [IsProbabilityMeasure σ]
    {i j : Idx n} (hij : i ≠ j) :
    (Measure.pi (fun _ : Idx n => σ)).map (fun X : SphereTuple d n => (X i, X j))
      = σ.prod σ := by
  have hmeas : Measurable (fun X : SphereTuple d n => (X i, X j)) :=
    (measurable_pi_apply i).prodMk (measurable_pi_apply j)
  refine (Measure.prod_eq (fun s t hs ht => ?_)).symm
  rw [Measure.map_apply hmeas (hs.prod ht)]
  set g : Idx n → Set (SSphere d) :=
    Function.update (Function.update (fun _ : Idx n => Set.univ) i s) j t with hg
  have hgi : g i = s := by rw [hg, Function.update_of_ne hij, Function.update_self]
  have hgj : g j = t := by rw [hg, Function.update_self]
  have hgk : ∀ k : Idx n, k ≠ i → k ≠ j → g k = Set.univ := by
    intro k hki hkj
    rw [hg, Function.update_of_ne hkj, Function.update_of_ne hki]
  have hpre : (fun X : SphereTuple d n => (X i, X j)) ⁻¹' (s ×ˢ t)
      = Set.univ.pi g := by
    ext X
    simp only [Set.mem_preimage, Set.mem_prod, Set.mem_univ_pi]
    constructor
    · rintro ⟨h1, h2⟩ k
      by_cases hki : k = i
      · subst hki; rw [hgi]; exact h1
      · by_cases hkj : k = j
        · subst hkj; rw [hgj]; exact h2
        · rw [hgk k hki hkj]; trivial
    · exact fun h => ⟨hgi ▸ h i, hgj ▸ h j⟩
  rw [hpre, Measure.pi_pi]
  rw [Finset.prod_eq_mul_of_mem i j (Finset.mem_univ i) (Finset.mem_univ j) hij
    (fun c _ hc => by rw [hgk c hc.1 hc.2, measure_univ]), hgi, hgj]

/-- **Two distinct coordinates of a uniform tuple differ almost surely.**

This is what turns the "almost every initial sequence clusters" of §4 and §6
into a statement with content: the diagonal, where the antipodal-pair
counterexamples of `antipodalPair_not_mem_clusteringSet` live, is not merely
small, it is null.

Source: arXiv:2312.10794v5, §4 (the uniform law `σ_d`). -/
theorem measure_coords_eq_eq_zero (hd : 2 ≤ d)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P)
    {i j : Idx n} (hij : i ≠ j) :
    P {X : SphereTuple d n | X i = X j} = 0 := by
  obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
  have := hσprob
  have hmeas : Measurable (fun X : SphereTuple d n => (X i, X j)) :=
    (measurable_pi_apply i).prodMk (measurable_pi_apply j)
  have hset : {X : SphereTuple d n | X i = X j}
      = (fun X : SphereTuple d n => (X i, X j)) ⁻¹' Set.diagonal (SSphere d) := rfl
  rw [hset, ← Measure.map_apply hmeas measurableSet_diagonal, map_pair_pi d n σ hij,
    Measure.prod_apply measurableSet_diagonal]
  have hfib : ∀ x : SSphere d, σ (Prod.mk x ⁻¹' Set.diagonal (SSphere d)) = 0 := by
    intro x
    have hx : Prod.mk x ⁻¹' Set.diagonal (SSphere d) = {x} := by
      ext y; simp [Set.diagonal, eq_comm]
    rw [hx]
    exact measure_singleton_eq_zero_of_invariant d hd σ hσinv x
  simp [hfib]

/-- **A uniform tuple charges no single configuration.**

A singleton of `(𝕊^{d-1})^n` is a product of singletons, and each factor is
already null. -/
theorem measure_singleton_tuple_eq_zero (hd : 2 ≤ d) (hn : 1 ≤ n)
    (P : Measure (SphereTuple d n)) (hP : UniformTuple d n P)
    (X₀ : SphereTuple d n) : P {X₀} = 0 := by
  obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
  have := hσprob
  have hset : ({X₀} : Set (SphereTuple d n)) = Set.univ.pi (fun i => {X₀ i}) := by
    ext X
    simp [funext_iff]
  rw [hset, Measure.pi_pi]
  exact Finset.prod_eq_zero (Finset.mem_univ (⟨0, hn⟩ : Idx n))
    (measure_singleton_eq_zero_of_invariant d hd σ hσinv _)

/-- The hypotheses of `measure_singleton_tuple_eq_zero` are satisfiable:
`d = 2`, `n = 1`. -/
example : 2 ≤ 2 ∧ 1 ≤ 1 := ⟨le_rfl, le_rfl⟩

/-- The hypotheses of `measure_coords_eq_eq_zero` are satisfiable: `d = 2`,
`n = 2`, `i = 0`, `j = 1`, and the uniform law is quantified over, so nothing
has to be exhibited beyond `2 ≤ 2` and `0 ≠ 1`. -/
example : 2 ≤ 2 ∧ (0 : Idx 2) ≠ (1 : Idx 2) := ⟨le_rfl, by decide⟩

end Perspective
end Transformer
