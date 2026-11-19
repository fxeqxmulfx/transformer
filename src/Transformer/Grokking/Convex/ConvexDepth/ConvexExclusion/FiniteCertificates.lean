module

public import Transformer.Grokking.Convex.ConvexDepth.ConvexExclusion.Basic

/-!
# Finite certificates for convex hull intersections

Source: `review/convex/moments.md`, CM50. Two nonnegative finite mixtures of
original coordinates certify the common point. The candidate mixture contains
the explicitly tagged query with positive weight. The other mixture contains
only observations of a different raw label. No unknown answer is an input.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ConvexExclusion

open RawRegions
open scoped BigOperators

variable {E I C : Type*} [AddCommGroup E] [Module ℝ E] [Fintype I]

/-- A query coordinate and a finite observed mixture lie in the augmented
candidate hull when their nonnegative masses sum to one, CM50. -/
theorem augmented_barycenter_mem_hull (point : I → E) (labels : I → C)
    (x : E) (c : C) (a : ℝ) (w : I → ℝ)
    (ha : 0 ≤ a) (hw : ∀ i, 0 ≤ w i) (hs : a + ∑ i, w i = 1)
    (hlabels : ∀ i, w i ≠ 0 → labels i = c) :
    a • x + ∑ i, w i • point i ∈
      convexHull ℝ (insert x (observedClass point labels c)) := by
  let weight : Option I → ℝ := fun i => i.elim a w
  let coordinate : Option I → E := fun i => i.elim x point
  have hnonneg : ∀ i, 0 ≤ weight i := by
    rintro (_ | i)
    · exact ha
    · exact hw i
  have hmass : (∑ᶠ i, weight i) = 1 := by
    simpa [finsum_eq_sum_of_fintype, Fintype.sum_option, weight] using hs
  have hmem : ∀ i, weight i ≠ 0 → coordinate i ∈
      convexHull ℝ (insert x (observedClass point labels c)) := by
    rintro (_ | i) hi
    · apply subset_convexHull
      exact Or.inl rfl
    · apply subset_convexHull
      exact Or.inr ⟨i, hlabels i hi, rfl⟩
  have h := (convex_convexHull ℝ (insert x (observedClass point labels c))).finsum_mem
    hnonneg hmass hmem
  simpa [finsum_eq_sum_of_fintype, Fintype.sum_option, weight, coordinate] using h

/-- Matching finite barycenters certify a CM49 exclusion, CM50. The query
has positive mass only on the hypothetical candidate side; both label supports
are checked against original observations, and the conflicting labels differ. -/
theorem excluded_of_barycenters (point : I → E) (labels : I → C)
    (x : E) (c d : C) (a : ℝ) (w v : I → ℝ)
    (hdc : d ≠ c) (ha : 0 < a) (hw : ∀ i, 0 ≤ w i) (hv : ∀ i, 0 ≤ v i)
    (hw_sum : a + ∑ i, w i = 1) (hv_sum : ∑ i, v i = 1)
    (hw_labels : ∀ i, w i ≠ 0 → labels i = c)
    (hv_labels : ∀ i, v i ≠ 0 → labels i = d)
    (hmatch : ∑ i, v i • point i = a • x + ∑ i, w i • point i) :
    Excluded point labels x c := by
  have hd := barycenter_mem_hull point labels v hv hv_sum hv_labels
  have hc := augmented_barycenter_mem_hull point labels x c a w
    (le_of_lt ha) hw hw_sum hw_labels
  refine ⟨d, ∑ i, v i • point i, hdc, hd, ?_⟩
  rw [hmatch]
  exact hc

-- Both conditional certificate theorems have concrete positive-mass instances.
example : (-3 : ℝ) ∈
    convexHull ℝ (insert (-4) (observedClass regionPoints regionLabels 2)) := by
  have h := augmented_barycenter_mem_hull regionPoints regionLabels (-4) 2 (4 / 5)
    ![0, 0, 1 / 5, 0] (by norm_num)
    (by intro i; fin_cases i <;> norm_num)
    (by norm_num [Fin.sum_univ_four])
    (by intro i hi; fin_cases i <;> norm_num [regionLabels] at *)
  convert h using 1
  norm_num [Fin.sum_univ_four, regionPoints]

example : Excluded regionPoints regionLabels (-4) 2 := by
  apply excluded_of_barycenters regionPoints regionLabels (-4) 2 0 (4 / 5)
    ![0, 0, 1 / 5, 0] ![1, 0, 0, 0]
  · decide
  · norm_num
  · intro i
    fin_cases i <;> norm_num
  · intro i
    fin_cases i <;> norm_num
  · norm_num [Fin.sum_univ_four]
  · norm_num [Fin.sum_univ_four]
  · intro i hi
    fin_cases i <;> norm_num [regionLabels] at *
  · intro i hi
    fin_cases i <;> norm_num [regionLabels] at *
  · norm_num [regionPoints, Fin.sum_univ_four]

end Transformer.Grokking.Convex.ConvexDepth.ConvexExclusion
