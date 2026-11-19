module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion.FiniteCertificates

/-!
# Planar exclusions missed by an observed-point test

Source: `review/convex/moments.md`, CM50. Two vertical observed segments are
disjoint. Adding an exterior query to the opposite class makes a triangle
whose edges cross the other segment while containing neither of its endpoints.
The exact rational mixtures certify the crossing without a missing answer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion

open RawRegions
open scoped BigOperators

/-- Four original planar inputs, with the existing labels `(0,0,2,2)`. -/
def crossingPoints : Fin 4 → ℝ × ℝ := ![(-1, -1), (-1, 1), (1, -1), (1, 1)]

/-- Linear coordinate halfspaces used to bound the actual hulls, CM50. -/
theorem coordinate_halfspace_convex (a b r : ℝ) :
    Convex ℝ {p : ℝ × ℝ | r ≤ a * p.1 + b * p.2} := by
  apply convex_halfSpace_ge
  constructor
  · intro x y
    change a * (x.1 + y.1) + b * (x.2 + y.2) =
      (a * x.1 + b * x.2) + (a * y.1 + b * y.2)
    ring
  · intro t x
    change a * (t * x.1) + b * (t * x.2) = t * (a * x.1 + b * x.2)
    ring

/-- The left crossing is witnessed by two distinct positive mixtures,
CM50. Neither side's barycenter is an original observation. -/
theorem left_crossing_excluded : Excluded crossingPoints regionLabels (-2, 0) 2 := by
  apply excluded_of_barycenters crossingPoints regionLabels (-2, 0) 2 0 (2 / 3)
    ![0, 0, 1 / 3, 0] ![2 / 3, 1 / 3, 0, 0]
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
  · ext <;> norm_num [crossingPoints, Fin.sum_univ_four]

/-- The right exterior query has the symmetric two-sided certificate, CM50. -/
theorem right_crossing_excluded : Excluded crossingPoints regionLabels (2, 0) 0 := by
  apply excluded_of_barycenters crossingPoints regionLabels (2, 0) 0 2 (2 / 3)
    ![1 / 3, 0, 0, 0] ![0, 0, 2 / 3, 1 / 3]
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
  · ext <;> norm_num [crossingPoints, Fin.sum_univ_four]

/-- The left augmented triangle obeys both oblique edge bounds, CM50. -/
theorem left_crossing_bounds (z : ℝ × ℝ)
    (hz : z ∈ convexHull ℝ (insert (-2, 0) (observedClass crossingPoints regionLabels 2))) :
    -2 ≤ z.1 + 3 * z.2 ∧ -2 ≤ z.1 - 3 * z.2 := by
  have bound (k : ℝ) (hk : k = 3 ∨ k = -3) : -2 ≤ z.1 + k * z.2 := by
    have h := convexHull_min (t := {p : ℝ × ℝ | -2 ≤ 1 * p.1 + k * p.2})
      (by
        rintro u (rfl | ⟨i, hi, rfl⟩)
        · norm_num
        · rcases hk with rfl | rfl <;> fin_cases i <;>
            norm_num [crossingPoints, regionLabels] at *)
      (coordinate_halfspace_convex 1 k (-2)) hz
    simpa using h
  exact ⟨bound 3 (Or.inl rfl), by linarith [bound (-3) (Or.inr rfl)]⟩

/-- The right augmented triangle obeys the reflected bounds, CM50. -/
theorem right_crossing_bounds (z : ℝ × ℝ)
    (hz : z ∈ convexHull ℝ (insert (2, 0) (observedClass crossingPoints regionLabels 0))) :
    -2 ≤ -z.1 + 3 * z.2 ∧ -2 ≤ -z.1 - 3 * z.2 := by
  have bound (k : ℝ) (hk : k = 3 ∨ k = -3) : -2 ≤ -z.1 + k * z.2 := by
    have h := convexHull_min (t := {p : ℝ × ℝ | -2 ≤ (-1) * p.1 + k * p.2})
      (by
        rintro u (rfl | ⟨i, hi, rfl⟩)
        · norm_num
        · rcases hk with rfl | rfl <;> fin_cases i <;>
            norm_num [crossingPoints, regionLabels] at *)
      (coordinate_halfspace_convex (-1) k (-2)) hz
    simpa using h
  exact ⟨bound 3 (Or.inl rfl), by linarith [bound (-3) (Or.inr rfl)]⟩

/-- No opposite-label observed point certifies the left exclusion, CM50.
The genuine intersection established above therefore strengthens that test. -/
theorem left_no_observed_conflict : ¬ ∃ i, regionLabels i ≠ 2 ∧
    crossingPoints i ∈ convexHull ℝ
      (insert (-2, 0) (observedClass crossingPoints regionLabels 2)) := by
  rintro ⟨i, hi, hm⟩
  have hb := left_crossing_bounds (crossingPoints i) hm
  fin_cases i <;> norm_num [crossingPoints, regionLabels] at *

/-- The observed-point test also misses the symmetric right exclusion. -/
theorem right_no_observed_conflict : ¬ ∃ i, regionLabels i ≠ 0 ∧
    crossingPoints i ∈ convexHull ℝ
      (insert (2, 0) (observedClass crossingPoints regionLabels 0)) := by
  rintro ⟨i, hi, hm⟩
  have hb := right_crossing_bounds (crossingPoints i) hm
  fin_cases i <;> norm_num [crossingPoints, regionLabels] at *

/-- Every observed class hull stays between the two observed vertical lines. -/
theorem crossing_hull_first_bounds (c : Fin 4) (z : ℝ × ℝ)
    (hz : z ∈ convexHull ℝ (observedClass crossingPoints regionLabels c)) :
    -1 ≤ z.1 ∧ z.1 ≤ 1 := by
  have lower := convexHull_min (t := {p : ℝ × ℝ | -1 ≤ 1 * p.1 + 0 * p.2})
    (by rintro u ⟨i, _, rfl⟩; fin_cases i <;> norm_num [crossingPoints])
    (coordinate_halfspace_convex 1 0 (-1)) hz
  have upper := convexHull_min (t := {p : ℝ × ℝ | -1 ≤ (-1) * p.1 + 0 * p.2})
    (by rintro u ⟨i, _, rfl⟩; fin_cases i <;> norm_num [crossingPoints])
    (coordinate_halfspace_convex (-1) 0 (-1)) hz
  constructor <;> simp_all

/-- Both exterior queries lie outside every observed raw-class hull, CM50. -/
theorem crossing_queries_uncovered (c : Fin 4) :
    (-2, 0) ∉ convexHull ℝ (observedClass crossingPoints regionLabels c) ∧
      (2, 0) ∉ convexHull ℝ (observedClass crossingPoints regionLabels c) := by
  constructor
  · intro h
    have hb := crossing_hull_first_bounds c (-2, 0) h
    norm_num at hb
  · intro h
    have hb := crossing_hull_first_bounds c (2, 0) h
    norm_num at hb

-- Nonempty geometric examples for the conditional hull bounds.
example := left_crossing_bounds (-2, 0) (subset_convexHull ℝ _ (Or.inl rfl))
example := right_crossing_bounds (2, 0) (subset_convexHull ℝ _ (Or.inl rfl))
example := crossing_hull_first_bounds 0 (crossingPoints 0)
  (subset_convexHull ℝ _ ⟨(0 : Fin 4), rfl, rfl⟩)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ConvexExclusion
