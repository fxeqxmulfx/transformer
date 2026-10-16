/-
# Positive target caps and geodesic convex hulls

ArXiv:2411.04551v3, §1.5, the definition of `conv_g`, and §3.2,
the target ball in `lem: induction.barycenter`. A positive cap remains
inside the same cap after taking the original geodesic convex hull.
Its Euclidean hull stays in both the cap half-space and the unit ball;
normalization therefore increases its positive axial height.

Consequently a chordal ball of radius at most one retains its radius
under the original hull operation. The proved chordal-to-angular bound
then gives a radius ensuring the source's angular accuracy. A compact
set in the positive hemisphere also supplies a uniform positive height
floor for the original gated perceptron phase.
-/

import Transformer.Interpolation.SeparationField

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original geodesic hull preserves every strictly positive cap.
Source: arXiv:2411.04551v3, §1.5 and §3.2, target cap geometry. -/
theorem convG_cap_lower (p : SSphere d) (A : Set (SSphere d)) (a : ℝ) (ha : 0 < a)
    (hA : ∀ x ∈ A, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d)) :
    ∀ x ∈ convG d A, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
  have hcap : convexHull ℝ ((↑) '' A : Set (EucSpace d)) ⊆
      {v : EucSpace d | a ≤ inner ℝ v (p : EucSpace d)} :=
    convexHull_min (by rintro _ ⟨x, hx, rfl⟩; exact hA x hx)
      (convex_halfSpace_ge ⟨fun x y => inner_add_left x y (p : EucSpace d),
        fun c x => real_inner_smul_left x (p : EucSpace d) c⟩ a)
  have hnorm : convexHull ℝ ((↑) '' A : Set (EucSpace d)) ⊆ Metric.closedBall 0 1 :=
    convexHull_min (by rintro _ ⟨x, hx, rfl⟩; exact mem_closedBall_zero_iff.mpr (mem_sphere_zero_iff_norm.mp x.2).le)
      (convex_closedBall 0 1)
  rintro x (hzero | ⟨v, hv, c, hc, hx⟩)
  · have hh : a ≤ inner ℝ (0 : EucSpace d) (p : EucSpace d) := hcap hzero
    rw [inner_zero_left] at hh
    exact (ha.not_ge hh).elim
  · have hvn : ‖v‖ ≤ 1 := mem_closedBall_zero_iff.mp (hnorm hv)
    have hxn := mem_sphere_zero_iff_norm.mp x.2
    rw [hx, norm_smul, Real.norm_eq_abs, abs_of_pos hc] at hxn
    have hcone : 1 ≤ c := by nlinarith [mul_le_mul_of_nonneg_left hvn hc.le]
    have hvi : a ≤ inner ℝ v (p : EucSpace d) := hcap hv
    rw [hx, real_inner_smul_left]
    exact hvi.trans (by simpa only [one_mul] using mul_le_mul_of_nonneg_right hcone (ha.le.trans hvi))

/-- Small chordal balls are preserved by the original geodesic hull.
Auxiliary for arXiv:2411.04551v3, §3.2, the final target ball. -/
theorem convG_chordal_radius (p : SSphere d) (A : Set (SSphere d)) (r : ℝ)
    (hr : 0 ≤ r) (hr1 : r ≤ 1) (hA : ∀ x ∈ A, dist x p ≤ r) :
    ∀ x ∈ convG d A, dist x p ≤ r := by
  have hn (x : SSphere d) : ‖(x : EucSpace d) - (p : EucSpace d)‖ ^ 2 =
      2 - 2 * inner ℝ (x : EucSpace d) (p : EucSpace d) := by
    rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp p.2]
    ring
  have ha : 0 < 1 - r ^ 2 / 2 := by nlinarith
  have hcap (x : SSphere d) (hx : x ∈ A) : 1 - r ^ 2 / 2 ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
    have hd : ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ r := hA x hx
    nlinarith [hn x, norm_nonneg ((x : EucSpace d) - p)]
  intro x hx
  have hh := convG_cap_lower p A (1 - r ^ 2 / 2) ha hcap x hx
  change ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ r
  nlinarith [hn x, norm_nonneg ((x : EucSpace d) - p)]

/-- The original geodesic hull is monotone under inclusion.
Source: arXiv:2411.04551v3, §1.5, the definition of `conv_g`. -/
theorem convG_mono {A B : Set (SSphere d)} (hAB : A ⊆ B) : convG d A ⊆ convG d B := by
  have hh : convexHull ℝ ((↑) '' A : Set (EucSpace d)) ⊆ convexHull ℝ ((↑) '' B) :=
    convexHull_mono (image_mono hAB)
  rintro x (hzero | ⟨v, hv, c, hc, hx⟩)
  · exact Or.inl (hh hzero)
  · exact Or.inr ⟨v, hh hv, c, hc, hx⟩

/-- The original geodesic hull of one unit point is that same point.
Source: arXiv:2411.04551v3, §1.5, the definition of `conv_g`. -/
theorem convG_singleton (p : SSphere d) : convG d {p} = {p} := by
  apply Set.Subset.antisymm
  · intro x hx
    have hd : dist x p ≤ 0 := convG_chordal_radius p {p} 0 (by norm_num) (by norm_num)
      (by intro y hy; simp only [mem_singleton_iff] at hy; subst y; simp) x hx
    exact mem_singleton_iff.mpr (dist_eq_zero.mp (le_antisymm hd dist_nonneg))
  · exact subset_convG d {p}

/-- One positive chordal tolerance ensures the source angular tolerance
for every original hull point. Source: arXiv:2411.04551v3, §3.2. -/
theorem exists_chordal_radius_for_geodesic_hull (p : SSphere d) (ε : ℝ) (hε : 0 < ε) :
    ∃ r : ℝ, 0 < r ∧ ∀ A : Set (SSphere d), (∀ x ∈ A, dist x p ≤ r) →
      ∀ x ∈ convG d A, arccos (inner ℝ (x : EucSpace d) (p : EucSpace d)) < ε := by
  let r := min 1 (ε / Real.pi)
  have hr : 0 < r := lt_min zero_lt_one (div_pos hε pi_pos)
  refine ⟨r, hr, fun A hA x hx => ?_⟩
  have hd : ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ r :=
    convG_chordal_radius p A r hr.le (min_le_left _ _) hA x hx
  have hc := arccos_inner_le_norm_sub d (mem_sphere_zero_iff_norm.mp x.2) (mem_sphere_zero_iff_norm.mp p.2)
  have hm : Real.pi * r ≤ ε := by
    simpa only [mul_comm] using (le_div_iff₀ pi_pos).mp (min_le_right 1 (ε / Real.pi))
  nlinarith [mul_le_mul_of_nonneg_left hd (div_nonneg pi_pos.le (by norm_num : (0 : ℝ) ≤ 2))]

/-- A singleton hull has zero angular distance to its own unit point.
Auxiliary for arXiv:2411.04551v3, §3.2, the one-dimensional case. -/
theorem arccos_on_convG_singleton (p : SSphere d) (x : SSphere d) (hx : x ∈ convG d {p}) :
    arccos (inner ℝ (x : EucSpace d) (p : EucSpace d)) = 0 := by
  rw [convG_singleton, mem_singleton_iff] at hx
  subst x
  rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2, one_pow, arccos_one]

/-- Compact positive-hemisphere data have a uniform positive axial
height. Auxiliary for arXiv:2411.04551v3, §3.2, the active perceptron phase. -/
theorem exists_positive_height_floor (p : SSphere d) (K : Set (SSphere d))
    (hK : IsCompact K) (hne : K.Nonempty)
    (hpos : ∀ x ∈ K, 0 < inner ℝ (x : EucSpace d) (p : EucSpace d)) :
    ∃ a : ℝ, 0 < a ∧ ∀ x ∈ K, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
  have hc : Continuous (fun x : SSphere d => inner ℝ (x : EucSpace d) (p : EucSpace d)) :=
    continuous_subtype_val.inner continuous_const
  obtain ⟨x, hx, hm⟩ := hK.exists_isMinOn hne hc.continuousOn
  exact ⟨inner ℝ (x : EucSpace d) (p : EucSpace d), hpos x hx, fun y hy => hm hy⟩

/-- A nonempty actual spherical singleton and its unit pole jointly
realize every cap, radius, inclusion, compactness and strict-height
hypothesis. The angular accuracy is positive. -/
example : ∃ (p : SSphere 2) (A B K : Set (SSphere 2)) (a r : ℝ),
    0 < a ∧ 0 ≤ r ∧ r ≤ 1 ∧ A ⊆ B ∧ IsCompact K ∧ K.Nonempty ∧
    (∀ x ∈ A, a ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2)) ∧
    (∀ x ∈ A, dist x p ≤ r) ∧
    (∀ x ∈ K, 0 < inner ℝ (x : EucSpace 2) (p : EucSpace 2)) ∧ (0 : ℝ) < 1 := by
  let p : SSphere 2 := basePoint 1
  have hp : inner ℝ (p : EucSpace 2) (p : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  refine ⟨p, {p}, univ, {p}, 1 / 2, 1 / 2, by norm_num, by norm_num, by norm_num,
    subset_univ _, isCompact_singleton, ⟨p, rfl⟩, ?_, ?_, ?_, zero_lt_one⟩
  · intro x hx
    rw [mem_singleton_iff] at hx
    subst x
    rw [hp]
    norm_num
  · intro x hx
    rw [mem_singleton_iff] at hx
    subst x
    rw [dist_self]
    norm_num
  · intro x hx
    rw [mem_singleton_iff] at hx
    subst x
    rw [hp]
    exact zero_lt_one

end Transformer.Interpolation
