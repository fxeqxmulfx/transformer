module

public import Transformer.Interpolation.Section5_BallSweep
public import Mathlib.Analysis.SpecialFunctions.Complex.Arg

/-!
# The unit circle in angle coordinates

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, on the circle
(`d = 2`), where the pieces of the supports are arcs. The point at angle
`s` is `(cos s, sin s)`, the great circle of `Section5_BallSweep` through
the two unit axes. Every point of the circle has an angle in each window
of length `2π`, angles closer than `2π` give distinct points, and a ball
is the arc of the angles closer to the angle of its center than its
angular radius. Hence a metric sphere of the circle has at most two
points, and a law without atoms gives it no mass.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

/-- The first unit axis of the plane. -/
noncomputable def axisX : SSphere 2 := ⟨EuclideanSpace.single 0 1, by simp⟩

/-- The second unit axis of the plane. -/
noncomputable def axisY : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩

/-- The two unit axes are orthogonal. -/
theorem inner_axisX_axisY : inner (𝕜 := ℝ) (axisX : EucSpace 2) (axisY : EucSpace 2) = 0 := by
  simp [axisX, axisY, EuclideanSpace.inner_single_left]

/-- The point of the circle at angle `s`, `(cos s, sin s)`. Source
auxiliary: arXiv:2411.04551v3, §2.2, the arcs of the circle. -/
noncomputable def circlePoint (s : ℝ) : SSphere 2 := sweepPoint axisX axisY inner_axisX_axisY s

theorem circlePoint_apply_zero (s : ℝ) : (circlePoint s : EucSpace 2) 0 = Real.cos s := by
  simp [circlePoint, sweepPoint, Perceptron.greatCircle, axisX, axisY]

theorem circlePoint_apply_one (s : ℝ) : (circlePoint s : EucSpace 2) 1 = Real.sin s := by
  simp [circlePoint, sweepPoint, Perceptron.greatCircle, axisX, axisY]

theorem continuous_circlePoint : Continuous circlePoint :=
  continuous_sweepPoint axisX axisY inner_axisX_axisY

/-- The inner product of two points of the circle is the cosine of the
difference of their angles. -/
theorem inner_circlePoint (s t : ℝ) :
    inner (𝕜 := ℝ) (circlePoint s : EucSpace 2) (circlePoint t : EucSpace 2) = Real.cos (s - t) :=
  inner_sweepPoint axisX axisY inner_axisX_axisY s t

theorem dist_circlePoint_sq (s t : ℝ) :
    dist (circlePoint s) (circlePoint t) ^ 2 = 2 - 2 * Real.cos (s - t) :=
  dist_sweepPoint_sq axisX axisY inner_axisX_axisY s t

theorem dist_circlePoint_le (s t : ℝ) : dist (circlePoint s) (circlePoint t) ≤ |s - t| :=
  dist_sweepPoint_le_abs axisX axisY inner_axisX_axisY s t

/-- **Angles.** Every point of the circle has an angle in each window
`(t - π, t + π]`. -/
theorem exists_circlePoint_eq (y : SSphere 2) (t : ℝ) :
    ∃ s ∈ Ioc (t - π) (t + π), circlePoint s = y := by
  set z : ℂ := ⟨(y : EucSpace 2) 0, (y : EucSpace 2) 1⟩
  have hy : (y : EucSpace 2) 0 ^ 2 + (y : EucSpace 2) 1 ^ 2 = 1 := by
    have h := mem_sphere_zero_iff_norm.mp y.2
    rw [EuclideanSpace.norm_eq, Fin.sum_univ_two, Real.norm_eq_abs, Real.norm_eq_abs, sq_abs,
      sq_abs, Real.sqrt_eq_one] at h
    exact h
  have hz : ‖z‖ = 1 := by
    rw [Complex.norm_eq_sqrt_sq_add_sq]
    simp [z, hy]
  have hz0 : z ≠ 0 := fun h => by simp [h] at hz
  have hcos : Real.cos (Complex.arg z) = (y : EucSpace 2) 0 := by
    rw [Complex.cos_arg hz0, hz, div_one]
  have hsin : Real.sin (Complex.arg z) = (y : EucSpace 2) 1 := by
    rw [Complex.sin_arg, hz, div_one]
  have hs := toIocMod_mem_Ioc two_pi_pos (t - π) (Complex.arg z)
  have hs' : toIocMod two_pi_pos (t - π) (Complex.arg z) = Complex.arg z -
      (toIocDiv two_pi_pos (t - π) (Complex.arg z) : ℝ) * (2 * π) := by
    rw [eq_sub_iff_add_eq, ← zsmul_eq_mul]
    exact toIocMod_add_toIocDiv_zsmul _ _ _
  refine ⟨_, ⟨hs.1, by linarith [hs.2]⟩, Subtype.ext ?_⟩
  ext k
  fin_cases k
  · simp only [Fin.zero_eta, circlePoint_apply_zero, hs', Real.cos_sub_int_mul_two_pi, hcos]
  · simp only [Fin.mk_one, circlePoint_apply_one, hs', Real.sin_sub_int_mul_two_pi, hsin]

/-- Angles closer than `2π` with the same point are equal. -/
theorem eq_of_circlePoint_eq {s t : ℝ} (h : circlePoint s = circlePoint t)
    (hst : |s - t| < 2 * π) : s = t := by
  have hd := dist_circlePoint_sq s t
  rw [h, dist_self] at hd
  have hc : Real.cos (s - t) = 1 := by nlinarith
  have := (Real.cos_eq_one_iff_of_lt_of_lt (by linarith [neg_abs_le (s - t)])
    (by linarith [le_abs_self (s - t)])).mp hc
  linarith

/-- The angle chart is injective on every window `(a, b]` of length at
most `2π`. -/
theorem injOn_circlePoint {a b : ℝ} (hab : b - a ≤ 2 * π) : InjOn circlePoint (Ioc a b) :=
  fun s hs t ht h => eq_of_circlePoint_eq h (abs_lt.mpr ⟨by linarith [hs.1, ht.2],
    by linarith [hs.2, ht.1]⟩)

/-- Disjoint parts of a window of length at most `2π` have disjoint
arcs. -/
theorem disjoint_image_circlePoint {a b : ℝ} (hab : b - a ≤ 2 * π) {I J : Set ℝ}
    (hI : I ⊆ Ioc a b) (hJ : J ⊆ Ioc a b) (hIJ : Disjoint I J) :
    Disjoint (circlePoint '' I) (circlePoint '' J) := by
  rw [Set.disjoint_iff_inter_eq_empty, ← (injOn_circlePoint hab).image_inter hI hJ,
    Set.disjoint_iff_inter_eq_empty.mp hIJ, image_empty]

/-- The chord of the angle `h`, the radius of the ball whose arc has
angular radius `h`. -/
noncomputable def chord (h : ℝ) : ℝ := √(2 - 2 * Real.cos h)

theorem continuous_chord : Continuous chord :=
  (continuous_const.sub (continuous_const.mul Real.continuous_cos)).sqrt

theorem chord_zero : chord 0 = 0 := by simp [chord]

theorem chord_le_two (h : ℝ) : chord h ≤ 2 := by
  rw [chord, Real.sqrt_le_left]
  · nlinarith [Real.neg_one_le_cos h]
  · norm_num

theorem chord_pos {h : ℝ} (h0 : 0 < h) (hπ : h ≤ π) : 0 < chord h :=
  Real.sqrt_pos.mpr (by
    linarith [Real.cos_lt_cos_of_nonneg_of_le_pi le_rfl hπ h0, Real.cos_zero])

/-- **Arcs are balls.** For `0 ≤ h ≤ π`, the ball around the point at
angle `t` of radius the chord of `h` is the arc of the angles in
`(t - h, t + h)`. -/
theorem ball_circlePoint (t h : ℝ) (h0 : 0 ≤ h) (hπ : h ≤ π) :
    Metric.ball (circlePoint t) (chord h) = circlePoint '' Ioo (t - h) (t + h) := by
  have key (s : ℝ) (hs : |s - t| ≤ π) :
      circlePoint s ∈ Metric.ball (circlePoint t) (chord h) ↔ |s - t| < h := by
    rw [Metric.mem_ball, chord, Real.lt_sqrt dist_nonneg, dist_circlePoint_sq, ← Real.cos_abs]
    constructor
    · intro hlt
      by_contra hnot
      linarith [Real.cos_le_cos_of_nonneg_of_le_pi h0 hs (not_lt.mp hnot)]
    · intro hlt
      linarith [Real.cos_lt_cos_of_nonneg_of_le_pi (abs_nonneg (s - t)) hπ hlt]
  ext y
  constructor
  · intro hy
    obtain ⟨s, hs, rfl⟩ := exists_circlePoint_eq y t
    have hlt := (key s (abs_le.mpr ⟨by linarith [hs.1], by linarith [hs.2]⟩)).mp hy
    exact ⟨s, ⟨by linarith [(abs_lt.mp hlt).1], by linarith [(abs_lt.mp hlt).2]⟩, rfl⟩
  · rintro ⟨s, hs, rfl⟩
    have hlt : |s - t| < h := abs_lt.mpr ⟨by linarith [hs.1], by linarith [hs.2]⟩
    exact (key s (by linarith)).mpr hlt

/-- A metric sphere of the circle has at most two points. -/
theorem sphere_circlePoint_subset (t r : ℝ) :
    Metric.sphere (circlePoint t) r ⊆
      {circlePoint (t - arccos (1 - r ^ 2 / 2)), circlePoint (t + arccos (1 - r ^ 2 / 2))} := by
  intro y hy
  obtain ⟨s, hs, rfl⟩ := exists_circlePoint_eq y t
  have hd := dist_circlePoint_sq s t
  rw [Metric.mem_sphere.mp hy] at hd
  have hc : Real.cos |s - t| = 1 - r ^ 2 / 2 := by
    rw [Real.cos_abs]
    linarith
  have habs : |s - t| = arccos (1 - r ^ 2 / 2) := by
    rw [← hc, Real.arccos_cos (abs_nonneg _)
      (abs_le.mpr ⟨by linarith [hs.1], by linarith [hs.2]⟩)]
  rcases (abs_eq (Real.arccos_nonneg _)).mp habs with h | h
  · right
    rw [mem_singleton_iff, show s = t + arccos (1 - r ^ 2 / 2) by linarith]
  · left
    rw [show s = t - arccos (1 - r ^ 2 / 2) by linarith]

/-- **Null spheres.** A law on the circle without atoms gives no mass to
any metric sphere. -/
theorem measure_sphere_circle_eq_zero (μ : Measure (SSphere 2)) (hatom : ∀ y, μ {y} = 0)
    (z : SSphere 2) (r : ℝ) : μ (Metric.sphere z r) = 0 := by
  obtain ⟨t, -, rfl⟩ := exists_circlePoint_eq z 0
  refine measure_mono_null (sphere_circlePoint_subset t r) ?_
  rw [Set.insert_eq]
  exact measure_union_null (hatom _) (hatom _)

/-- The hypotheses of the chart lemmas are satisfiable: the angle `0`
twice, the window `(0, 2π]` split into `(0, π]` and `(π, 2π]`, the
angle `π`, and the zero measure. -/
example : circlePoint 0 = circlePoint 0 ∧ |(0 : ℝ) - 0| < 2 * π ∧ 2 * π - 0 ≤ 2 * π ∧
    Ioc 0 π ⊆ Ioc 0 (2 * π) ∧ Ioc π (2 * π) ⊆ Ioc 0 (2 * π) ∧
    Disjoint (Ioc 0 π) (Ioc π (2 * π)) ∧ (0 < π ∧ π ≤ π) ∧
    ∀ y : SSphere 2, (0 : Measure (SSphere 2)) {y} = 0 :=
  ⟨rfl, by simp [Real.pi_pos], by linarith, Ioc_subset_Ioc_right (by linarith [Real.pi_pos]),
    Ioc_subset_Ioc_left Real.pi_pos.le, Ioc_disjoint_Ioc_of_le le_rfl, ⟨Real.pi_pos, le_rfl⟩,
    fun _ => rfl⟩

end Transformer.Interpolation
