/-
# Distances and parking points for the actual spherical sweep

ArXiv:2411.04551v3, §5, `cl: balls`. The inner product of two
actual centers is the cosine of their angular separation. This gives
both the small-displacement estimates and an exact membership test for
the actual ambient balls in terms of their geodesic radii.
The geodesic radius is `arccos(1-r²/2)`, with the positivity and
short-arc conditions derived from the actual ambient radius.

For two successive, strictly ordered centers, moving a little less than
one radius forward from the earlier ball's center gives a point inside
that ball and outside every later one. The construction proves the
source's parking-point clause for the whole remaining segment, rather
than assuming it follows merely from equality of the radii.
-/

import Transformer.Interpolation.Section5_BallGeometry
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

open Real Metric Set

namespace Transformer.Interpolation

variable {d : ℕ} (p v : SSphere d)
  (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0)

/-- The actual center inner product is the cosine of its angular
separation. Source: arXiv:2411.04551v3, §5, the geodesic in `cl: balls`. -/
theorem inner_sweepPoint (s t : ℝ) : inner (𝕜 := ℝ)
    (sweepPoint p v hperp s : EucSpace d) (sweepPoint p v hperp t : EucSpace d) =
      Real.cos (s - t) := by
  have hpp : inner (𝕜 := ℝ) (p : EucSpace d) (p : EucSpace d) = 1 := by simp
  have hvv : inner (𝕜 := ℝ) (v : EucSpace d) (v : EucSpace d) = 1 := by simp
  have hvp : inner (𝕜 := ℝ) (v : EucSpace d) (p : EucSpace d) = 0 := by
    rw [real_inner_comm, hperp]
  change inner (𝕜 := ℝ) (Perceptron.greatCircle _ _ s) (Perceptron.greatCircle _ _ t) = _
  simp only [Perceptron.greatCircle, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right, hpp, hvv, hperp, hvp,
    mul_zero, mul_one, zero_add, add_zero]
  rw [Real.cos_sub]
  ring

/-- The actual squared chord distance of the centers.
Source: arXiv:2411.04551v3, §5, `cl: balls`, ambient-radius convention. -/
theorem dist_sweepPoint_sq (s t : ℝ) :
    dist (sweepPoint p v hperp s) (sweepPoint p v hperp t) ^ 2 = 2 - 2 * Real.cos (s - t) := by
  change ‖(sweepPoint p v hperp s : EucSpace d) - (sweepPoint p v hperp t : EucSpace d)‖ ^ 2 = _
  rw [norm_sub_sq_real, norm_eq_of_mem_sphere, norm_eq_of_mem_sphere, inner_sweepPoint]
  ring

/-- Chord displacement is at most the actual angular displacement.
Source: arXiv:2411.04551v3, §5, `cl: balls`, controlling all centers. -/
theorem dist_sweepPoint_le_abs (s t : ℝ) :
    dist (sweepPoint p v hperp s) (sweepPoint p v hperp t) ≤ |s - t| := by
  have hs := Real.sin_sq_add_cos_sq ((s - t) / 2)
  have hc := Real.cos_two_mul ((s - t) / 2)
  rw [show 2 * ((s - t) / 2) = s - t by ring] at hc
  have hb := Real.abs_sin_le_abs (x := (s - t) / 2)
  have hb2 := sq_le_sq.mpr hb
  have hd := dist_sweepPoint_sq p v hperp s t
  nlinarith [dist_nonneg (x := sweepPoint p v hperp s) (y := sweepPoint p v hperp t),
    abs_nonneg (s - t), sq_abs (s - t)]

/-- On a short positive segment the endpoint displacement is at
least half its angular length. Source: arXiv:2411.04551v3, §5,
choosing the first center away from the small carrying cap. -/
theorem half_le_dist_sweepPoint_zero {a : ℝ} (ha : 0 ≤ a) (ha1 : a ≤ 1) :
    a / 2 ≤ dist (sweepPoint p v hperp a) p := by
  have hsin := Real.mul_le_sin ha (by linarith [Real.pi_gt_three])
  have hcoef : (1 / 2 : ℝ) ≤ 2 / Real.pi := by
    apply (le_div_iff₀ Real.pi_pos).mpr
    linarith [Real.pi_lt_four]
  have hinner : inner (𝕜 := ℝ) ((sweepPoint p v hperp a : EucSpace d) - (p : EucSpace d))
      (v : EucSpace d) = Real.sin a := by
    change inner (𝕜 := ℝ) (Perceptron.greatCircle _ _ a - (p : EucSpace d)) (v : EucSpace d) = _
    simp [Perceptron.greatCircle, inner_sub_left, inner_add_left, real_inner_smul_left, hperp]
  have h := real_inner_le_norm ((sweepPoint p v hperp a : EucSpace d) - (p : EucSpace d)) (v : EucSpace d)
  rw [hinner, norm_eq_of_mem_sphere, mul_one] at h
  change a / 2 ≤ ‖(sweepPoint p v hperp a : EucSpace d) - (p : EucSpace d)‖
  nlinarith [mul_nonneg ha (sub_nonneg.mpr hcoef)]

/-- The actual ambient-radius membership test agrees with the
actual angular radius on a segment of length at most `π`.
Source: arXiv:2411.04551v3, §5, `cl: balls`, equal-radius geometry. -/
theorem mem_sweep_ball_iff_abs {r s t : ℝ} (hr : 0 < r) (hr1 : r ≤ 1)
    (hst : |s - t| ≤ Real.pi) :
    sweepPoint p v hperp s ∈ ball (sweepPoint p v hperp t) r ↔
      |s - t| < Real.arccos (1 - r ^ 2 / 2) := by
  have hc0 : 0 ≤ 1 - r ^ 2 / 2 := by nlinarith
  have hc1 : 1 - r ^ 2 / 2 ≤ 1 := by nlinarith
  have hcos := Real.cos_arccos (by linarith : -1 ≤ 1 - r ^ 2 / 2) hc1
  have hρ := Real.arccos_nonneg (1 - r ^ 2 / 2)
  have hρpi := (Real.arccos_le_pi_div_two.mpr hc0).trans (by linarith [Real.pi_pos] : Real.pi / 2 ≤ Real.pi)
  rw [mem_sphere_ball_iff_inner _ _ hr, inner_sweepPoint, ← Real.cos_abs]
  conv_lhs => rw [← hcos]
  constructor
  · intro h
    by_contra hnot
    exact (not_lt_of_ge (Real.cos_le_cos_of_nonneg_of_le_pi hρ hst (le_of_not_gt hnot))) h
  · exact Real.cos_lt_cos_of_nonneg_of_le_pi (abs_nonneg _) hρpi

/-- A genuine point inside the current ball lies outside every ball
whose center is on the remaining segment. Source: arXiv:2411.04551v3,
§5, property 2 of `cl: balls`. Strictly separated successive centers,
a positive radius, and the short-segment bounds are all explicit. -/
theorem exists_sweep_parking {r a s t : ℝ} (hr : 0 < r) (hr1 : r ≤ 1)
    (hs : 0 ≤ s) (hst : s < t) (hta : t ≤ a) (ha : a ≤ Real.pi / 4) :
    ∃ z ∈ ball (sweepPoint p v hperp t) r, ∀ u : ℝ, 0 ≤ u → u ≤ s →
      z ∉ ball (sweepPoint p v hperp u) r := by
  let ρ := Real.arccos (1 - r ^ 2 / 2)
  have hρ : 0 < ρ := Real.arccos_pos.mpr (by nlinarith)
  have hρpi : ρ ≤ Real.pi / 2 := Real.arccos_le_pi_div_two.mpr (by nlinarith)
  let e := min (ρ / 2) ((t - s) / 2)
  have he : 0 < e := lt_min (by positivity) (by linarith)
  have heρ : e ≤ ρ / 2 := min_le_left _ _
  have het : e ≤ (t - s) / 2 := min_le_right _ _
  let w := t + ρ - e
  have hwt : 0 ≤ w - t := by dsimp [w]; linarith
  have hw : w ≤ Real.pi := by dsimp [w]; linarith [Real.pi_pos]
  refine ⟨sweepPoint p v hperp w, ?_, fun u hu hus => ?_⟩
  · apply (mem_sweep_ball_iff_abs p v hperp hr hr1 ?_).mpr
    · rw [abs_of_nonneg hwt]
      dsimp [w]
      change _ < ρ
      linarith
    · rw [abs_of_nonneg hwt]
      linarith
  · have hwu : 0 ≤ w - u := by linarith
    rw [mem_sweep_ball_iff_abs p v hperp hr hr1 (by rw [abs_of_nonneg hwu]; linarith),
      abs_of_nonneg hwu]
    change ¬ w - u < ρ
    dsimp [w]
    linarith

/-- Actual orthogonal axes and a short nonconstant segment jointly
realize the distance, radius, frame and parking hypotheses. -/
example : ∃ (p v : SSphere 2)
    (hperp : inner (𝕜 := ℝ) (p : EucSpace 2) (v : EucSpace 2) = 0),
    (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
      (0 : ℝ) < 1 / 8 ∧ (1 / 8 : ℝ) ≤ 1 / 4 ∧ (1 / 4 : ℝ) ≤ Real.pi / 4 ∧
      ∃ z ∈ ball (sweepPoint p v hperp (1 / 8)) (1 / 4),
        z ∉ ball (sweepPoint p v hperp 0) (1 / 4) := by
  obtain ⟨v, hv⟩ := exists_unit_tangent le_rfl (basePoint 1)
  refine ⟨basePoint 1, v, hv, by norm_num, by norm_num, le_rfl,
    by norm_num, by norm_num, by linarith [Real.pi_gt_three], ?_⟩
  obtain ⟨z, hz, hrest⟩ := exists_sweep_parking (basePoint 1) v hv (a := 1 / 4)
    (s := 0) (t := 1 / 8) (r := 1 / 4) (by norm_num) (by norm_num) le_rfl
    (by norm_num) (by norm_num) (by linarith [Real.pi_gt_three])
  exact ⟨z, hz, hrest 0 le_rfl le_rfl⟩

end Transformer.Interpolation
