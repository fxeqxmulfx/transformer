/-
# A continuous time change for Kolmogorov's increment criterion

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
The auxiliary clock sqrt(1+t)-1 turns the drift's fourth-power
increment bound into a global quadratic bound. Its inverse restores
the original physical time after constructing a continuous modification.
-/

import Transformer.BatchSize.Section4_EulerLimitIncrements

open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

/-- A slow continuous clock used only to construct the Section 4.3
diffusion's continuous modification; its value is sqrt(1+t)-1. -/
def eulerTimeCompress (t : ℝ≥0) : ℝ≥0 :=
  ⟨Real.sqrt (1 + (t : ℝ)) - 1, by
    have h := Real.sqrt_le_sqrt (show (1 : ℝ) ≤ 1 + t by linarith [t.coe_nonneg])
    rw [Real.sqrt_one] at h
    linarith⟩

/-- The inverse auxiliary clock, t^2+2t, Section 4.3 (2)--(3). -/
def eulerTimeExpand (t : ℝ≥0) : ℝ≥0 := t ^ 2 + 2 * t

/-- The slow clock is continuous, Section 4.3 (2)--(3). -/
theorem eulerTimeCompress_continuous : Continuous eulerTimeCompress := by
  apply Continuous.subtype_mk
  exact ((continuous_const.add NNReal.continuous_coe).sqrt.sub continuous_const)

/-- The inverse clock is continuous, Section 4.3 (2)--(3). -/
theorem eulerTimeExpand_continuous : Continuous eulerTimeExpand :=
  (continuous_id.pow 2).add (continuous_const.mul continuous_id)

/-- Returning from the auxiliary clock recovers exactly the physical
time, Section 4.3 (2)--(3). -/
theorem eulerTimeCompress_expand (t : ℝ≥0) : eulerTimeCompress (eulerTimeExpand t) = t := by
  apply NNReal.coe_injective
  change Real.sqrt (1 + (eulerTimeExpand t : ℝ)) - 1 = (t : ℝ)
  have heq : 1 + (eulerTimeExpand t : ℝ) = (1 + (t : ℝ)) ^ 2 := by
    simp only [eulerTimeExpand, NNReal.coe_add, NNReal.coe_pow, NNReal.coe_mul, NNReal.coe_ofNat]
    ring
  rw [heq, Real.sqrt_sq (by positivity)]
  ring

/-- The slow clock preserves time order, Section 4.3 (2)--(3). -/
theorem eulerTimeCompress_monotone : Monotone eulerTimeCompress := by
  intro s v hsv
  apply NNReal.coe_le_coe.mpr
  change Real.sqrt (1 + (s : ℝ)) - 1 ≤ Real.sqrt (1 + (v : ℝ)) - 1
  exact sub_le_sub_right (Real.sqrt_le_sqrt (by linarith [NNReal.coe_le_coe.mpr hsv])) 1

/-- Both second and fourth powers of the slow-clock increment are
bounded by the squared original-time increment, Section 4.3, Theorem 1.
This yields a single global Kolmogorov constant on all nonnegative times. -/
theorem eulerTimeCompress_increment_bounds (s v : ℝ≥0) (hsv : s ≤ v) :
    ((eulerTimeCompress v : ℝ) - eulerTimeCompress s) ^ 2 ≤ ((v : ℝ) - s) ^ 2 ∧
    ((eulerTimeCompress v : ℝ) - eulerTimeCompress s) ^ 4 ≤ ((v : ℝ) - s) ^ 2 := by
  have hxsq := Real.sq_sqrt (show 0 ≤ 1 + (s : ℝ) by positivity)
  have hysq := Real.sq_sqrt (show 0 ≤ 1 + (v : ℝ) by positivity)
  have hx : 1 ≤ Real.sqrt (1 + (s : ℝ)) := by
    simpa only [Real.sqrt_one] using Real.sqrt_le_sqrt (show (1 : ℝ) ≤ 1 + s by linarith [s.coe_nonneg])
  have hxy : Real.sqrt (1 + (s : ℝ)) ≤ Real.sqrt (1 + (v : ℝ)) :=
    Real.sqrt_le_sqrt (by linarith [NNReal.coe_le_coe.mpr hsv])
  let u : ℝ := Real.sqrt (1 + (v : ℝ)) - Real.sqrt (1 + (s : ℝ))
  have hu : 0 ≤ u := sub_nonneg.mpr hxy
  have husq : u ^ 2 ≤ (v : ℝ) - s := by
    dsimp [u]
    nlinarith [mul_nonneg (Real.sqrt_nonneg (1 + (s : ℝ))) (sub_nonneg.mpr hxy)]
  have hule : u ≤ (v : ℝ) - s := by
    dsimp [u]
    nlinarith [mul_nonneg (sub_nonneg.mpr hx) (sub_nonneg.mpr hxy)]
  have h2 := pow_le_pow_left₀ hu hule 2
  have h4 := pow_le_pow_left₀ (sq_nonneg u) husq 2
  have heq : (eulerTimeCompress v : ℝ) - eulerTimeCompress s = u := by
    change (Real.sqrt (1 + (v : ℝ)) - 1) - (Real.sqrt (1 + (s : ℝ)) - 1) = u
    dsimp [u]
    ring
  rw [heq]
  exact ⟨h2, by convert h4 using 1; ring⟩

/-- Nonvacuity of the time-change increment hypothesis,
Section 4.3: distinct nonnegative times. -/
example : (0 : ℝ≥0) ≤ 1 := by norm_num

end Transformer.BatchSize
