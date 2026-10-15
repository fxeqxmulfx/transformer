/-
# Explicit flow of a constant tangent projection

ArXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`.
The two phases use `x' = Proj_x w` with a fixed unit vector `w`.
The hyperbolic formula below constructs that flow on the original sphere.
Its denominator is positive at every real time, and the formula preserves
unit norm. No existence or transport assertion is part of the definition.
-/

import Transformer.Interpolation.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

open Real

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The denominator of the constant projection flow in the proof of
`lem: first.quadrant`, arXiv:2411.04551v3, §3.1. -/
noncomputable def projectionDenom (w : EucSpace d) (t : ℝ) (x : EucSpace d) : ℝ :=
  cosh t + sinh t * inner (𝕜 := ℝ) w x

/-- Every initial point has denominator one at time zero.
Source: arXiv:2411.04551v3, §3.1, the initial condition of
`lem: first.quadrant`. -/
@[simp] theorem projectionDenom_zero (w x : EucSpace d) : projectionDenom w 0 x = 1 := by
  simp [projectionDenom]

/-- The numerator of the same flow. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`, solving its constant tangent field explicitly. -/
noncomputable def projectionNumerator (w : EucSpace d) (t : ℝ)
    (x : EucSpace d) : EucSpace d :=
  x + (sinh t + (cosh t - 1) * inner (𝕜 := ℝ) w x) • w

/-- The ambient hyperbolic formula for the flow of `Proj_x w`.
Source: arXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`.
The sphere restriction, rather than arbitrary ambient points, solves the ODE. -/
noncomputable def projectionFlow (w : EucSpace d) (t : ℝ) (x : EucSpace d) : EucSpace d :=
  (projectionDenom w t x)⁻¹ • projectionNumerator w t x

/-- The hyperbolic denominator is positive whenever the height lies in
`[-1,1]`. Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem cosh_add_sinh_mul_pos (t h : ℝ) (hh : |h| ≤ 1) :
    0 < cosh t + sinh t * h := by
  have hc := cosh_pos t
  have hi := cosh_sq_sub_sinh_sq t
  have hab : |sinh t| < cosh t := by
    have hs : |sinh t| ^ 2 = sinh t ^ 2 := sq_abs _
    nlinarith [abs_nonneg (sinh t)]
  have hm : |sinh t * h| ≤ |sinh t| := by
    rw [abs_mul]
    exact mul_le_of_le_one_right (abs_nonneg _) hh
  linarith [neg_abs_le (sinh t * h)]

example : |(1 / 2 : ℝ)| ≤ 1 := by norm_num

/-- A unit axis and a unit point have a positive flow denominator at every
real time. Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionDenom_pos (w x : EucSpace d) (hw : ‖w‖ = 1) (hx : ‖x‖ = 1)
    (t : ℝ) : 0 < projectionDenom w t x := by
  apply cosh_add_sinh_mul_pos
  simpa only [hw, hx, mul_one] using abs_real_inner_le_norm w x

/-- The numerator's height is the scalar hyperbolic numerator.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem inner_projectionNumerator (w x : EucSpace d) (hw : ‖w‖ = 1) (t : ℝ) :
    inner (𝕜 := ℝ) w (projectionNumerator w t x) =
      sinh t + cosh t * inner (𝕜 := ℝ) w x := by
  rw [projectionNumerator, inner_add_right, real_inner_smul_right,
    real_inner_self_eq_norm_sq, hw]
  ring

/-- The numerator's squared norm is the denominator's square on the sphere.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem norm_projectionNumerator_sq (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (t : ℝ) :
    ‖projectionNumerator w t x‖ ^ 2 = projectionDenom w t x ^ 2 := by
  rw [projectionNumerator, norm_add_sq_real, real_inner_smul_right,
    norm_smul, mul_pow, Real.norm_eq_abs, sq_abs, hw, hx, one_pow, mul_one]
  simp only [real_inner_comm w x]
  have hi := cosh_sq_sub_sinh_sq t
  dsimp [projectionDenom]
  have hprod := congrArg (fun z : ℝ => z * (inner (𝕜 := ℝ) w x) ^ 2) hi
  nlinarith

/-- The explicitly constructed flow stays on the original sphere.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem norm_projectionFlow (w x : EucSpace d) (hw : ‖w‖ = 1) (hx : ‖x‖ = 1)
    (t : ℝ) : ‖projectionFlow w t x‖ = 1 := by
  have hD := projectionDenom_pos w x hw hx t
  have hN : ‖projectionNumerator w t x‖ = projectionDenom w t x := by
    nlinarith [norm_projectionNumerator_sq w x hw hx t,
      norm_nonneg (projectionNumerator w t x)]
  rw [projectionFlow, norm_smul, Real.norm_eq_abs, abs_inv, abs_of_pos hD, hN,
    inv_mul_cancel₀ hD.ne']

/-- The actual constant-projection flow on `𝕊^{d-1}`.
Source: arXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`. -/
noncomputable def sphereProjectionFlow (w : SSphere d) (t : ℝ) (x : SSphere d) :
    SSphere d :=
  ⟨projectionFlow (w : EucSpace d) t (x : EucSpace d), mem_sphere_zero_iff_norm.mpr
    (norm_projectionFlow (w : EucSpace d) (x : EucSpace d)
      (mem_sphere_zero_iff_norm.mp w.2) (mem_sphere_zero_iff_norm.mp x.2) t)⟩

/-- The explicit flow starts at the given point.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
@[simp] theorem projectionFlow_zero (w x : EucSpace d) : projectionFlow w 0 x = x := by
  simp [projectionFlow, projectionDenom, projectionNumerator]

@[simp] theorem sphereProjectionFlow_zero (w x : SSphere d) :
    sphereProjectionFlow w 0 x = x := by
  apply Subtype.ext
  exact projectionFlow_zero (w : EucSpace d) (x : EucSpace d)

/-- The height of the actual trajectory is the scalar Riccati flow.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem inner_projectionFlow (w x : EucSpace d) (hw : ‖w‖ = 1) (t : ℝ) :
    inner (𝕜 := ℝ) w (projectionFlow w t x) =
      (sinh t + cosh t * inner (𝕜 := ℝ) w x) / projectionDenom w t x := by
  rw [projectionFlow, real_inner_smul_right, inner_projectionNumerator w x hw t,
    div_eq_inv_mul]

/-- The actual spherical flow is jointly continuous in time and initial
point. Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem continuous_sphereProjectionFlow (w : SSphere d) :
    Continuous (fun q : ℝ × SSphere d => sphereProjectionFlow w q.1 q.2) := by
  have hD : Continuous (fun q : ℝ × SSphere d =>
      projectionDenom (w : EucSpace d) q.1 (q.2 : EucSpace d)) := by
    unfold projectionDenom
    exact (continuous_cosh.comp continuous_fst).add
      ((continuous_sinh.comp continuous_fst).mul
        (continuous_const.inner (continuous_subtype_val.comp continuous_snd)))
  have hN : Continuous (fun q : ℝ × SSphere d =>
      projectionNumerator (w : EucSpace d) q.1 (q.2 : EucSpace d)) := by
    unfold projectionNumerator
    exact (continuous_subtype_val.comp continuous_snd).add
      (((continuous_sinh.comp continuous_fst).add
        (((continuous_cosh.comp continuous_fst).sub continuous_const).mul
          (continuous_const.inner (continuous_subtype_val.comp continuous_snd)))).smul
            continuous_const)
  exact (Continuous.smul (hD.inv₀ (fun q =>
    (projectionDenom_pos _ _ (mem_sphere_zero_iff_norm.mp w.2)
      (mem_sphere_zero_iff_norm.mp q.2.2) q.1).ne')) hN).subtype_mk _

/-- The unit-axis hypotheses are realized by an actual nonzero unit vector. -/
example : ‖(basePoint 1 : SSphere 2).val‖ = 1 ∧
    ‖(basePoint 1 : SSphere 2).val‖ = 1 := by
  constructor <;> exact mem_sphere_zero_iff_norm.mp (basePoint 1).2

end Transformer.Interpolation
