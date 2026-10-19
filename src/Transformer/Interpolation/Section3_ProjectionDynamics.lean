module

public import Transformer.Interpolation.Section3_ProjectionFlow
public import Transformer.Landscape.GlobalFlow.Existence
public import Mathlib.Analysis.Calculus.ContDiff.RCLike
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# The constant projection formula solves the original ODE

ArXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`.
The hyperbolic formula is differentiated at every real time. Uniqueness
of the polynomial ambient field then gives the group law and its continuous
inverse on the sphere. These are actual flow maps of `x' = Proj_x w`.
-/

@[expose] public section

open Real

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original tangent projection field is polynomial, hence locally
Lipschitz. Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem locallyLipschitz_projectionField (w : EucSpace d) :
    LocallyLipschitz (fun x : EucSpace d => proj d x w) := by
  have hc : ContDiff ℝ 1 (fun x : EucSpace d => proj d x w) := by
    unfold proj
    exact contDiff_const.sub ((contDiff_id.inner ℝ contDiff_const).smul contDiff_id)
  exact hc.locallyLipschitz

/-- The constructed formula solves `x' = Proj_x w` at every real time.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem hasDerivAt_projectionFlow (w x : EucSpace d) (hw : ‖w‖ = 1)
    (hx : ‖x‖ = 1) (t : ℝ) :
    HasDerivAt (fun s => projectionFlow w s x) (proj d (projectionFlow w t x) w) t := by
  have hD : HasDerivAt (fun s => projectionDenom w s x)
      (sinh t + cosh t * inner (𝕜 := ℝ) w x) t :=
    (hasDerivAt_cosh t).add ((hasDerivAt_sinh t).mul_const _)
  have hN : HasDerivAt (fun s => projectionNumerator w s x)
      (projectionDenom w t x • w) t := by
    convert (((hasDerivAt_sinh t).fun_add
      (((hasDerivAt_cosh t).sub_const 1).mul_const
        (inner (𝕜 := ℝ) w x))).smul_const w).const_add x using 1 <;>
      simp only [projectionNumerator, projectionDenom]
  have hD0 := (projectionDenom_pos w x hw hx t).ne'
  apply ((hD.inv hD0).fun_smul hN).congr_deriv
  change (projectionDenom w t x)⁻¹ • (projectionDenom w t x • w) +
    (-(sinh t + cosh t * inner (𝕜 := ℝ) w x) / projectionDenom w t x ^ 2) •
      projectionNumerator w t x = _
  rw [smul_smul, inv_mul_cancel₀ hD0, one_smul, proj,
    real_inner_comm w (projectionFlow w t x), inner_projectionFlow w x hw t]
  have hc : -(sinh t + cosh t * inner (𝕜 := ℝ) w x) / projectionDenom w t x ^ 2 =
      -((sinh t + cosh t * inner (𝕜 := ℝ) w x) / projectionDenom w t x) *
        (projectionDenom w t x)⁻¹ := by
    field_simp
  rw [hc, projectionFlow, mul_smul, neg_smul, sub_eq_add_neg]

/-- The constructed sphere-valued curve satisfies exactly the characteristic
equation used by the paper. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`. -/
theorem hasDerivAt_sphereProjectionFlow (w x : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => (sphereProjectionFlow w s x : EucSpace d))
      (proj d (sphereProjectionFlow w t x : EucSpace d) (w : EucSpace d)) t :=
  hasDerivAt_projectionFlow _ _ (mem_sphere_zero_iff_norm.mp w.2)
    (mem_sphere_zero_iff_norm.mp x.2) t

/-- The explicit trajectories have the group law of the original
autonomous equation. Source: arXiv:2411.04551v3, §3.1,
the characteristic flows in `lem: first.quadrant`. -/
theorem projectionFlow_add (w x : EucSpace d) (hw : ‖w‖ = 1) (hx : ‖x‖ = 1)
    (s t : ℝ) : projectionFlow w s (projectionFlow w t x) = projectionFlow w (s + t) x := by
  have heq : (fun r => projectionFlow w r (projectionFlow w t x)) =
      (fun r => projectionFlow w (r + t) x) := by
    apply Landscape.GlobalFlow.eq_of_hasDerivAt (locallyLipschitz_projectionField w)
    · intro r
      exact hasDerivAt_projectionFlow w _ hw (norm_projectionFlow w x hw hx t) r
    · intro r
      simpa only [Function.comp_def, one_smul, id_eq] using
        (hasDerivAt_projectionFlow w x hw hx (r + t)).scomp r
          ((hasDerivAt_id r).add_const t)
    · simp only [projectionFlow_zero, zero_add]
  exact congrFun heq s

/-- The same group law on the original sphere.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem sphereProjectionFlow_add (w x : SSphere d) (s t : ℝ) :
    sphereProjectionFlow w s (sphereProjectionFlow w t x) =
      sphereProjectionFlow w (s + t) x := by
  apply Subtype.ext
  exact projectionFlow_add _ _ (mem_sphere_zero_iff_norm.mp w.2)
    (mem_sphere_zero_iff_norm.mp x.2) s t

/-- Running the same field backward is the actual inverse map.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
@[simp] theorem sphereProjectionFlow_neg (w x : SSphere d) (t : ℝ) :
    sphereProjectionFlow w (-t) (sphereProjectionFlow w t x) = x := by
  rw [sphereProjectionFlow_add, neg_add_cancel, sphereProjectionFlow_zero]

/-- Running backward and then forward is the other inverse identity.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
@[simp] theorem sphereProjectionFlow_neg_right (w x : SSphere d) (t : ℝ) :
    sphereProjectionFlow w t (sphereProjectionFlow w (-t) x) = x := by
  rw [sphereProjectionFlow_add, add_neg_cancel, sphereProjectionFlow_zero]

/-- Each constant-field phase is a homeomorphism of the actual sphere.
Source: arXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`. -/
noncomputable def sphereProjectionHomeomorph (w : SSphere d) (t : ℝ) :
    SSphere d ≃ₜ SSphere d where
  toFun := sphereProjectionFlow w t
  invFun := sphereProjectionFlow w (-t)
  left_inv x := sphereProjectionFlow_neg w x t
  right_inv x := sphereProjectionFlow_neg_right w x t
  continuous_toFun := (continuous_sphereProjectionFlow w).comp
    (continuous_const.prodMk continuous_id)
  continuous_invFun := (continuous_sphereProjectionFlow w).comp
    (continuous_const.prodMk continuous_id)

/-- The attracting axis is fixed by the actual flow.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionFlow_axis (w : EucSpace d) (hw : ‖w‖ = 1) (t : ℝ) :
    projectionFlow w t w = w := by
  have hD0 := (projectionDenom_pos w w hw hw t).ne'
  have hN : projectionNumerator w t w = projectionDenom w t w • w := by
    simp only [projectionNumerator, projectionDenom, real_inner_self_eq_norm_sq, hw,
      one_pow, mul_one]
    module
  rw [projectionFlow, hN, smul_smul, inv_mul_cancel₀ hD0, one_smul]

/-- The attracting pole is fixed on the original sphere.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
@[simp] theorem sphereProjectionFlow_axis (w : SSphere d) (t : ℝ) :
    sphereProjectionFlow w t w = w :=
  Subtype.ext (projectionFlow_axis _ (mem_sphere_zero_iff_norm.mp w.2) t)

/-- The repelling pole is also fixed. Its exclusion in the contraction
statement is necessary. Source: arXiv:2411.04551v3, §3.1,
the two stationary points in the proof of `lem: first.quadrant`. -/
@[simp] theorem sphereProjectionFlow_antipode (w : SSphere d) (t : ℝ) :
    sphereProjectionFlow w t (antipode d w) = antipode d w := by
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hz : proj d (-(w : EucSpace d)) (w : EucSpace d) = 0 := by
    simp only [proj, inner_neg_left, real_inner_self_eq_norm_sq, hw, one_pow,
      neg_one_smul, neg_neg, sub_self]
  have heq := Landscape.GlobalFlow.eq_of_hasDerivAt (locallyLipschitz_projectionField (w : EucSpace d))
    (fun s => hasDerivAt_projectionFlow _ _ hw (mem_sphere_zero_iff_norm.mp
      (antipode d w).2) s)
    (g := fun _ => -(w : EucSpace d))
    (fun s => by rw [hz]; exact hasDerivAt_const s _) (by simp [antipode])
  exact Subtype.ext (congrFun heq t)

/-- An actual unit axis and initial point realize all norm hypotheses. -/
example : ‖(basePoint 1 : SSphere 2).val‖ = 1 ∧
    ‖(basePoint 1 : SSphere 2).val‖ = 1 := by
  constructor <;> exact mem_sphere_zero_iff_norm.mp (basePoint 1).2

end Transformer.Interpolation
