/-
# A smooth ambient extension of the spherical projection flow

ArXiv:2411.04551v3, §3.1, `lem: first.quadrant`, and §1,
`eq: cauchy.pb`. Weak tests are globally smooth ambient functions.
The direct flow formula can have a zero denominator away from the sphere.
Replacing its reciprocal by `D / (D² + (‖x‖²-1)²)` removes those zeros
globally while retaining the exact original spherical map.
The extension transports the actual tangent field into itself, as follows
from the proved group law and differentiation of the original trajectories.
-/

import Transformer.Interpolation.Section3_ProjectionDynamics

open Real

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A globally positive denominator for the smooth ambient extension.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, applied to the
characteristic maps of §3.1, `lem: first.quadrant`. -/
noncomputable def projectionSafeDenom (w : EucSpace d) (t : ℝ) (x : EucSpace d) : ℝ :=
  projectionDenom w t x ^ 2 + (‖x‖ ^ 2 - 1) ^ 2

/-- The auxiliary denominator never vanishes for a unit axis, including
at points off the sphere. Source: arXiv:2411.04551v3, §3.1,
the characteristic construction in `lem: first.quadrant`. -/
theorem projectionSafeDenom_pos (w : EucSpace d) (hw : ‖w‖ = 1)
    (t : ℝ) (x : EucSpace d) : 0 < projectionSafeDenom w t x := by
  by_cases hx : ‖x‖ = 1
  · have hp := projectionDenom_pos w x hw hx t
    dsimp [projectionSafeDenom]
    nlinarith [sq_nonneg (‖x‖ ^ 2 - 1)]
  · have hne : ‖x‖ ^ 2 - 1 ≠ 0 := by
      intro heq
      apply hx
      nlinarith [norm_nonneg x]
    exact add_pos_of_nonneg_of_pos (sq_nonneg _) (sq_pos_of_ne_zero hne)

/-- A globally smooth extension agreeing with the original flow at every
unit point. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and
§3.1, `lem: first.quadrant`. Only the sphere restriction solves the ODE. -/
noncomputable def projectionExtension (w : EucSpace d) (t : ℝ) (x : EucSpace d) :
    EucSpace d :=
  (projectionDenom w t x / projectionSafeDenom w t x) • projectionNumerator w t x

/-- The extension retains the exact spherical formula.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionExtension_eq (w x : EucSpace d) (hw : ‖w‖ = 1) (hx : ‖x‖ = 1)
    (t : ℝ) : projectionExtension w t x = projectionFlow w t x := by
  have hD0 := (projectionDenom_pos w x hw hx t).ne'
  simp only [projectionExtension, projectionSafeDenom, hx, one_pow, sub_self,
    zero_pow (by decide : 2 ≠ 0), add_zero, projectionFlow]
  congr 1
  field_simp

/-- The ambient extension is jointly smooth at every time and point.
Source: arXiv:2411.04551v3, §1, the ambient test convention of
`eq: cauchy.pb`, and §3.1, `lem: first.quadrant`. -/
theorem contDiff_projectionExtension (w : EucSpace d) (hw : ‖w‖ = 1) :
    ContDiff ℝ (⊤ : ℕ∞) (fun q : ℝ × EucSpace d => projectionExtension w q.1 q.2) := by
  have hheight : ContDiff ℝ (⊤ : ℕ∞) (fun q : ℝ × EucSpace d =>
      inner (𝕜 := ℝ) w q.2) := contDiff_const.inner ℝ contDiff_snd
  have hD : ContDiff ℝ (⊤ : ℕ∞) (fun q : ℝ × EucSpace d =>
      projectionDenom w q.1 q.2) := by
    exact contDiff_fst.cosh.add (contDiff_fst.sinh.mul hheight)
  have hQ : ContDiff ℝ (⊤ : ℕ∞) (fun q : ℝ × EucSpace d =>
      projectionSafeDenom w q.1 q.2) :=
    (hD.pow 2).add (((contDiff_snd.norm_sq (𝕜 := ℝ)).sub contDiff_const).pow 2)
  have hN : ContDiff ℝ (⊤ : ℕ∞) (fun q : ℝ × EucSpace d =>
      projectionNumerator w q.1 q.2) :=
    contDiff_snd.add ((contDiff_fst.sinh.add
      ((contDiff_fst.cosh.sub contDiff_const).mul hheight)).smul contDiff_const)
  exact (hD.div hQ (fun q => (projectionSafeDenom_pos w hw q.1 q.2).ne')).smul hN

/-- Every fixed-time ambient extension is smooth on the entire ambient
space. Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem contDiff_projectionExtension_slice (w : EucSpace d) (hw : ‖w‖ = 1) (t : ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) (projectionExtension w t) :=
  ContDiff.comp (f := fun x : EucSpace d => (t, x))
    (g := fun q : ℝ × EucSpace d => projectionExtension w q.1 q.2)
    (contDiff_projectionExtension w hw) (contDiff_const.prodMk contDiff_id)

/-- The time derivative of the extension on the original sphere is the
original tangent velocity. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`. -/
theorem hasDerivAt_projectionExtension (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (t : ℝ) :
    HasDerivAt (fun s => projectionExtension w s x) (proj d (projectionFlow w t x) w) t := by
  simpa only [projectionExtension_eq w x hw hx] using hasDerivAt_projectionFlow w x hw hx t

/-- The extension transports the actual tangent direction into the same
field at the transported point. This differentiates the proved group law,
so no ambient equality off the sphere is assumed.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem fderiv_projectionExtension_field (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (t : ℝ) :
    fderiv ℝ (projectionExtension w t) x (proj d x w) = proj d (projectionFlow w t x) w := by
  have hc := contDiff_projectionExtension_slice w hw t
  have hd : HasFDerivAt (projectionExtension w t) (fderiv ℝ (projectionExtension w t) x)
      (projectionFlow w 0 x) := by
    simpa only [projectionFlow_zero] using (hc.differentiable (by simp) x).hasFDerivAt
  have hcomp := hd.comp_hasDerivAt 0 (hasDerivAt_projectionFlow w x hw hx 0)
  simp only [projectionFlow_zero] at hcomp
  have heq : (fun s => projectionExtension w t (projectionFlow w s x)) =
      (fun s => projectionFlow w (t + s) x) := by
    funext s
    rw [projectionExtension_eq w _ hw (norm_projectionFlow w x hw hx s) t,
      projectionFlow_add w x hw hx t s]
  change HasDerivAt (fun s => projectionExtension w t (projectionFlow w s x)) _ 0 at hcomp
  rw [heq] at hcomp
  have htime := (hasDerivAt_projectionFlow w x hw hx (t + 0)).scomp 0
    ((hasDerivAt_id 0).const_add t)
  simp only [Function.comp_def, add_zero, one_smul] at htime
  exact hcomp.unique htime

/-- The inverse characteristic has the negative transported velocity.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem hasDerivAt_projectionInverse (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (t : ℝ) :
    HasDerivAt (fun s => projectionExtension w (-s) x)
      (-(proj d (projectionFlow w (-t) x) w)) t := by
  simpa only [Function.comp_def, neg_one_smul] using
    (hasDerivAt_projectionExtension w x hw hx (-t)).scomp t (hasDerivAt_neg t)

/-- A smooth test pulled back by the inverse characteristic satisfies
the exact backward transport identity on the sphere.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
the characteristic construction of `lem: first.quadrant`. -/
theorem hasDerivAt_projection_pullback (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => φ (projectionExtension w (-s) x))
      (-(fderiv ℝ (fun y => φ (projectionExtension w (-t) y)) x (proj d x w))) t := by
  have hφd := (hφ.differentiable (by decide) (projectionExtension w (-t) x)).hasFDerivAt
  have hxd := ((contDiff_projectionExtension_slice w hw (-t)).differentiable
    (by simp) x).hasFDerivAt
  have hspace := (hφd.comp x hxd).fderiv
  change fderiv ℝ (fun y => φ (projectionExtension w (-t) y)) x = _ at hspace
  have ht := hφd.comp_hasDerivAt t (hasDerivAt_projectionInverse w x hw hx t)
  apply ht.congr_deriv
  rw [hspace, ContinuousLinearMap.comp_apply, map_neg,
    fderiv_projectionExtension_field w x hw hx (-t)]

/-- The regularity and tangent transport inputs hold for an actual pair
of orthogonal, nonzero unit vectors on the circle. -/
example :
    ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)‖ = 1 ∧
      ‖(EuclideanSpace.single (1 : Fin 2) (1 : ℝ) : EucSpace 2)‖ = 1 ∧
      inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)
        (EuclideanSpace.single (1 : Fin 2) (1 : ℝ) : EucSpace 2) = 0 := by
  simp [PiLp.norm_single, EuclideanSpace.inner_single_left]

end Transformer.Interpolation
