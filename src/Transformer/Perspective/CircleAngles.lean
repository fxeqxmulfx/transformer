/-
# Angles and the circle

On the circle `𝕊¹ ⊂ ℝ²` a particle is an angle: `x = (cos θ, sin θ)`
(`circlePoint`), and a configuration of `n` particles is a tuple of angles
(`anglesToSphere`).  The inner product of two points is the cosine of the
difference of their angles (`inner_circlePoint`), and the tangent
projection of `(cos b, sin b)` at `(cos a, sin a)` is `-sin (a - b)` times
the unit tangent `(-sin a, cos a)` (`proj_circlePoint`).  So the right-hand
side of `USA` at `(cos θ_i, sin θ_i)` is the right-hand side of the angular
equation `eq:onangles` times that tangent, which is the velocity of
`(cos θ_i, sin θ_i)` (`hasDerivAt_circlePoint`): solutions of `eq:onangles`
give solutions of `USA` on `(𝕊¹)ⁿ` (`usa_anglesToSphere`).

The circle is the image of `θ ↦ e^{iθ}` under the isometry `ℂ ≃ ℝ²`,
`z ↦ (re z, im z)` (`complexEuc`), so that these computations are complex
arithmetic.  The angle map is `2π`-periodic, and two angles give one point only
if they differ by a multiple of `2π` (`exists_int_of_circlePoint_eq`); the
argument is a section of it (`anglesToSphere_sphereAngles`).  Every linear
isometry of the plane acts on angles by a translation or a reflection
(`exists_sphereMap_circlePoint`).

Source: arXiv:2312.10794v5, §7.1, which derives `eq:onangles` from `USA` by
writing `x_i(t) = (cos θ_i(t), sin θ_i(t))`; here the computation runs the
other way.
-/

import Transformer.Perspective.Section6_Circle
import Transformer.Perspective.Section2_GradientFlow
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.Complex.Isometry
import Mathlib.Analysis.SpecialFunctions.Complex.Log

open Complex
open scoped Real

namespace Transformer
namespace Perspective

/-- The isometry `ℂ ≃ ℝ²`, `z ↦ (re z, im z)`. -/
noncomputable abbrev complexEuc : ℂ ≃ₗᵢ[ℝ] EucSpace 2 :=
  Complex.orthonormalBasisOneI.repr

/-- The point `(cos θ, sin θ)` of the unit circle, the image of `e^{iθ}`. -/
noncomputable def circlePoint (θ : ℝ) : SSphere 2 :=
  ⟨complexEuc (exp (θ * I)), by
    rw [mem_sphere_zero_iff_norm, LinearIsometryEquiv.norm_map, norm_exp_ofReal_mul_I]⟩

theorem coe_circlePoint (θ : ℝ) :
    (circlePoint θ : EucSpace 2) = complexEuc (exp (θ * I)) :=
  rfl

/-- `circlePoint θ` has the coordinates `(cos θ, sin θ)`. -/
theorem circlePoint_apply (θ : ℝ) :
    (circlePoint θ : EucSpace 2) 0 = Real.cos θ ∧ (circlePoint θ : EucSpace 2) 1 = Real.sin θ := by
  simp [coe_circlePoint]

/-- The configuration of the points `(cos θ_i, sin θ_i)` of the circle. -/
noncomputable def anglesToSphere {n : ℕ} (θ : Angles n) : SphereTuple 2 n :=
  fun i => circlePoint (θ i)

/-- The unit tangent `(-sin θ, cos θ)` to the circle at `circlePoint θ`, the
image of `i e^{iθ}`. -/
noncomputable def circleTangent (θ : ℝ) : EucSpace 2 :=
  complexEuc (I * exp (θ * I))

/-- `⟨(cos a, sin a), (cos b, sin b)⟩ = cos (a - b)`. -/
theorem inner_circlePoint (a b : ℝ) :
    inner ℝ (circlePoint a : EucSpace 2) (circlePoint b : EucSpace 2) = Real.cos (a - b) := by
  rw [coe_circlePoint, coe_circlePoint, LinearIsometryEquiv.inner_map_map, Complex.inner]
  simp [Complex.mul_re, Real.cos_sub]
  ring

/-- **The tangent projection on the circle**: the projection of
`(cos b, sin b)` onto the tangent line at `(cos a, sin a)` is `-sin (a - b)`
times the unit tangent `(-sin a, cos a)`. -/
theorem proj_circlePoint (a b : ℝ) :
    proj 2 (circlePoint a) (circlePoint b) = (-Real.sin (a - b)) • circleTangent a := by
  have h : exp (b * I) - Real.cos (a - b) • exp (a * I) =
      (-Real.sin (a - b)) • (I * exp (a * I)) := by
    apply Complex.ext <;> simp only [Complex.sub_re, Complex.sub_im, Complex.smul_re,
      Complex.smul_im, Complex.mul_re, Complex.mul_im, Complex.I_re, Complex.I_im,
      Complex.exp_ofReal_mul_I_re, Complex.exp_ofReal_mul_I_im, Real.cos_sub, Real.sin_sub,
      smul_eq_mul]
    · linear_combination (-Real.cos b) * Real.sin_sq_add_cos_sq a
    · linear_combination (-Real.sin b) * Real.sin_sq_add_cos_sq a
  have h' := congrArg complexEuc h
  rw [map_sub, map_smul, map_smul] at h'
  rw [proj, inner_circlePoint, coe_circlePoint, coe_circlePoint, circleTangent]
  exact h'

/-- **The velocity on the circle**: if `θ' = θ̇(t)`, the point
`(cos θ, sin θ)` moves with velocity `θ'` times the unit tangent. -/
theorem hasDerivAt_circlePoint {θ : ℝ → ℝ} {θ' t : ℝ} (h : HasDerivAt θ θ' t) :
    HasDerivAt (fun s => (circlePoint (θ s) : EucSpace 2)) (θ' • circleTangent (θ t)) t := by
  have hc := (h.ofReal_comp.mul_const I).cexp
  have h2 := (complexEuc.toContinuousLinearEquiv.toContinuousLinearMap.hasFDerivAt
    (x := exp (θ t * I))).comp_hasDerivAt t hc
  refine h2.congr_deriv ?_
  rw [circleTangent, ← map_smul]
  show complexEuc _ = complexEuc _
  rw [Complex.real_smul]
  ring_nf

/-- `Proj_x` is linear: it commutes with linear combinations. -/
theorem proj_smul_sum {d n : ℕ} (x : EucSpace d) (c : ℝ) (a : Idx n → ℝ) (y : Idx n → EucSpace d) :
    proj d x (c • ∑ j, a j • y j) = c • ∑ j, a j • proj d x (y j) := by
  simp only [proj, inner_smul_right, inner_sum, mul_smul, Finset.sum_smul, Finset.sum_sub_distrib,
    smul_sub]

/-- **`eq:onangles` gives `USA` on the circle.**  If the angles `θ(t)` solve
the angular equation `eq:onangles` with parameter `β`, the points
`(cos θ_i(t), sin θ_i(t))` solve `USA` on `(𝕊¹)ⁿ` with the same `β`. -/
theorem usa_anglesToSphere {n : ℕ} {β : ℝ} {θ : ℝ → Angles n} (hθ : angularUSA n β θ) :
    USA 2 n β (fun t => anglesToSphere (θ t)) := by
  intro t i
  refine (hasDerivAt_circlePoint (hθ t i)).congr_deriv ?_
  simp only [anglesToSphere, inner_circlePoint, proj_smul_sum, proj_circlePoint, smul_smul,
    ← Finset.sum_smul]
  congr 1
  simp only [mul_neg, Finset.sum_neg_distrib, one_div]
  ring

theorem continuous_circlePoint : Continuous circlePoint :=
  (complexEuc.continuous.comp (continuous_ofReal.mul continuous_const).cexp).subtype_mk _

/-- `θ ↦ (cos θ, sin θ)` has period `2π`. -/
theorem circlePoint_periodic : Function.Periodic circlePoint (2 * π) := fun θ => by
  apply Subtype.ext
  rw [coe_circlePoint, coe_circlePoint, ofReal_add, add_mul, Complex.exp_add, ofReal_mul,
    ofReal_ofNat, Complex.exp_two_pi_mul_I, mul_one]

/-- **The fibres of the angle map**: two angles give the same point of the
circle only if they differ by a multiple of `2π`. -/
theorem exists_int_of_circlePoint_eq {a b : ℝ} (h : circlePoint a = circlePoint b) :
    ∃ k : ℤ, a = b + 2 * π * k := by
  obtain ⟨k, hk⟩ := Complex.exp_eq_exp_iff_exists_int.1
    (complexEuc.injective (congrArg Subtype.val h))
  refine ⟨k, ?_⟩
  have him := congrArg Complex.im hk
  simp at him
  linear_combination him

/-- **Isometries act on angles by translations and reflections**: for a
linear isometry `U` of the plane there is an `α` with
`U (cos θ, sin θ) = (cos (α + θ), sin (α + θ))` for every `θ`, or with
`U (cos θ, sin θ) = (cos (α - θ), sin (α - θ))` for every `θ`. -/
theorem exists_sphereMap_circlePoint (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) :
    ∃ α : ℝ, (∀ θ, sphereMap 2 U (circlePoint θ) = circlePoint (α + θ)) ∨
      ∀ θ, sphereMap 2 U (circlePoint θ) = circlePoint (α - θ) := by
  obtain ⟨a, ha⟩ := linear_isometry_complex ((complexEuc.trans U).trans complexEuc.symm)
  have ha1 : exp (arg (a : ℂ) * I) = a := by
    simpa using Complex.norm_mul_exp_arg_mul_I (a : ℂ)
  have hU : ∀ θ : ℝ, (sphereMap 2 U (circlePoint θ) : EucSpace 2) =
      complexEuc ((complexEuc.trans U).trans complexEuc.symm (exp (θ * I))) := fun θ => by
    simp only [LinearIsometryEquiv.trans_apply, LinearIsometryEquiv.apply_symm_apply]
    rfl
  refine ⟨arg (a : ℂ), ha.imp (fun ha θ => Subtype.ext ?_) (fun ha θ => Subtype.ext ?_)⟩
  · rw [hU, ha, rotation_apply, coe_circlePoint, ofReal_add, add_mul, Complex.exp_add, ha1]
  · rw [hU, ha, LinearIsometryEquiv.trans_apply, Complex.conjLIE_apply, rotation_apply,
      coe_circlePoint, ← Complex.exp_conj, map_mul (starRingEnd ℂ), conj_ofReal, conj_I,
      ofReal_sub, sub_mul, Complex.exp_sub, ha1, mul_neg, Complex.exp_neg, div_eq_mul_inv]

/-- The angles of a configuration on the circle, each in `(-π, π]`: the
argument of `x_i` seen in `ℂ`. -/
noncomputable def sphereAngles {n : ℕ} (X : SphereTuple 2 n) : Angles n :=
  fun i => arg (complexEuc.symm (X i : EucSpace 2))

/-- The argument is a section of the angle map. -/
theorem anglesToSphere_sphereAngles {n : ℕ} (X : SphereTuple 2 n) :
    anglesToSphere (sphereAngles X) = X := by
  funext i
  apply Subtype.ext
  have hz : ‖complexEuc.symm (X i : EucSpace 2)‖ = 1 := by
    rw [LinearIsometryEquiv.norm_map]
    exact norm_eq_of_mem_sphere (X i)
  have h := Complex.norm_mul_exp_arg_mul_I (complexEuc.symm (X i : EucSpace 2))
  rw [hz, ofReal_one, one_mul] at h
  rw [anglesToSphere, coe_circlePoint, sphereAngles, h, LinearIsometryEquiv.apply_symm_apply]

/-- The hypotheses of `hasDerivAt_circlePoint`, `usa_anglesToSphere` and
`exists_int_of_circlePoint_eq` are satisfiable: the identity has derivative
`1`, a single token at angle `0` solves `eq:onangles` (its only interaction is
with itself, `sin 0 = 0`), and `0` and `2π` give the same point. -/
example : HasDerivAt (fun s : ℝ => s) 1 0 ∧ angularUSA 1 1 (fun _ _ => (0 : ℝ)) ∧
    circlePoint (0 + 2 * π) = circlePoint 0 :=
  ⟨hasDerivAt_id 0, fun t _ => by simpa using hasDerivAt_const t (0 : ℝ), circlePoint_periodic 0⟩

end Perspective
end Transformer
