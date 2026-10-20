module

public import Transformer.Causal.Section4_FrozenFollowerField
public import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-!
# Tangent eigenvectors from exact transverse component equations

ArXiv:2411.04990v2, §4, the critical-manifold issue preceding `thm2`.
Differentiating an actual scalar factorization at a zero transverse
component gives a left eigenvector of the full derivative. This keeps
all derivatives of the attention weights. A nonzero left eigenvector
forces a right eigenvector in finite dimension, without symmetry.

Projecting the actual derivative onto the actual tangent space retains
this eigenvalue. A positive eigenvalue supplies a nonzero tangent
eigenvector, rather than a spurious radial direction of an extension.
-/

@[expose] public section

open Real

namespace Transformer.Causal

variable {d : ℕ}

/-- The exact scalar factorization differentiates to a left eigenvector
when the transverse component vanishes. The scalar coefficient is
differentiated as well. Source: arXiv:2411.04990v2, §4, an auxiliary
linearization step for the critical configurations in `thm2`. -/
theorem inner_fderiv_of_transverse_factor (F : EucSpace d → EucSpace d)
    (h : EucSpace d → ℝ) (v x : EucSpace d)
    (hF : DifferentiableAt ℝ F x) (hh : DifferentiableAt ℝ h x)
    (hfactor : ∀ y, inner ℝ v (F y) = h y * inner ℝ v y)
    (hvx : inner ℝ v x = 0) (w : EucSpace d) :
    inner ℝ v (fderiv ℝ F x w) = h x * inner ℝ v w := by
  have hl := (innerSL ℝ v).hasFDerivAt.comp x hF.hasFDerivAt
  have hr := hh.hasFDerivAt.mul (innerSL ℝ v).hasFDerivAt
  simp only [innerSL_apply_apply, hvx, zero_smul, add_zero] at hr
  have hr' : HasFDerivAt (fun y => inner ℝ v (F y))
      (h x • innerSL ℝ v) x :=
    hr.congr_of_eventuallyEq (Filter.Eventually.of_forall hfactor)
  have he := congrArg (fun A : EucSpace d →L[ℝ] ℝ => A w) (hl.unique hr')
  simpa only [ContinuousLinearMap.comp_apply, smul_apply,
    innerSL_apply_apply, smul_eq_mul] using he

/-- A nonzero actual left eigenvector yields an actual right eigenvector
of the same real eigenvalue. No self-adjointness is assumed.
Source: arXiv:2411.04990v2, §4, the finite-dimensional linearization
needed for the critical-manifold instability argument before `thm2`. -/
theorem exists_eigenvector_of_left_inner (A : ParamMatrix d) (v : EucSpace d)
    (hv : v ≠ 0) (c : ℝ) (hleft : ∀ w, inner ℝ v (A w) = c * inner ℝ v w) :
    ∃ w : EucSpace d, w ≠ 0 ∧ A w = c • w := by
  let B : EucSpace d →ₗ[ℝ] EucSpace d := A.toLinearMap - c • LinearMap.id
  have hb (w : EucSpace d) : inner ℝ v (B w) = 0 := by
    simp only [B, LinearMap.sub_apply, LinearMap.smul_apply, LinearMap.id_apply,
      ContinuousLinearMap.coe_coe, inner_sub_right, real_inner_smul_right, hleft, sub_self]
  have hns : ¬Function.Surjective B := by
    intro hs
    obtain ⟨w, hw⟩ := hs v
    have he := hb w
    rw [hw] at he
    exact (real_inner_self_pos.mpr hv).ne' he
  have hk : LinearMap.ker B ≠ ⊥ := by
    intro hz
    exact hns (LinearMap.injective_iff_surjective.mp (LinearMap.ker_eq_bot.mp hz))
  obtain ⟨w, hw, hw0⟩ := (Submodule.ne_bot_iff (LinearMap.ker B)).mp hk
  refine ⟨w, hw0, ?_⟩
  change A w - c • w = 0 at hw
  exact sub_eq_zero.mp hw

/-- The continuous linear map of the original tangent projection.
Source: arXiv:2411.04990v2, §2, `eqn:csa`; its argument is the
base point of the same polynomial projection used in the original field. -/
noncomputable def csaTangentProjection (x : EucSpace d) : ParamMatrix d :=
  ContinuousLinearMap.id ℝ _ - (innerSL ℝ x).smulRight x

/-- The linear projection map evaluates to the original projection.
Source: arXiv:2411.04990v2, §2, `eqn:csa`. -/
theorem csaTangentProjection_apply (x w : EucSpace d) :
    csaTangentProjection x w = proj d x w := rfl

/-- A tangent covector is unchanged by the original tangent projection.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, `thm2`. -/
theorem inner_csaTangentProjection (x v : EucSpace d) (hv : inner ℝ v x = 0)
    (w : EucSpace d) : inner ℝ v (csaTangentProjection x w) = inner ℝ v w := by
  rw [csaTangentProjection_apply, proj, inner_sub_right, real_inner_smul_right, hv,
    mul_zero, sub_zero]

/-- A positive actual left eigenvalue survives in the actual tangent
linearization and has a nonzero tangent right eigenvector.
Source: arXiv:2411.04990v2, §4, the critical-manifold issue before `thm2`.
This is an algebraic instability statement, not a basin-nullity claim. -/
theorem exists_tangent_eigenvector_of_left_inner (A : ParamMatrix d) (x : SSphere d)
    (v : EucSpace d) (hv : v ≠ 0) (hvx : inner ℝ v x = 0)
    (c : ℝ) (hc : 0 < c) (hleft : ∀ w, inner ℝ v (A w) = c * inner ℝ v w) :
    ∃ w : EucSpace d, w ≠ 0 ∧ inner ℝ (x : EucSpace d) w = 0 ∧
      csaTangentProjection x (A w) = c • w := by
  let P := csaTangentProjection (x : EucSpace d)
  let J := P.comp (A.comp P)
  have hJ (w : EucSpace d) : inner ℝ v (J w) = c * inner ℝ v w := by
    change inner ℝ v (P (A (P w))) = _
    rw [inner_csaTangentProjection _ _ hvx, hleft, inner_csaTangentProjection _ _ hvx]
  obtain ⟨w, hw, he⟩ := exists_eigenvector_of_left_inner J v hv c hJ
  have ht : inner ℝ (x : EucSpace d) (J w) = 0 :=
    inner_proj_eq_zero (mem_sphere_zero_iff_norm.mp x.2) _
  rw [he, real_inner_smul_right] at ht
  have hxw : inner ℝ (x : EucSpace d) w = 0 := (mul_eq_zero.mp ht).resolve_left hc.ne'
  have hPw : P w = w := by
    rw [csaTangentProjection_apply, proj, hxw, zero_smul, sub_zero]
  refine ⟨w, hw, hxw, ?_⟩
  change P (A w) = c • w
  simpa only [J, ContinuousLinearMap.comp_apply, hPw] using he

/-- At an actual critical point, differentiating the original projected
field maps tangent directions into tangent directions. Thus projecting
the derivative does not add an artificial tangent eigenvalue.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, critical linearization. -/
theorem inner_fderiv_projectedField_of_critical (a : EucSpace d → EucSpace d)
    (x : SSphere d) (ha : DifferentiableAt ℝ a x)
    (hcrit : proj d (x : EucSpace d) (a x) = 0) (w : EucSpace d)
    (hxw : inner ℝ (x : EucSpace d) w = 0) :
    inner ℝ (x : EucSpace d) (fderiv ℝ (fun y => proj d y (a y)) x w) = 0 := by
  have hax : inner ℝ w (a x) = 0 := by
    have he : a x = inner ℝ (x : EucSpace d) (a x) • (x : EucSpace d) := sub_eq_zero.mp hcrit
    rw [he, real_inner_smul_right, real_inner_comm (x : EucSpace d) w, hxw, mul_zero]
  have hd := ha.hasFDerivAt.sub
    (((hasFDerivAt_id (x : EucSpace d)).inner ℝ ha.hasFDerivAt).smul
      (hasFDerivAt_id (x : EucSpace d)))
  simp only [id_eq] at hd
  have he : (fun y => proj d y (a y)) = a - (fun y => inner ℝ y (a y)) • id := rfl
  rw [he, hd.fderiv]
  simp only [sub_apply, add_apply, smul_apply,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.id_apply, ContinuousLinearMap.prod_apply, fderivInnerCLM_apply,
    inner_sub_right, inner_add_right, real_inner_smul_right, hax, hxw, real_inner_self_eq_norm_sq,
    mem_sphere_zero_iff_norm.mp x.2, one_pow]
  ring

/-- Doubling on the actual plane realizes every derivative, factor,
nonzero left eigenvector and unit tangent-eigenvector premise jointly.
Source: arXiv:2411.04990v2, §4, the positive transverse linearization
used in the instability argument for `thm2`. -/
example : ∃ (F : EucSpace 2 → EucSpace 2) (h : EucSpace 2 → ℝ)
    (A : ParamMatrix 2) (x : SSphere 2) (v : EucSpace 2),
    DifferentiableAt ℝ F x ∧ DifferentiableAt ℝ h x ∧
    (∀ y, inner ℝ v (F y) = h y * inner ℝ v y) ∧
    inner ℝ v x = 0 ∧ v ≠ 0 ∧ (0 : ℝ) < 2 ∧
    (∀ w, inner ℝ v (A w) = 2 * inner ℝ v w) ∧
    proj 2 (x : EucSpace 2) (F x) = 0 := by
  let v : EucSpace 2 := EuclideanSpace.single (1 : Fin 2) 1
  refine ⟨fun y => (2 : ℝ) • y, fun _ => 2, (2 : ℝ) • ContinuousLinearMap.id ℝ _,
    basePoint 1, v, (differentiable_id.const_smul (2 : ℝ)).differentiableAt,
    differentiableAt_const (2 : ℝ), fun y => by simp only [real_inner_smul_right],
    ?_, ?_, by norm_num, fun w => by simp only [smul_apply, ContinuousLinearMap.id_apply,
      real_inner_smul_right], ?_⟩
  · simp [v, basePoint, EuclideanSpace.inner_single_left]
  · intro hv
    have he := congrArg (fun z : EucSpace 2 => z (1 : Fin 2)) hv
    simp [v] at he

  · exact proj_smul_self (mem_sphere_zero_iff_norm.mp (basePoint 1).2) (2 : ℝ)

end Transformer.Causal
