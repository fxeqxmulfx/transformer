/-
# Saddle avoidance — symmetries

A linear isometry `g` of `E` onto itself with `f ∘ g = f` carries the gradient
of `f` along: `∇f(g x) = g ∇f(x)` (`gradient_apply_of_invariant`); in finite
dimensions every linear isometry of `E` into itself is onto
(`LinearIsometry.toLinearIsometryEquiv`).  So `g` commutes
with gradient steps and their iterates (`gradientStep_apply_of_invariant`,
`iterate_gradientStep_apply_of_invariant`) and, when `∇f` is bounded and
Lipschitz, with the gradient flow (`gradientFlow_apply_of_invariant`).

The vectors fixed by a set `S` of such isometries, a group for instance, form
a subspace `fixedSubspace S`.  The gradient, gradient descent and the gradient
flow map it into itself (`mapsTo_gradient_fixedSubspace`,
`mapsTo_gradientStep_fixedSubspace`, `mapsTo_gradientFlow_fixedSubspace`).  A
point of it at which the derivative of `f` vanishes along it is a critical
point of `f` (`gradient_eq_zero_of_mem_fixedSubspace`), the principle of
symmetric criticality.  The fixed subspace is an inner product space of its
own, closed in `E` (`isClosed_fixedSubspace`) and so complete when `E` is.
Where `f` is differentiable, the gradient of `f` restricted to it is `∇f`
(`coe_gradient_restrict`), and the orbits of gradient descent of the
restriction are those of `f` (`coe_iterate_gradientStep_restrict`).  The
theorems on gradient descent then apply on the fixed subspace, for its own
Lebesgue measure.  In `E` a proper subspace is null
(`MeasureTheory.Measure.addHaar_submodule`), so a bridge to `E` also needs the
subspace to attract.

Source: item M6.10 of `landscape_plan.md`.
-/

import Transformer.SaddleAvoidance.TimeOne
import Transformer.Normalization.LojasiewiczInnerProduct

open Set
open scoped NNReal

namespace Transformer.SaddleAvoidance

open GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The subspace of the vectors fixed by every isometry in `S`. -/
def fixedSubspace (S : Set (E ≃ₗᵢ[ℝ] E)) : Submodule ℝ E where
  carrier := {x | ∀ g ∈ S, g x = x}
  add_mem' hx hy g hg := by rw [map_add, hx g hg, hy g hg]
  zero_mem' g _ := map_zero g
  smul_mem' c _ hx g hg := by rw [map_smul, hx g hg]

/-- A vector is in the fixed subspace when every isometry in `S` fixes it. -/
theorem mem_fixedSubspace {S : Set (E ≃ₗᵢ[ℝ] E)} {x : E} :
    x ∈ fixedSubspace S ↔ ∀ g ∈ S, g x = x :=
  Iff.rfl

/-- The fixed subspace is closed: each `{x | g x = x}` is. -/
theorem isClosed_fixedSubspace (S : Set (E ≃ₗᵢ[ℝ] E)) : IsClosed (fixedSubspace S : Set E) := by
  have h : (fixedSubspace S : Set E) = ⋂ g ∈ S, {x | g x = x} := by
    ext x
    simp [mem_fixedSubspace]
  rw [h]
  exact isClosed_biInter fun g _ => isClosed_eq g.continuous continuous_id

variable [CompleteSpace E] {f : E → ℝ} {g : E ≃ₗᵢ[ℝ] E}

/-- The fixed subspace of a complete space is complete, so `f` restricted to it
has a gradient there. -/
instance completeSpace_fixedSubspace (S : Set (E ≃ₗᵢ[ℝ] E)) : CompleteSpace (fixedSubspace S) :=
  (isClosed_fixedSubspace S).completeSpace_coe

/-- **An isometry preserving `f` carries its gradient along**:
`∇f(g x) = g ∇f(x)`, at every point, differentiable or not.  Since
`f ∘ g⁻¹ = f`, this is `Normalization.gradient_comp_linearIsometryEquiv_symm`. -/
theorem gradient_apply_of_invariant (hg : ∀ x, f (g x) = f x) (x : E) :
    gradient f (g x) = g (gradient f x) := by
  have hfg : f ∘ g.symm = f := funext fun y => by
    rw [Function.comp_apply, ← hg (g.symm y), g.apply_symm_apply]
  have h := Normalization.gradient_comp_linearIsometryEquiv_symm g f (g x)
  rwa [hfg, g.symm_apply_apply] at h

/-- **Gradient steps commute with an isometry preserving `f`**:
`g x - η ∇f(g x) = g (x - η ∇f(x))`. -/
theorem gradientStep_apply_of_invariant (hg : ∀ x, f (g x) = f x) (η : ℝ) (x : E) :
    gradientStep f η (g x) = g (gradientStep f η x) := by
  rw [gradientStep, gradientStep, gradient_apply_of_invariant hg, map_sub, map_smul]

/-- **Orbits of gradient descent commute with an isometry preserving `f`.** -/
theorem iterate_gradientStep_apply_of_invariant (hg : ∀ x, f (g x) = f x) (η : ℝ) (k : ℕ)
    (x : E) : (gradientStep f η)^[k] (g x) = g ((gradientStep f η)^[k] x) :=
  (Function.Semiconj.iterate_right (fun y => (gradientStep_apply_of_invariant hg η y).symm)
    k x).symm

/-- **The gradient flow commutes with an isometry preserving `f`**, when `∇f`
is `L`-Lipschitz and bounded by `M`: `t ↦ g Φ^t(x)` solves `ẋ = -∇f(x)`,
since `g (-∇f(y)) = -∇f(g y)`, and starts at `g x`. -/
theorem gradientFlow_apply_of_invariant {L : ℝ≥0} {M : ℝ} (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) (hg : ∀ x, f (g x) = f x) (t : ℝ) (x : E) :
    gradientFlow f t (g x) = g (gradientFlow f t x) := by
  have hγ : ∀ τ, HasDerivAt (fun τ => g (gradientFlow f τ x))
      (-gradient f (g (gradientFlow f τ x))) τ := fun τ => by
    rw [gradient_apply_of_invariant hg, ← map_neg]
    exact g.hasFDerivAt.comp_hasDerivAt τ (hasDerivAt_gradientFlow hL hM τ x)
  have h := (isFlowField_neg_gradient hL hM).flow_eq hγ 0 t
  have h0 : gradientFlow f 0 x = x := (isFlowField_neg_gradient hL hM).flow_self 0 x
  rw [h0] at h
  exact h

variable {S : Set (E ≃ₗᵢ[ℝ] E)}

/-- **The gradient maps the fixed subspace into itself**: for `x` fixed by
every `g ∈ S`, `g ∇f(x) = ∇f(g x) = ∇f(x)`. -/
theorem mapsTo_gradient_fixedSubspace (hS : ∀ g ∈ S, ∀ x, f (g x) = f x) :
    MapsTo (gradient f) (fixedSubspace S) (fixedSubspace S) := fun x hx g hg => by
  rw [← gradient_apply_of_invariant (hS g hg), hx g hg]

/-- **Gradient descent keeps the fixed subspace.** -/
theorem mapsTo_gradientStep_fixedSubspace (hS : ∀ g ∈ S, ∀ x, f (g x) = f x) (η : ℝ) :
    MapsTo (gradientStep f η) (fixedSubspace S) (fixedSubspace S) := fun _ hx =>
  (fixedSubspace S).sub_mem hx ((fixedSubspace S).smul_mem η (mapsTo_gradient_fixedSubspace hS hx))

/-- **The gradient flow keeps the fixed subspace**, when `∇f` is bounded and
Lipschitz: `g Φ^t(x) = Φ^t(g x) = Φ^t(x)`. -/
theorem mapsTo_gradientFlow_fixedSubspace {L : ℝ≥0} {M : ℝ} (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) (hS : ∀ g ∈ S, ∀ x, f (g x) = f x) (t : ℝ) :
    MapsTo (gradientFlow f t) (fixedSubspace S) (fixedSubspace S) := fun x hx g hg => by
  rw [← gradientFlow_apply_of_invariant hL hM (hS g hg), hx g hg]

/-- **The principle of symmetric criticality.**  A point `x` of the fixed
subspace at which the derivative of `f` vanishes along the fixed subspace is
a critical point of `f`: `∇f(x)` lies in the fixed subspace, so
`‖∇f(x)‖² = Df(x) ∇f(x) = 0`. -/
theorem gradient_eq_zero_of_mem_fixedSubspace (hS : ∀ g ∈ S, ∀ x, f (g x) = f x) {x : E}
    (hx : x ∈ fixedSubspace S) (h : ∀ v ∈ fixedSubspace S, fderiv ℝ f x v = 0) :
    gradient f x = 0 := by
  have hv := h _ (mapsTo_gradient_fixedSubspace hS hx)
  rwa [← toDual_gradient, InnerProductSpace.toDual_apply_apply, inner_self_eq_zero] at hv

/-- **Restricting `f` to the fixed subspace restricts its gradient**: where `f`
is differentiable, the gradient of `f` on the fixed subspace, an inner product
space of its own, is `∇f`, which lies in it. -/
theorem coe_gradient_restrict (hS : ∀ g ∈ S, ∀ x, f (g x) = f x) {v : fixedSubspace S}
    (hf : DifferentiableAt ℝ f v) :
    ((gradient (fun u : fixedSubspace S => f u) v : fixedSubspace S) : E) = gradient f v := by
  have h : HasGradientAt (fun u : fixedSubspace S => f u)
      ⟨gradient f v, mapsTo_gradient_fixedSubspace hS v.2⟩ v := by
    rw [hasGradientAt_iff_hasFDerivAt]
    refine (hf.hasFDerivAt.comp v (fixedSubspace S).subtypeL.hasFDerivAt).congr_fderiv ?_
    ext u
    rw [ContinuousLinearMap.comp_apply, Submodule.subtypeL_apply, ← toDual_gradient,
      InnerProductSpace.toDual_apply_apply, InnerProductSpace.toDual_apply_apply,
      Submodule.coe_inner]
  rw [h.gradient]

/-- **Gradient descent on the fixed subspace is the restriction of gradient
descent**, for differentiable `f`: its orbits are the orbits of gradient
descent on `E` that start in the fixed subspace, so the theorems on gradient
descent apply to them with the Lebesgue measure of the fixed subspace. -/
theorem coe_iterate_gradientStep_restrict (hS : ∀ g ∈ S, ∀ x, f (g x) = f x)
    (hf : Differentiable ℝ f) (η : ℝ) (k : ℕ) (v : fixedSubspace S) :
    (((gradientStep (fun u : fixedSubspace S => f u) η)^[k] v : fixedSubspace S) : E) =
      (gradientStep f η)^[k] v := by
  have h : Function.Semiconj (Subtype.val : fixedSubspace S → E)
      (gradientStep (fun u : fixedSubspace S => f u) η) (gradientStep f η) := fun u => by
    rw [gradientStep, gradientStep, Submodule.coe_sub, Submodule.coe_smul,
      coe_gradient_restrict hS (hf u)]
  exact h.iterate_right k v

/-- The hypotheses of the theorems above hold for `cos` on the line, which is
even, with `g = -id`, the set `S = {-id}` and its fixed point `0`; `∇ cos = -sin`
is `1`-Lipschitz and bounded by `1`. -/
example : let g := LinearIsometryEquiv.neg ℝ (E := ℝ)
    (∀ x, Real.cos (g x) = Real.cos x) ∧
      (∀ g' ∈ ({g} : Set (ℝ ≃ₗᵢ[ℝ] ℝ)), ∀ x, Real.cos (g' x) = Real.cos x) ∧
      LipschitzWith 1 (gradient Real.cos) ∧ (∀ x, ‖gradient Real.cos x‖ ≤ 1) ∧
      Differentiable ℝ Real.cos ∧ (0 : ℝ) ∈ fixedSubspace {g} ∧
      ∀ v ∈ fixedSubspace {g}, fderiv ℝ Real.cos 0 v = 0 := by
  intro g
  have hgrad : gradient Real.cos = fun x => -Real.sin x :=
    funext fun x => (Real.hasDerivAt_cos x).hasGradientAt'.gradient
  have hg : ∀ x, Real.cos (g x) = Real.cos x := fun x => Real.cos_neg x
  refine ⟨hg, fun g' hg' => Set.mem_singleton_iff.1 hg' ▸ hg,
    by rw [hgrad]; exact Real.lipschitzWith_sin.neg, fun x => ?_, Real.differentiable_cos,
    (fixedSubspace {g}).zero_mem, fun v _ => ?_⟩
  · rw [hgrad, norm_neg, Real.norm_eq_abs]
    exact Real.abs_sin_le_one x
  · rw [(Real.hasDerivAt_cos 0).hasFDerivAt.fderiv]
    simp

end Transformer.SaddleAvoidance
