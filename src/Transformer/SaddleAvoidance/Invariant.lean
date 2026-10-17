/-
# Saddle avoidance — the Łojasiewicz inequality under a translation symmetry

Let `f` be invariant under the translations by a subspace `W`.  Its gradient
is then orthogonal to `W` (`inner_gradient_eq_zero_of_invariant`), so the
derivative of the gradient maps into `Wᗮ`
(`mem_orthogonal_of_hasFDerivAt_gradient`), and it vanishes on `W`: no
critical point is nondegenerate when `W ≠ 0`.  The Łojasiewicz inequality
with exponent `1/2` still holds at a critical point `z` where the Hessian is
injective on `Wᗮ` (`isLojasiewiczAt_of_invariant`).  The proof restricts `f`
to the slice `z + Wᗮ`, where the Hessian is injective, applies
`isLojasiewiczAt_of_injective` there, and carries the inequality back along
the orthogonal projection onto the slice, which changes neither the value of
`f` nor its gradient.

This covers the synchronized states of `𝖤_β` on angles, which is invariant
under adding one constant to every angle, `W = ℝ𝟙`.
-/

import Transformer.SaddleAvoidance.CriticalPoints
import Mathlib.Analysis.Calculus.LineDeriv.Basic

open Filter Metric InnerProductSpace
open scoped Topology

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {f : E → ℝ}
  {W : Submodule ℝ E}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **The gradient of an invariant function is orthogonal to the
symmetry**: if `f` is invariant under the translations by `W` and
differentiable at `y`, then `⟪∇f(y), w⟫ = 0` for `w ∈ W`, since `f` is
constant on the line `y + ℝ w`. -/
theorem inner_gradient_eq_zero_of_invariant [CompleteSpace E] (hW : ∀ w ∈ W, ∀ y, f (y + w) = f y)
    {y : E} (hf : DifferentiableAt ℝ f y) {w : E} (hw : w ∈ W) : ⟪gradient f y, w⟫ = 0 := by
  have hline : (fun t : ℝ => f (y + t • w)) = fun _ => f y :=
    funext fun t => hW _ (W.smul_mem t hw) y
  have h := hf.lineDeriv_eq_fderiv (v := w)
  rw [lineDeriv, hline, deriv_const, hf.hasGradientAt.hasFDerivAt.fderiv,
    toDual_apply_apply] at h
  exact h.symm

/-- **The Hessian of an invariant function maps into `Wᗮ`**: if `f` is
differentiable and invariant under the translations by `W`, every derivative
`A` of its gradient has `A v ∈ Wᗮ`, since the gradient takes values in the
closed subspace `Wᗮ`. -/
theorem mem_orthogonal_of_hasFDerivAt_gradient [CompleteSpace E] (hf : Differentiable ℝ f)
    (hW : ∀ w ∈ W, ∀ y, f (y + w) = f y) {z : E} {A : E →L[ℝ] E}
    (hA : HasFDerivAt (gradient f) A z) (v : E) : A v ∈ Wᗮ := by
  rw [Submodule.mem_orthogonal]
  intro w hw
  have h := (innerSL ℝ w).hasFDerivAt.comp z hA
  have hzero : (innerSL ℝ w ∘ gradient f) = fun _ => 0 := funext fun y => by
    rw [Function.comp_apply, innerSL_apply_apply, real_inner_comm]
    exact inner_gradient_eq_zero_of_invariant hW (hf y) hw
  rw [hzero] at h
  have := congrArg (fun T : E →L[ℝ] ℝ => T v) ((hasFDerivAt_const 0 z).unique h)
  simpa using this.symm

/-- **The Łojasiewicz inequality at a critical point nondegenerate
transversally to a symmetry**, with exponent `1/2`.  Let `f` be
differentiable and invariant under the translations by a subspace `W`.  If
`∇f(z) = 0` and the derivative `A` of the gradient at `z` is injective on
`Wᗮ`, then `IsLojasiewiczAt f z (1/2)`.  This is `isLojasiewiczAt_of_injective`
applied to the restriction of `f` to the slice `z + Wᗮ`. -/
theorem isLojasiewiczAt_of_invariant [FiniteDimensional ℝ E] (hf : Differentiable ℝ f)
    (hW : ∀ w ∈ W, ∀ y, f (y + w) = f y) {z : E} {A : E →L[ℝ] E}
    (hA : HasFDerivAt (gradient f) A z) (hz : gradient f z = 0)
    (hinj : ∀ v ∈ Wᗮ, A v = 0 → v = 0) : IsLojasiewiczAt f z (1 / 2) := by
  set P := Wᗮ.orthogonalProjectionOnto
  set g : Wᗮ → ℝ := fun v => f (z + v)
  have hslice : ∀ v : Wᗮ, HasFDerivAt (fun v : Wᗮ => z + (v : E)) Wᗮ.subtypeL v :=
    fun v => Wᗮ.subtypeL.hasFDerivAt.const_add z
  have hgrad : ∀ v : Wᗮ, gradient g v = P (gradient f (z + v)) := by
    intro v
    have hd : toDual ℝ Wᗮ (P (gradient f (z + v))) = (fderiv ℝ f (z + v)).comp Wᗮ.subtypeL := by
      ext u
      rw [toDual_apply_apply, Submodule.inner_orthogonalProjectionOnto_eq_of_mem_right,
        ContinuousLinearMap.comp_apply, (hf (z + v)).hasGradientAt.hasFDerivAt.fderiv,
        toDual_apply_apply, Submodule.subtypeL_apply]
    refine HasGradientAt.gradient ?_
    rw [hasGradientAt_iff_hasFDerivAt, hd]
    exact (hf (z + v)).hasFDerivAt.comp v (hslice v)
  have hB : HasFDerivAt (gradient g) (P.comp (A.comp Wᗮ.subtypeL)) 0 := by
    rw [funext hgrad]
    have h0 : HasFDerivAt (gradient f) A (z + ((0 : Wᗮ) : E)) := by
      rwa [Submodule.coe_zero, add_zero]
    exact P.hasFDerivAt.comp 0 (HasFDerivAt.comp (f := fun v : Wᗮ => z + (v : E)) 0 h0 (hslice 0))
  have hinj' : Function.Injective (P.comp (A.comp Wᗮ.subtypeL)) := by
    refine (injective_iff_map_eq_zero _).2 fun v hv => ?_
    have h1 : A v ∈ Wᗮᗮ := Submodule.orthogonalProjectionOnto_eq_zero_iff.1 hv
    have h2 : A v ∈ Wᗮ ⊓ Wᗮᗮ := ⟨mem_orthogonal_of_hasFDerivAt_gradient hf hW hA v, h1⟩
    rw [Submodule.inf_orthogonal_eq_bot, Submodule.mem_bot] at h2
    exact Subtype.ext (hinj v v.2 h2)
  have hg : Differentiable ℝ g := fun v => (hf _).comp v (hslice v).differentiableAt
  have hg0 : gradient g 0 = 0 := by
    rw [hgrad, Submodule.coe_zero, add_zero, hz, map_zero]
  obtain ⟨m, hm, ε, hε, hev⟩ := isLojasiewiczAt_of_injective hg hB hg0 hinj'
  refine ⟨m, hm, ε, hε, ?_⟩
  have hφ : Tendsto (fun y => P (y - z)) (𝓝 z) (𝓝 0) := by
    have h := ((P.continuous.comp (continuous_sub_right z)).tendsto z)
    rwa [Function.comp_apply, sub_self, map_zero] at h
  filter_upwards [hφ.eventually hev] with y hy
  have hw : y - z - Wᗮ.starProjection (y - z) ∈ W := by
    have h := Wᗮ.sub_starProjection_mem_orthogonal (y - z)
    rwa [Submodule.orthogonal_orthogonal] at h
  have hsplit : z + (P (y - z) : E) + (y - z - Wᗮ.starProjection (y - z)) = y := by
    rw [Submodule.coe_orthogonalProjectionOnto_apply]
    abel
  have hfy : g (P (y - z)) = f y := by
    have h := hW _ hw (z + (P (y - z) : E))
    rw [hsplit] at h
    exact h.symm
  have hgy : ‖gradient g (P (y - z))‖ ≤ ‖gradient f y‖ := by
    have h := gradient_add_eq (hW _ hw) (z + (P (y - z) : E))
    rw [hsplit] at h
    rw [hgrad, ← h]
    exact Wᗮ.norm_orthogonalProjectionOnto_apply_le _
  have hgz : g 0 = f z := by
    simp only [g, Submodule.coe_zero, add_zero]
  rw [hfy, hgz] at hy
  exact fun h1 h2 => (hy h1 h2).trans hgy

/-- The hypotheses of the theorems above hold for half the squared distance
to the diagonal `W = ℝ (1, 1)` of the plane, `f x = ‖P x‖² / 2` with `P` the
orthogonal projection onto `Wᗮ`, at its critical point `0`: `f` is invariant
under `W`, its gradient is `P`, and `P` is injective on `Wᗮ`. -/
example : let W : Submodule ℝ (EuclideanSpace ℝ (Fin 2)) :=
      ℝ ∙ (EuclideanSpace.single 0 1 + EuclideanSpace.single 1 1)
    let f := fun x => (1 / 2 : ℝ) * ‖Wᗮ.starProjection x‖ ^ 2
    EuclideanSpace.single 0 1 + EuclideanSpace.single 1 1 ∈ W ∧ Differentiable ℝ f ∧
      (∀ w ∈ W, ∀ y, f (y + w) = f y) ∧ HasFDerivAt (gradient f) Wᗮ.starProjection 0 ∧
      gradient f 0 = 0 ∧ ∀ v ∈ Wᗮ, Wᗮ.starProjection v = 0 → v = 0 := by
  intro W f
  set P := Wᗮ.starProjection
  have hf : ∀ x, HasFDerivAt f ((1 / 2 : ℝ) • (2 • (innerSL ℝ (P x)).comp P)) x :=
    fun x => P.hasFDerivAt.norm_sq.const_mul (1 / 2)
  have hgrad : gradient f = P := by
    funext x
    refine HasGradientAt.gradient ?_
    rw [hasGradientAt_iff_hasFDerivAt]
    convert hf x using 1
    ext h
    have hPP : P (P x) = P x :=
      Submodule.starProjection_eq_self_iff.2 (Wᗮ.starProjection_apply_mem x)
    rw [toDual_apply_apply, smul_apply, smul_apply, ContinuousLinearMap.comp_apply,
      innerSL_apply_apply, ← Submodule.inner_starProjection_left_eq_right, hPP, smul_eq_mul,
      two_smul]
    ring
  refine ⟨Submodule.mem_span_singleton_self _, fun x => (hf x).differentiableAt, fun w hw y => ?_,
    by rw [hgrad]; exact P.hasFDerivAt, by rw [hgrad]; exact map_zero P, fun v hv hPv => ?_⟩
  · have hPw : P w = 0 :=
      (Submodule.starProjection_apply_eq_zero_iff Wᗮ).2 (Submodule.le_orthogonal_orthogonal W hw)
    show (1 / 2 : ℝ) * ‖P (y + w)‖ ^ 2 = (1 / 2 : ℝ) * ‖P y‖ ^ 2
    rw [map_add, hPw, add_zero]
  · rwa [Submodule.starProjection_eq_self_iff.2 hv] at hPv

end Transformer.SaddleAvoidance
