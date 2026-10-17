/-
# Saddle avoidance — necessary conditions for a local minimizer

At a local minimizer `z` of a differentiable `f` the gradient vanishes
(`gradient_eq_zero_of_isLocalMin`), and the derivative `A` of the gradient
at `z` has `⟪A h, h⟫ ≥ 0` for every `h` (`inner_nonneg_of_isLocalMin`):
along a direction with `⟪A h, h⟫ < 0`, `f` decreases.  For `f` of class `C²`
on a finite-dimensional space, a critical point with such a direction is a
strict saddle (`isStrictSaddle_of_inner_neg`): the Hessian is symmetric, and
`⟪A h, h⟫` is a combination of its eigenvalues with nonnegative weights.
-/

import Transformer.SaddleAvoidance.Descent
import Transformer.SaddleAvoidance.GradientDescent
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open Filter Metric
open scoped Topology

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- The gradient vanishes at a local minimizer. -/
theorem gradient_eq_zero_of_isLocalMin {z : E} (hmin : IsLocalMin f z) : gradient f z = 0 := by
  simp [gradient, hmin.fderiv_eq_zero]

/-- **Second-order necessary condition.**  At a local minimizer `z` of a
differentiable `f` whose gradient has the derivative `A` at `z`,
`0 ≤ ⟪A h, h⟫` for every `h`: if `⟪A h, h⟫ = -c < 0`, the derivative
`⟪∇f(z + t h), h⟫` of `f` along `z + t h` is below `-t c / 2` for small
`t > 0`, and `f` drops below `f z`. -/
theorem inner_nonneg_of_isLocalMin (hf : Differentiable ℝ f) {z : E} {A : E →L[ℝ] E}
    (hA : HasFDerivAt (gradient f) A z) (hmin : IsLocalMin f z) (h : E) : 0 ≤ ⟪A h, h⟫ := by
  by_contra hneg
  push Not at hneg
  have hh : h ≠ 0 := by
    rintro rfl
    simp at hneg
  have hn : 0 < ‖h‖ := norm_pos_iff.2 hh
  set c := -⟪A h, h⟫ with hc
  have hc0 : 0 < c := by linarith
  have hz := gradient_eq_zero_of_isLocalMin hmin
  obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1
    (hA.isLittleO.def (show 0 < c / (2 * ‖h‖ ^ 2) by positivity))
  obtain ⟨ρ, hρ, hle⟩ := Metric.eventually_nhds_iff.1 hmin
  set s := min δ ρ / (2 * ‖h‖) with hs
  have hs0 : 0 < s := by positivity
  have hsh : s * ‖h‖ = min δ ρ / 2 := by rw [hs]; field_simp
  have hderiv : ∀ t : ℝ, HasDerivAt (fun t : ℝ => f (z + t • h)) ⟪gradient f (z + t • h), h⟫ t := by
    intro t
    have h1 : HasDerivAt (fun t : ℝ => z + t • h) h t := by
      simpa using ((hasDerivAt_id t).smul_const h).const_add z
    have h2 := (hasGradientAt_iff_hasFDerivAt.1 (hf (z + t • h)).hasGradientAt).comp_hasDerivAt t h1
    rw [InnerProductSpace.toDual_apply_apply] at h2
    exact h2
  have hanti : StrictAntiOn (fun t : ℝ => f (z + t • h)) (Set.Icc 0 s) := by
    refine strictAntiOn_of_deriv_neg (convex_Icc 0 s)
      (fun t _ => (hderiv t).continuousAt.continuousWithinAt) fun t ht => ?_
    rw [interior_Icc] at ht
    rw [(hderiv t).deriv]
    have hth : ‖t • h‖ = t * ‖h‖ := by rw [norm_smul, Real.norm_of_nonneg ht.1.le]
    have hr := hball (y := z + t • h) (by
      rw [dist_eq_norm, add_sub_cancel_left, hth]
      have : t * ‖h‖ < s * ‖h‖ := mul_lt_mul_of_pos_right ht.2 hn
      have : min δ ρ ≤ δ := min_le_left δ ρ
      linarith)
    rw [hz, sub_zero, add_sub_cancel_left, hth] at hr
    have h2 := (abs_real_inner_le_norm (gradient f (z + t • h) - A (t • h)) h).trans
      (mul_le_mul_of_nonneg_right hr (norm_nonneg h))
    rw [inner_sub_left, map_smul, real_inner_smul_left] at h2
    have h3 : c / (2 * ‖h‖ ^ 2) * (t * ‖h‖) * ‖h‖ = t * c / 2 := by field_simp
    rw [h3] at h2
    nlinarith [abs_le.1 h2, ht.1]
  have h1 := hanti ⟨le_rfl, hs0.le⟩ ⟨hs0.le, le_rfl⟩ hs0
  simp only [zero_smul, add_zero] at h1
  have h2 := hle (y := z + s • h) (by
    rw [dist_eq_norm, add_sub_cancel_left, norm_smul, Real.norm_of_nonneg hs0.le, hsh]
    have : min δ ρ ≤ ρ := min_le_right δ ρ
    have : 0 < min δ ρ := lt_min hδ hρ
    linarith)
  linarith

variable [FiniteDimensional ℝ E]

/-- **A critical point with a direction of negative curvature is a strict
saddle**: for `f` of class `C²`, if `⟪A h, h⟫ < 0` for the Hessian `A` at a
critical point `z`, then `A` has a negative eigenvalue.  Otherwise, in an
orthonormal basis `b` of eigenvectors with eigenvalues `λ_i ≥ 0`,
`⟪A h, h⟫ = Σ_i λ_i ⟪b_i, h⟫² ≥ 0`. -/
theorem isStrictSaddle_of_inner_neg (hf : ContDiff ℝ 2 f) {z : E} (hz : gradient f z = 0)
    {h : E} (hh : ⟪fderiv ℝ (gradient f) z h, h⟫ < 0) : IsStrictSaddle f z := by
  set A : E →ₗ[ℝ] E := ↑(fderiv ℝ (gradient f) z) with hAdef
  change ⟪A h, h⟫ < 0 at hh
  have hA : A.IsSymmetric := isSymmetric_fderiv_gradient hf z
  set b := hA.eigenvectorBasis rfl with hb
  set lam := hA.eigenvalues rfl with hlam
  refine ⟨hz, ?_⟩
  by_contra hneg
  push Not at hneg
  have hpos : ∀ i, 0 ≤ lam i := fun i =>
    not_lt.1 fun hi => hneg (lam i) hi (hA.hasEigenvalue_eigenvalues rfl i)
  have hAb : ∀ i, ⟪A h, b i⟫ = lam i * ⟪b i, h⟫ := by
    intro i
    rw [real_inner_comm, ← hA (b i) h, hb, hA.apply_eigenvectorBasis, real_inner_smul_left]
    rfl
  have h0 : 0 ≤ ⟪A h, h⟫ := by
    rw [← b.sum_inner_mul_inner (A h) h]
    refine Finset.sum_nonneg fun i _ => ?_
    rw [hAb i, mul_assoc]
    exact mul_nonneg (hpos i) (mul_self_nonneg _)
  linarith

/-- The hypotheses of the first two theorems above hold for `f(x) = ‖x‖² / 2`
on the line at its minimizer `0`, where the Hessian is the identity. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    Differentiable ℝ f ∧ HasFDerivAt (gradient f) (ContinuousLinearMap.id ℝ ℝ) 0 ∧
      IsLocalMin f 0 := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  exact ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero,
    by rw [hg]; exact hasFDerivAt_id 0,
    Filter.Eventually.of_forall fun y => by simp only [f, norm_zero]; nlinarith [sq_nonneg ‖y‖]⟩

/-- The hypotheses of `isStrictSaddle_of_inner_neg` hold for
`f(x) = -‖x‖² / 2` on the line at its critical point `0`, where the Hessian
is `-1`, and `h = 1`. -/
example : let f := fun x : ℝ => (-1 / 2 : ℝ) * ‖x‖ ^ 2
    ContDiff ℝ 2 f ∧ gradient f 0 = 0 ∧ ⟪fderiv ℝ (gradient f) 0 1, 1⟫ < 0 := by
  intro f
  have hg : gradient f = fun x => -x := by
    funext y
    apply HasGradientAt.gradient
    rw [hasGradientAt_iff_hasFDerivAt]
    convert (hasStrictFDerivAt_norm_sq y).hasFDerivAt.const_mul (-1 / 2 : ℝ) using 1
    ext
    simp only [InnerProductSpace.toDual_apply_apply, smul_apply, innerSL_apply_apply,
      smul_eq_mul, inner_neg_left]
    ring
  have hd : fderiv ℝ (gradient f) 0 = -ContinuousLinearMap.id ℝ ℝ := by
    rw [hg]
    exact (hasFDerivAt_id (0 : ℝ)).neg.fderiv
  refine ⟨contDiff_const.mul (contDiff_norm_sq ℝ), by rw [hg]; exact neg_zero, ?_⟩
  rw [hd]
  simp

end Transformer.SaddleAvoidance
