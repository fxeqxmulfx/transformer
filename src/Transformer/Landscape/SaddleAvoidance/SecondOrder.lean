/-
# Saddle avoidance — the second-order test

A critical point where the derivative `A` of the gradient satisfies
`c ‖h‖² ≤ ⟪A h, h⟫` with `c > 0` is a local minimizer
(`isLocalMin_of_mul_norm_sq_le_inner`): along each short segment from it,
`f` increases.  For `f` of class `C²` such a `c` exists at every critical
point that is not a strict saddle and where the Hessian `D∇f` is injective
(`exists_pos_mul_norm_sq_le_inner`): the Hessian is symmetric, and its
eigenvalues are neither negative nor zero.
-/

import Transformer.Landscape.SaddleAvoidance.Descent
import Transformer.Landscape.SaddleAvoidance.GradientDescent
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open Filter Metric
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **Second-order test.**  A critical point `z` where the derivative `A` of
the gradient satisfies `c ‖h‖² ≤ ⟪A h, h⟫` with `c > 0` is a local minimizer:
along each short segment from `z` the derivative `⟪∇f(z + t h), h⟫` of `f` is
at least `t c ‖h‖² / 2 ≥ 0`. -/
theorem isLocalMin_of_mul_norm_sq_le_inner (hf : Differentiable ℝ f) {z : E} {A : E →L[ℝ] E}
    (hA : HasFDerivAt (gradient f) A z) (hz : gradient f z = 0) {c : ℝ} (hc : 0 < c)
    (hpos : ∀ h, c * ‖h‖ ^ 2 ≤ ⟪A h, h⟫) : IsLocalMin f z := by
  obtain ⟨δ, hδ, hball⟩ := Metric.eventually_nhds_iff.1 (hA.isLittleO.def (half_pos hc))
  show ∀ᶠ y in 𝓝 z, f z ≤ f y
  filter_upwards [ball_mem_nhds z hδ] with y hy
  set h := y - z with hh
  have hderiv : ∀ t : ℝ, HasDerivAt (fun t : ℝ => f (z + t • h)) ⟪gradient f (z + t • h), h⟫ t := by
    intro t
    have h1 : HasDerivAt (fun t : ℝ => z + t • h) h t := by
      simpa using ((hasDerivAt_id t).smul_const h).const_add z
    have h2 := (hasGradientAt_iff_hasFDerivAt.1 (hf (z + t • h)).hasGradientAt).comp_hasDerivAt t h1
    rw [InnerProductSpace.toDual_apply_apply] at h2
    exact h2
  have hmono : MonotoneOn (fun t : ℝ => f (z + t • h)) (Set.Icc 0 1) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 1)
      (fun t _ => (hderiv t).continuousAt.continuousWithinAt)
      (fun t _ => (hderiv t).differentiableAt.differentiableWithinAt) fun t ht => ?_
    rw [interior_Icc] at ht
    rw [(hderiv t).deriv]
    have hth : ‖t • h‖ ≤ ‖h‖ := by
      rw [norm_smul, Real.norm_of_nonneg ht.1.le]
      exact mul_le_of_le_one_left (norm_nonneg h) ht.2.le
    have hr := hball (y := z + t • h) (by
      rw [dist_eq_norm, add_sub_cancel_left]
      exact hth.trans_lt (by rwa [mem_ball, dist_eq_norm] at hy))
    rw [hz, sub_zero, add_sub_cancel_left] at hr
    have h1 : t * (c * ‖h‖ ^ 2) ≤ ⟪A (t • h), h⟫ := by
      rw [map_smul, real_inner_smul_left]
      exact mul_le_mul_of_nonneg_left (hpos h) ht.1.le
    have h2 := (abs_real_inner_le_norm (gradient f (z + t • h) - A (t • h)) h).trans
      (mul_le_mul_of_nonneg_right hr (norm_nonneg h))
    rw [norm_smul, Real.norm_of_nonneg ht.1.le, inner_sub_left] at h2
    nlinarith [abs_le.1 h2, mul_nonneg (mul_nonneg ht.1.le hc.le) (sq_nonneg ‖h‖)]
  have := hmono ⟨le_rfl, zero_le_one⟩ ⟨zero_le_one, le_rfl⟩ zero_le_one
  simpa [hh] using this

variable [FiniteDimensional ℝ E]

/-- **A critical point that is not a strict saddle, with an injective
Hessian, has a positive definite Hessian.**  For `f` of class `C²` the Hessian
`D∇f(z)` is symmetric and so has an orthonormal basis of eigenvectors; its
eigenvalues are not negative, as `z` is not a strict saddle, nor zero, as it
is injective. -/
theorem exists_pos_mul_norm_sq_le_inner (hf : ContDiff ℝ 2 f) {z : E} (hz : gradient f z = 0)
    (hs : ¬IsStrictSaddle f z) (hinj : Function.Injective (fderiv ℝ (gradient f) z)) :
    ∃ c > 0, ∀ h, c * ‖h‖ ^ 2 ≤ ⟪fderiv ℝ (gradient f) z h, h⟫ := by
  set A : E →ₗ[ℝ] E := ↑(fderiv ℝ (gradient f) z) with hAdef
  show ∃ c > 0, ∀ h, c * ‖h‖ ^ 2 ≤ ⟪A h, h⟫
  have hA : A.IsSymmetric := isSymmetric_fderiv_gradient hf z
  set b := hA.eigenvectorBasis rfl with hb
  set lam := hA.eigenvalues rfl with hlam
  have hAb : ∀ w i, ⟪b i, A w⟫ = lam i * ⟪b i, w⟫ := by
    intro w i
    rw [← hA (b i) w, hb, hA.apply_eigenvectorBasis, real_inner_smul_left]
    rfl
  have hpos : ∀ i, 0 < lam i := by
    intro i
    rcases lt_trichotomy (lam i) 0 with h | h | h
    · exact absurd ⟨hz, lam i, h, hA.hasEigenvalue_eigenvalues rfl i⟩ hs
    · refine absurd (hinj ?_) (b.orthonormal.ne_zero i)
      have h0 : A (b i) = lam i • b i := hA.apply_eigenvectorBasis rfl i
      rw [h, zero_smul] at h0
      rw [map_zero]
      exact h0
    · exact h
  obtain ⟨c, hc, hcl⟩ : ∃ c > 0, ∀ i, c ≤ lam i := by
    rcases isEmpty_or_nonempty (Fin (Module.finrank ℝ E)) with hι | hι
    · exact ⟨1, one_pos, fun i => isEmptyElim i⟩
    · obtain ⟨i, hi⟩ := Finite.exists_min lam
      exact ⟨lam i, hpos i, hi⟩
  refine ⟨c, hc, fun h => ?_⟩
  calc c * ‖h‖ ^ 2 = ∑ i, c * ⟪b i, h⟫ ^ 2 := by rw [norm_sq_eq_sum_inner_sq b h, Finset.mul_sum]
    _ ≤ ∑ i, lam i * ⟪b i, h⟫ ^ 2 :=
      Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_right (hcl i) (sq_nonneg _)
    _ = ⟪A h, h⟫ := by
      rw [← b.sum_inner_mul_inner (A h) h]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [real_inner_comm (b i) (A h), hAb h i]
      ring

/-- The hypotheses of both theorems hold for `f(x) = ‖x‖² / 2` on the line at
its critical point `0`, where the Hessian is the identity: injective with the
single eigenvalue `1`, and `⟪h, h⟫ = ‖h‖²`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    Differentiable ℝ f ∧ HasFDerivAt (gradient f) (ContinuousLinearMap.id ℝ ℝ) 0 ∧
      gradient f 0 = 0 ∧ (0 : ℝ) < 1 ∧
      (∀ h : ℝ, 1 * ‖h‖ ^ 2 ≤ ⟪ContinuousLinearMap.id ℝ ℝ h, h⟫) ∧
      ContDiff ℝ 2 f ∧ ¬IsStrictSaddle f 0 ∧ Function.Injective (fderiv ℝ (gradient f) 0) := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hd : fderiv ℝ (gradient f) 0 = ContinuousLinearMap.id ℝ ℝ := by
    rw [hg]
    exact fderiv_id
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero,
    by rw [hg]; exact hasFDerivAt_id 0, by rw [hg]; rfl, one_pos, fun h => ?_,
    contDiff_const.mul (contDiff_norm_sq ℝ), ?_, by rw [hd]; exact Function.injective_id⟩
  · rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq, one_mul]
  · rintro ⟨-, μ, hμ, hev⟩
    obtain ⟨v, hv, hv0⟩ := hev.exists_hasEigenvector
    have h1 := Module.End.mem_eigenspace_iff.1 hv
    rw [hd] at h1
    simp only [ContinuousLinearMap.coe_id, LinearMap.id_apply, smul_eq_mul] at h1
    have h2 : (1 - μ) * v = 0 := by
      rw [sub_mul, one_mul, sub_eq_zero]
      exact h1
    rcases mul_eq_zero.1 h2 with h3 | h3
    · linarith
    · exact hv0 h3

end Transformer.Landscape.SaddleAvoidance
