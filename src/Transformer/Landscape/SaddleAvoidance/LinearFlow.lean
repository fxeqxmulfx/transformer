/-
# Saddle avoidance — the linear flow `exp (τ A)`

For a continuous linear map `A` of a real Banach space, `J τ = exp (τ A)`
solves the linear equation `J' = A J` with `J 0 = id` (`exp_zero_smul`,
`hasDerivAt_exp_smul`, `continuous_exp_smul`).  This is the variational
equation of the flow of a field at a fixed point where the field's derivative
is `A`, so `exp (τ A)` is the derivative of the flow there.

An eigenvector of `A` with eigenvalue `μ` is an eigenvector of `exp A` with
eigenvalue `e^μ`: `A` multiplies it by `μ`, so each term of the exponential
series does by `μⁿ / n!` (`hasEigenvector_exp`, `hasEigenvalue_exp`).  On a Hilbert
space, `exp A` is symmetric when `A` is (`isSymmetric_exp`), since the
adjoint commutes with the series.

For the gradient flow of `f` at a critical point `z`, `A = -∇²f(z)`: a
negative eigenvalue `λ` of the Hessian gives the eigenvalue `e^{-λ} > 1` of
the derivative of the time-one map.  This is the step that the proof of
`l:nosaddleconv` in Appendix A of arXiv:2312.10794v5 takes when it notes
that the Jacobian of a gradient field is, at a critical point, the Hessian.
-/

import Transformer.Landscape.SaddleAvoidance.Defs
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.InnerProductSpace.Adjoint

open NormedSpace

namespace Transformer.Landscape.SaddleAvoidance

section Banach

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The linear flow starts at the identity: `exp (0 • A) = id`. -/
theorem exp_zero_smul (A : E →L[ℝ] E) :
    exp ((0 : ℝ) • A) = ContinuousLinearMap.id ℝ E := by
  rw [zero_smul, exp_zero, ContinuousLinearMap.one_def]

variable [CompleteSpace E]

/-- **The linear flow solves `J' = A J`**: the derivative of `τ ↦ exp (τ A)`
at `t` is `A ∘ exp (t A)`. -/
theorem hasDerivAt_exp_smul (A : E →L[ℝ] E) (t : ℝ) :
    HasDerivAt (fun τ : ℝ => exp (τ • A)) (A.comp (exp (t • A))) t :=
  hasDerivAt_exp_smul_const' A t

theorem continuous_exp_smul (A : E →L[ℝ] E) : Continuous fun τ : ℝ => exp (τ • A) :=
  continuous_iff_continuousAt.2 fun t => (hasDerivAt_exp_smul A t).continuousAt

/-- **An eigenvector of `A` with eigenvalue `μ` is an eigenvector of `exp A`
with eigenvalue `e^μ`.**  The exponential series of `A`, applied to `v`, is
the exponential series of `μ` times `v`. -/
theorem hasEigenvector_exp {A : E →L[ℝ] E} {μ : ℝ} {v : E}
    (h : Module.End.HasEigenvector (A : E →ₗ[ℝ] E) μ v) :
    Module.End.HasEigenvector ((exp A : E →L[ℝ] E) : E →ₗ[ℝ] E) (Real.exp μ) v := by
  refine ⟨Module.End.mem_eigenspace_iff.2 ?_, h.2⟩
  have hA : HasSum (fun n : ℕ => ((n.factorial : ℝ)⁻¹ * μ ^ n) • v) (exp A v) := by
    have hs := (exp_series_hasSum_exp' (𝕂 := ℝ) A).mapL (ContinuousLinearMap.apply ℝ E v)
    refine hs.congr_fun fun n => ?_
    have hpow : (A ^ n) v = μ ^ n • v := by
      rw [← ContinuousLinearMap.coe_coe, ContinuousLinearMap.toLinearMap_pow, h.pow_apply]
    simp only [ContinuousLinearMap.apply_apply, smul_apply, hpow, smul_smul]
  have hμ : HasSum (fun n : ℕ => ((n.factorial : ℝ)⁻¹ * μ ^ n) • v) (Real.exp μ • v) := by
    rw [Real.exp_eq_exp_ℝ]
    simpa only [smul_eq_mul] using (exp_series_hasSum_exp' (𝕂 := ℝ) μ).smul_const v
  exact hA.unique hμ

/-- **An eigenvalue `μ` of `A` gives the eigenvalue `e^μ` of `exp A`.** -/
theorem hasEigenvalue_exp {A : E →L[ℝ] E} {μ : ℝ}
    (h : Module.End.HasEigenvalue (A : E →ₗ[ℝ] E) μ) :
    Module.End.HasEigenvalue ((exp A : E →L[ℝ] E) : E →ₗ[ℝ] E) (Real.exp μ) := by
  obtain ⟨v, hv⟩ := h.exists_hasEigenvector
  exact Module.End.hasEigenvalue_of_hasEigenvector (hasEigenvector_exp hv)

end Banach

section Hilbert

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- **The exponential of a symmetric map is symmetric**: a symmetric map is
self-adjoint, and so is its exponential (`IsSelfAdjoint.exp`). -/
theorem isSymmetric_exp {A : E →L[ℝ] E} (hA : (A : E →ₗ[ℝ] E).IsSymmetric) :
    ((exp A : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric :=
  ContinuousLinearMap.isSelfAdjoint_iff_isSymmetric.1 hA.isSelfAdjoint.exp

end Hilbert

/-- The hypotheses of `isSymmetric_exp`, `hasEigenvector_exp` and
`hasEigenvalue_exp` hold for twice the identity of the line, which is
symmetric with the eigenvector `1` for the eigenvalue `2`. -/
example : let A : EuclideanSpace ℝ (Fin 1) →L[ℝ] EuclideanSpace ℝ (Fin 1) :=
      (2 : ℝ) • ContinuousLinearMap.id ℝ _
    (A : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)).IsSymmetric ∧
      Module.End.HasEigenvector (A : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)) 2
        (EuclideanSpace.single 0 1) ∧
      Module.End.HasEigenvalue (A : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)) 2 := by
  intro A
  have hv : Module.End.HasEigenvector
      (A : EuclideanSpace ℝ (Fin 1) →ₗ[ℝ] EuclideanSpace ℝ (Fin 1)) 2
      (EuclideanSpace.single 0 1) := ⟨by simp [A], by simp⟩
  refine ⟨fun u v => ?_, hv, Module.End.hasEigenvalue_of_hasEigenvector hv⟩
  simp [A, real_inner_smul_left, real_inner_smul_right]

end Transformer.Landscape.SaddleAvoidance
