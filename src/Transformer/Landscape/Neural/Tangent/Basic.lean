module

public import Transformer.Landscape.SaddleAvoidance.Defs
public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# The actual quadratic objective of the frozen-Jacobian control

Source: `landscape_grokking_plan.md`, M9.4 and Mechanisms to test;
`experiments/mod97_representation_controls/access.py`, `linearized_path`.
`J` is a fixed prediction derivative and `r` the initial prediction residual.
Every parameter displacement remains an argument. The objective is half the
squared Euclidean residual divided by the example count `n`. The proofs
derive its parameter gradient and its induced residual update, including
the divisor used in the measured kernel. No network Jacobian is assumed
constant along actual neural training, and no AdamW equivalence is claimed.
-/

@[expose] public section

namespace Transformer.Landscape.Neural.Tangent

open Transformer.Landscape.SaddleAvoidance

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [CompleteSpace E] [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]

/-- The frozen model's empirical half-squared loss. For an actual mean loss,
`n` is the positive number of measured training equations (M9.4 diagnostic). -/
noncomputable def loss (J : E →L[ℝ] F) (r : F) (n : ℝ) (x : E) : ℝ :=
  (n⁻¹ / 2) * ‖J x + r‖ ^ 2

/-- The prediction kernel includes the same example divisor as the loss.
This is `J J^T / n` of the frozen full-batch diagnostic. -/
noncomputable def kernel (J : E →L[ℝ] F) (n : ℝ) : F →L[ℝ] F :=
  n⁻¹ • J.comp J.adjoint

/-- Differentiating the actual quadratic gives the adjoint residual, rather
than stipulating a kernel recurrence as the optimizer (M9.4 diagnostic). -/
theorem hasGradientAt_loss (J : E →L[ℝ] F) (r : F) (n : ℝ) (x : E) :
    HasGradientAt (loss J r n) (n⁻¹ • J.adjoint (J x + r)) x := by
  apply hasGradientAt_iff_hasFDerivAt.mpr
  have hd : HasFDerivAt (fun y => J y + r) J x := J.hasFDerivAt.add_const r
  have h := hd.norm_sq.const_mul (n⁻¹ / 2)
  convert h using 1
  · rfl
  · ext v
    simp only [InnerProductSpace.toDual_apply_apply, real_inner_smul_left,
      smul_apply, ContinuousLinearMap.comp_apply,
      innerSL_apply_apply, smul_eq_mul, J.adjoint_inner_left]
    ring

/-- The gradient used by the existing Landscape GD operator is the derived
one, with every displacement coordinate free (M9.4 diagnostic). -/
theorem gradient_loss (J : E →L[ℝ] F) (r : F) (n : ℝ) (x : E) :
    gradient (loss J r n) x = n⁻¹ • J.adjoint (J x + r) := by
  exact (hasGradientAt_loss J r n x).gradient

/-- A supplied derivative certificate identifies this kernel with the
derivative of an actual prediction map at its stated base weights. The
network's value at those weights supplies the initial residual (M9.4). -/
theorem gradient_loss_of_hasFDerivAt {prediction : E → F} {J : E →L[ℝ] F}
    {base : E} (h : HasFDerivAt prediction J base) (target : F) (n : ℝ) (x : E) :
    gradient (loss (fderiv ℝ prediction base) (prediction base - target) n) x =
      n⁻¹ • J.adjoint (J x + (prediction base - target)) := by
  rw [h.fderiv]
  exact gradient_loss J (prediction base - target) n x

/-- Applying a genuine parameter GD step produces the measured residual
recurrence, retaining the initial prediction offset (M9.4 diagnostic). -/
theorem residual_gradientStep (J : E →L[ℝ] F) (r : F) (n η : ℝ) (x : E) :
    J (gradientStep (loss J r n) η x) + r =
      (J x + r) - η • kernel J n (J x + r) := by
  simp only [gradientStep, gradient_loss, map_sub, map_smul, kernel,
    smul_apply, ContinuousLinearMap.comp_apply]
  module

/-- The kernel is symmetric by the adjoint identities, not by a supplied
spectral assumption (M9.4 diagnostic, symmetric empirical Gram matrix). -/
theorem inner_kernel (J : E →L[ℝ] F) (n : ℝ) (u v : F) :
    inner ℝ u (kernel J n v) = inner ℝ (kernel J n u) v := by
  simp only [kernel, smul_apply, ContinuousLinearMap.comp_apply,
    real_inner_smul_left, real_inner_smul_right]
  rw [← J.adjoint_inner_left, ← J.adjoint_inner_right]

/-- A positive example count gives a nonnegative kernel quadratic form,
including singular Jacobians (M9.4 diagnostic, retained and zero modes). -/
theorem inner_kernel_nonneg (J : E →L[ℝ] F) {n : ℝ} (hn : 0 < n) (v : F) :
    0 ≤ inner ℝ v (kernel J n v) := by
  simp only [kernel, smul_apply, ContinuousLinearMap.comp_apply,
    real_inner_smul_right]
  rw [← J.adjoint_inner_left]
  exact mul_nonneg (inv_nonneg.mpr hn.le) (real_inner_self_nonneg)

/-- A unit eigenmode of this actual positive kernel has nonnegative
eigenvalue; negative curvature is not supplied to the frozen control
(M9.4 diagnostic, empirical Gram spectrum). -/
theorem eigenvalue_nonneg (J : E →L[ℝ] F) {n μ : ℝ} (hn : 0 < n)
    {v : F} (hv : ‖v‖ = 1) (heigen : kernel J n v = μ • v) : 0 ≤ μ := by
  have h := inner_kernel_nonneg J hn v
  rw [heigen, real_inner_smul_right, real_inner_self_eq_norm_sq, hv] at h
  simpa using h

omit [CompleteSpace E] in
/-- In the actual finite prediction space the norm objective is precisely
the mean of the row half-squared errors, without a second output divisor
(M9.3 scalar loss and M9.4 frozen subset). -/
theorem loss_eq_mean_squared {I : Type*} [Fintype I]
    (J : E →L[ℝ] EuclideanSpace ℝ I) (r : EuclideanSpace ℝ I) (x : E) :
    loss J r (Fintype.card I) x =
      (∑ i, ((J x + r).ofLp i) ^ 2) / (2 * Fintype.card I) := by
  rw [loss, EuclideanSpace.norm_sq_eq]
  simp only [Real.norm_eq_abs, sq_abs]
  ring

/- A genuine nonconstant one-parameter predictor with nonzero residual;
the example count is one, rather than an empty-data convention. -/
example : HasGradientAt (loss (ContinuousLinearMap.id ℝ ℝ) 1 1) 1 0 := by
  simpa [ContinuousLinearMap.adjoint_id] using
    hasGradientAt_loss (ContinuousLinearMap.id ℝ ℝ) 1 1 0

example : 0 < (1 : ℝ) ∧
    0 ≤ inner ℝ (1 : ℝ) (kernel (ContinuousLinearMap.id ℝ ℝ) 1 1) := by
  exact ⟨by norm_num, inner_kernel_nonneg _ (by norm_num) _⟩

example : ‖(1 : ℝ)‖ = 1 ∧
    kernel (ContinuousLinearMap.id ℝ ℝ) 1 (1 : ℝ) = (1 : ℝ) • (1 : ℝ) ∧
    0 ≤ (1 : ℝ) := by
  have he : kernel (ContinuousLinearMap.id ℝ ℝ) 1 (1 : ℝ) = (1 : ℝ) • (1 : ℝ) := by
    simp [kernel, ContinuousLinearMap.adjoint_id]
  exact ⟨by norm_num, he, eigenvalue_nonneg _ (by norm_num) (by norm_num) he⟩

example : HasFDerivAt (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ) 0 ∧
    gradient (loss (fderiv ℝ (fun x : ℝ => x) 0) (0 - 1) 1) 0 = -1 := by
  have h : HasFDerivAt (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ) 0 :=
    (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt
  refine ⟨h, ?_⟩
  simpa [ContinuousLinearMap.adjoint_id] using gradient_loss_of_hasFDerivAt h 1 1 0

/- Two nonconstant row residuals verify the actual finite mean convention:
sum squared error two, divided by twice the two observed equations. -/
example : loss (ContinuousLinearMap.id ℝ (EuclideanSpace ℝ (Fin 2)))
    (WithLp.toLp 2 (fun i : Fin 2 => if i = 0 then (1 : ℝ) else -1)) 2 0 = 1 / 2 := by
  norm_num [loss, EuclideanSpace.norm_sq_eq, Fin.sum_univ_two]

end Transformer.Landscape.Neural.Tangent
