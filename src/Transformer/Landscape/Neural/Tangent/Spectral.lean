module

public import Transformer.Landscape.Neural.Tangent.Basic
public import Mathlib.Algebra.Order.Ring.Pow

/-!
# Exact spectral speed limits for the frozen prediction control

Source: `landscape_grokking_plan.md`, M9.4, and `access.py`'s
`linearized_path`. A genuine GD orbit of the quadratic from `Basic`
multiplies an eigenmode of the normalized kernel by `1 - eta * lambda`
at every update. A unit eigenvector supplies a lower bound on the whole
loss. Before `k * eta * lambda` exceeds one half, that component retains
at least one quarter of its initial squared error. The concrete weak-feature
example has a positive eigenvalue and an attainable zero-loss solution.

These are exact-real statements about the frozen quadratic. The measured
eigenvalue cutoff, numerical eigenvectors, changing neural features, the
original minibatches and AdamW are not certified by these theorems.
-/

@[expose] public section

namespace Transformer.Landscape.Neural.Tangent

open Transformer.Landscape.SaddleAvoidance

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [CompleteSpace E] [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]

/-- Exact residual component of actual parameter GD on the frozen loss;
the eigenvalue belongs to the derived `J J^T / n` kernel (M9.4 diagnostic). -/
theorem inner_residual_iterate (J : E →L[ℝ] F) (r : F) (n η : ℝ)
    {v : F} {μ : ℝ} (heigen : kernel J n v = μ • v) (x : E) (k : ℕ) :
    inner ℝ v (J ((gradientStep (loss J r n) η)^[k] x) + r) =
      (1 - η * μ) ^ k * inner ℝ v (J x + r) := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ_apply', residual_gradientStep, inner_sub_right,
      real_inner_smul_right, inner_kernel, heigen, real_inner_smul_left, ih, pow_succ]
    ring

omit [CompleteSpace E] [CompleteSpace F] in
/-- A unit output direction bounds the actual mean half-squared loss by
Cauchy--Schwarz, with the same positive example divisor (M9.4 diagnostic). -/
theorem projection_sq_le_loss (J : E →L[ℝ] F) (r : F) {n : ℝ} (hn : 0 < n)
    {v : F} (hv : ‖v‖ = 1) (x : E) :
    (n⁻¹ / 2) * (inner ℝ v (J x + r)) ^ 2 ≤ loss J r n x := by
  have hc : |inner ℝ v (J x + r)| ≤ ‖J x + r‖ := by
    simpa [hv] using norm_inner_le_norm (𝕜 := ℝ) v (J x + r)
  have hs : (inner ℝ v (J x + r)) ^ 2 ≤ ‖J x + r‖ ^ 2 :=
    sq_le_sq.mpr (by simpa only [abs_of_nonneg (norm_nonneg _)] using hc)
  exact mul_le_mul_of_nonneg_left hs (by positivity)

/-- An exact kernel-null component never changes under this frozen GD,
so its squared residual supplies a permanent loss floor (M9.4 diagnostic).
The floating-point eigenvalue cutoff does not establish this premise. -/
theorem loss_ge_nullmode (J : E →L[ℝ] F) (r : F) {n : ℝ} (hn : 0 < n)
    {v : F} (hv : ‖v‖ = 1) (heigen : kernel J n v = 0) (η : ℝ) (x : E) (k : ℕ) :
    (n⁻¹ / 2) * (inner ℝ v (J x + r)) ^ 2 ≤
      loss J r n ((gradientStep (loss J r n) η)^[k] x) := by
  have he : kernel J n v = (0 : ℝ) • v := by simpa using heigen
  have h := projection_sq_le_loss J r hn hv ((gradientStep (loss J r n) η)^[k] x)
  simpa only [inner_residual_iterate J r n η he x k, mul_zero, sub_zero, one_pow,
    one_mul] using h

/-- A mode too slow for the given finite budget leaves positive error.
The budget and upper bound on `eta * mu` are numerical premises; no convergence or
failure conclusion is assumed (M9.4 diagnostic). -/
theorem loss_ge_eigenmode_of_budget (J : E →L[ℝ] F) (r : F) {n η μ : ℝ}
    (hn : 0 < n) {v : F} (hv : ‖v‖ = 1) (heigen : kernel J n v = μ • v)
    (hstep : η * μ ≤ 1) (x : E) (k : ℕ) (hbudget : (k : ℝ) * (η * μ) ≤ 1 / 2) :
    (n⁻¹ / 8) * (inner ℝ v (J x + r)) ^ 2 ≤
      loss J r n ((gradientStep (loss J r n) η)^[k] x) := by
  have hb := one_add_mul_le_pow (a := -(η * μ)) (by linarith : -2 ≤ -(η * μ)) k
  have hf : (1 / 2 : ℝ) ≤ (1 - η * μ) ^ k := by
    rw [show 1 + -(η * μ) = 1 - η * μ by ring] at hb
    nlinarith
  have hfsq : (1 / 4 : ℝ) ≤ ((1 - η * μ) ^ k) ^ 2 := by nlinarith
  have hprod := mul_le_mul_of_nonneg_right hfsq (sq_nonneg (inner ℝ v (J x + r)))
  have hscaled := mul_le_mul_of_nonneg_left hprod (show 0 ≤ n⁻¹ / 2 by positivity)
  have hp := projection_sq_le_loss J r hn hv
    ((gradientStep (loss J r n) η)^[k] x)
  rw [inner_residual_iterate J r n η heigen x k] at hp
  calc
    (n⁻¹ / 8) * (inner ℝ v (J x + r)) ^ 2 =
        (n⁻¹ / 2) * ((1 / 4) * (inner ℝ v (J x + r)) ^ 2) := by ring
    _ ≤ (n⁻¹ / 2) * (((1 - η * μ) ^ k) ^ 2 * (inner ℝ v (J x + r)) ^ 2) := hscaled
    _ = (n⁻¹ / 2) * ((1 - η * μ) ^ k * inner ℝ v (J x + r)) ^ 2 := by ring
    _ ≤ _ := hp

/-- A nonzero but weak scalar prediction feature, used only as an explicit
inhabited control for the spectral statements (M9.4 diagnostic). -/
noncomputable def weakFeature : ℝ →L[ℝ] ℝ :=
  (1 / 10 : ℝ) • ContinuousLinearMap.id ℝ ℝ

/-- This actual feature has positive kernel eigenvalue `1/100`, rather
than an absent gradient direction (M9.4 weak-feature witness). -/
theorem weakFeature_eigenvalue :
    kernel weakFeature 1 (1 : ℝ) = (1 / 100 : ℝ) • (1 : ℝ) := by
  have ha : weakFeature.adjoint 1 = 1 / 10 := by
    have h := weakFeature.adjoint_inner_left (1 : ℝ) (1 : ℝ)
    simpa [weakFeature, Real.inner_apply] using h
  change (1 : ℝ)⁻¹ • weakFeature (weakFeature.adjoint 1) = (1 / 100 : ℝ) • (1 : ℝ)
  rw [ha]
  norm_num [weakFeature]

/- Each spectral hypothesis has a nonzero witness, including an exact fit
at a different finite parameter point. Fifty small-feature GD steps retain
loss at least 1/8 although the same model's optimum is zero. -/

example : kernel weakFeature 1 (1 : ℝ) = (1 / 100 : ℝ) • (1 : ℝ) ∧
    inner ℝ (1 : ℝ) (weakFeature ((gradientStep (loss weakFeature 1 1) 1)^[3] 0) + 1) =
      (1 - 1 / 100 : ℝ) ^ 3 := by
  refine ⟨weakFeature_eigenvalue, ?_⟩
  simpa [weakFeature, Real.inner_apply] using
    inner_residual_iterate weakFeature 1 1 1 weakFeature_eigenvalue 0 3

example : 0 < (1 : ℝ) ∧ ‖(1 : ℝ)‖ = 1 ∧
    (1 / 2 : ℝ) ≤ loss weakFeature 1 1 0 := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  simpa [weakFeature, Real.inner_apply] using
    projection_sq_le_loss weakFeature 1 (by norm_num : (0 : ℝ) < 1)
      (by norm_num : ‖(1 : ℝ)‖ = 1) 0

example : kernel (0 : ℝ →L[ℝ] ℝ) 1 (1 : ℝ) = 0 ∧
    (1 / 2 : ℝ) ≤ loss (0 : ℝ →L[ℝ] ℝ) 1 1
      ((gradientStep (loss (0 : ℝ →L[ℝ] ℝ) 1 1) 1)^[100] 0) := by
  have he : kernel (0 : ℝ →L[ℝ] ℝ) 1 (1 : ℝ) = 0 := by simp [kernel]
  refine ⟨he, ?_⟩
  simpa [Real.inner_apply] using
    loss_ge_nullmode (0 : ℝ →L[ℝ] ℝ) 1 (by norm_num : (0 : ℝ) < 1)
      (by norm_num : ‖(1 : ℝ)‖ = 1) he 1 0 100

/- This witness uses a positive step and positive eigenvalue in the stable
GD regime. Its zero-loss point is an explicit finite assignment, rather
than an assumed target fit. -/
example : 0 < (1 / 100 : ℝ) ∧ ‖(1 : ℝ)‖ = 1 ∧
    kernel weakFeature 1 (1 : ℝ) = (1 / 100 : ℝ) • (1 : ℝ) ∧
    (1 : ℝ) * (1 / 100) ≤ 1 ∧ (50 : ℝ) * (1 * (1 / 100)) ≤ 1 / 2 ∧
    (1 / 8 : ℝ) ≤ loss weakFeature 1 1 ((gradientStep (loss weakFeature 1 1) 1)^[50] 0) ∧
    loss weakFeature 1 1 (-10) = 0 := by
  refine ⟨by norm_num, by norm_num, weakFeature_eigenvalue, by norm_num, by norm_num, ?_, ?_⟩
  · simpa [weakFeature, Real.inner_apply] using
      loss_ge_eigenmode_of_budget weakFeature 1 (by norm_num : (0 : ℝ) < 1)
        (by norm_num : ‖(1 : ℝ)‖ = 1) weakFeature_eigenvalue
        (by norm_num : (1 : ℝ) * (1 / 100) ≤ 1) 0 50 (by norm_num)
  · norm_num [loss, weakFeature]

end Transformer.Landscape.Neural.Tangent
