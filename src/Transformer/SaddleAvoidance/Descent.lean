/-
# Saddle avoidance — gradient descent decreases the function

Let `f` be differentiable on a real inner product space with an `L`-Lipschitz
gradient.  By the mean value inequality for `f - ⟪∇f(x), ·⟫`, whose
derivative `∇f(w) - ∇f(x)` has norm at most `L ‖y - x‖` on the segment
`[x, y]`, `f y ≤ f x + ⟪∇f(x), y - x⟫ + L ‖y - x‖²`
(`le_add_inner_add_mul_norm_sq`).  The gradient step therefore decreases `f`
by at least `η (1 - η L) ‖∇f(x)‖²` (`le_sub_of_gradientStep`), and `f`
decreases along every orbit when `η L ≤ 1` (`antitone_iterate`).

When `f` is invariant under translation by `v`, so is its gradient
(`gradient_add_eq`), and the orbits of gradient descent commute with the
translation (`iterate_gradientStep_add`).  A limit of an orbit is a critical
point (`gradient_eq_zero_of_tendsto`).  The running example of this file and
the next ones is `‖x‖² / 2`: its gradient is `x`
(`gradient_half_mul_norm_sq`), so gradient descent multiplies by `1 - η`
(`iterate_gradientStep_half_mul_norm_sq`).
-/

import Transformer.SaddleAvoidance.Defs
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.InnerProductSpace.Dual
import Mathlib.Analysis.SpecificLimits.Basic

open Filter Metric InnerProductSpace
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **Descent lemma.**  For `f` with an `L`-Lipschitz gradient,
`f y ≤ f x + ⟪∇f(x), y - x⟫ + L ‖y - x‖²`.  The usual constant `L/2` needs an
integral; the mean value inequality gives `L`. -/
theorem le_add_inner_add_mul_norm_sq {f : E → ℝ} (hf : Differentiable ℝ f) {L : ℝ≥0}
    (hL : LipschitzWith L (gradient f)) (x y : E) :
    f y ≤ f x + ⟪gradient f x, y - x⟫ + L * ‖y - x‖ ^ 2 := by
  set h : E → ℝ := fun w => f w - toDual ℝ E (gradient f x) w with hh
  have hderiv : ∀ w, HasFDerivAt h (toDual ℝ E (gradient f w - gradient f x)) w := fun w => by
    rw [map_sub]
    exact (hasGradientAt_iff_hasFDerivAt.1 (hf w).hasGradientAt).sub
      (toDual ℝ E (gradient f x)).hasFDerivAt
  have hbound : ∀ w ∈ segment ℝ x y,
      ‖toDual ℝ E (gradient f w - gradient f x)‖ ≤ L * ‖y - x‖ := fun w hw => by
    rw [LinearIsometryEquiv.norm_map]
    exact (hL.norm_sub_le w x).trans
      (mul_le_mul_of_nonneg_left (norm_sub_le_of_mem_segment hw) L.coe_nonneg)
  have hmv := Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun w _ => (hderiv w).hasFDerivWithinAt) hbound (convex_segment x y)
    (left_mem_segment ℝ x y) (right_mem_segment ℝ x y)
  have hle := (le_abs_self _).trans ((Real.norm_eq_abs _).symm.le.trans hmv)
  simp only [hh, toDual_apply_apply, inner_sub_right] at hle ⊢
  nlinarith

/-- **Sufficient decrease.**  One gradient step lowers `f` by at least
`η (1 - η L) ‖∇f(x)‖²`. -/
theorem le_sub_of_gradientStep {f : E → ℝ} (hf : Differentiable ℝ f) {L : ℝ≥0}
    (hL : LipschitzWith L (gradient f)) (η : ℝ) (x : E) :
    f (gradientStep f η x) ≤ f x - η * (1 - η * L) * ‖gradient f x‖ ^ 2 := by
  have h := le_add_inner_add_mul_norm_sq hf hL x (gradientStep f η x)
  have hstep : gradientStep f η x - x = -(η • gradient f x) := by
    simp [gradientStep]
  rw [hstep, inner_neg_right, real_inner_smul_right, real_inner_self_eq_norm_sq, norm_neg,
    norm_smul, Real.norm_eq_abs, mul_pow, sq_abs] at h
  calc f (gradientStep f η x)
      ≤ f x + -(η * ‖gradient f x‖ ^ 2) + L * (η ^ 2 * ‖gradient f x‖ ^ 2) := h
    _ = f x - η * (1 - η * L) * ‖gradient f x‖ ^ 2 := by ring

/-- `f` decreases along every orbit of gradient descent when `0 ≤ η` and
`η L ≤ 1`. -/
theorem antitone_iterate {f : E → ℝ} (hf : Differentiable ℝ f) {L : ℝ≥0}
    (hL : LipschitzWith L (gradient f)) {η : ℝ} (hη : 0 ≤ η) (hηL : η * L ≤ 1) (x : E) :
    Antitone fun k => f ((gradientStep f η)^[k] x) := by
  refine antitone_nat_of_succ_le fun k => ?_
  rw [Function.iterate_succ_apply']
  have h := le_sub_of_gradientStep hf hL η ((gradientStep f η)^[k] x)
  have hnn := mul_nonneg (mul_nonneg hη (sub_nonneg.2 hηL))
    (sq_nonneg ‖gradient f ((gradientStep f η)^[k] x)‖)
  linarith

/-- A function invariant under translation by `v` has a gradient invariant
under it. -/
theorem gradient_add_eq {f : E → ℝ} {v : E} (hv : ∀ y, f (y + v) = f y) (y : E) :
    gradient f (y + v) = gradient f y := by
  have h : fderiv ℝ f (y + v) = fderiv ℝ f y := by
    rw [← fderiv_comp_add_right]
    exact congrArg (fun F => fderiv ℝ F y) (funext hv)
  simp only [gradient, h]

/-- The orbits of gradient descent commute with the translations that leave
`f` invariant. -/
theorem iterate_gradientStep_add {f : E → ℝ} {v : E} (hv : ∀ y, f (y + v) = f y) (η : ℝ)
    (k : ℕ) (y : E) : (gradientStep f η)^[k] (y + v) = (gradientStep f η)^[k] y + v := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ih]
    simp only [gradientStep, gradient_add_eq hv]
    abel

/-- **A limit of gradient descent is a critical point**, for a continuous
gradient and a nonzero step: it is a fixed point of the continuous step map. -/
theorem gradient_eq_zero_of_tendsto {f : E → ℝ} (hc : Continuous (gradient f)) {η : ℝ}
    (hη : η ≠ 0) {x w : E} (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    gradient f w = 0 := by
  have hg : Continuous (gradientStep f η) := continuous_id.sub (hc.const_smul η)
  have h1 : Tendsto (fun k => (gradientStep f η)^[k + 1] x) atTop (𝓝 w) :=
    hx.comp (tendsto_add_atTop_nat 1)
  have h2 : Tendsto (fun k => (gradientStep f η)^[k + 1] x) atTop (𝓝 (gradientStep f η w)) := by
    simp only [Function.iterate_succ_apply']
    exact (hg.tendsto w).comp hx
  have hfix := tendsto_nhds_unique h2 h1
  simp only [gradientStep, sub_eq_self, smul_eq_zero] at hfix
  exact hfix.resolve_left hη

/-- The gradient of `‖x‖² / 2` is the identity. -/
theorem gradient_half_mul_norm_sq : gradient (fun x : E => (1 / 2 : ℝ) * ‖x‖ ^ 2) = id := by
  funext y
  apply HasGradientAt.gradient
  rw [hasGradientAt_iff_hasFDerivAt]
  convert (hasStrictFDerivAt_norm_sq y).hasFDerivAt.const_mul (1 / 2 : ℝ) using 1
  ext z
  simp only [id, toDual_apply_apply, smul_apply, innerSL_apply_apply, smul_eq_mul]
  ring

/-- A gradient step on `‖x‖² / 2` multiplies by `1 - η`. -/
theorem gradientStep_half_mul_norm_sq (η : ℝ) :
    gradientStep (fun x : E => (1 / 2 : ℝ) * ‖x‖ ^ 2) η = fun x => (1 - η) • x := by
  funext x
  simp only [gradientStep, gradient_half_mul_norm_sq, id, sub_smul, one_smul]

/-- `k` gradient steps on `‖x‖² / 2` multiply by `(1 - η) ^ k`. -/
theorem iterate_gradientStep_half_mul_norm_sq (η : ℝ) (k : ℕ) (x : E) :
    (gradientStep (fun x : E => (1 / 2 : ℝ) * ‖x‖ ^ 2) η)^[k] x = (1 - η) ^ k • x := by
  rw [gradientStep_half_mul_norm_sq, smul_iterate]

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line, whose gradient `x` is `1`-Lipschitz, with `η = 1/2`, the translation by
`0`, and the orbit `2⁻ᵏ` of `1`, which converges to `0`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    Differentiable ℝ f ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) ≤ 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) ≤ 1 ∧ (∀ y, f (y + 0) = f y) ∧ Continuous (gradient f) ∧
      (1 / 2 : ℝ) ≠ 0 ∧ Tendsto (fun k => (gradientStep f (1 / 2))^[k] 1) atTop (𝓝 0) := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, ?_, by norm_num,
    by norm_num, fun y => by rw [add_zero], ?_, by norm_num, ?_⟩
  · rw [hg]
    exact LipschitzWith.id
  · rw [hg]
    exact continuous_id
  · simp only [f, iterate_gradientStep_half_mul_norm_sq, smul_eq_mul, mul_one]
    exact tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)

end Transformer.SaddleAvoidance
