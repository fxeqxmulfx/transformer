/-
# Saddle avoidance — the time-one map of a gradient flow

Let `f` be a `C²` function on a finite-dimensional real inner product space
whose gradient is `L`-Lipschitz and bounded by `M`.  Then `-∇f` is a flow
field (`isFlowField_neg_gradient`), and its flow from time `0`, the gradient
flow `ẋ = -∇f(x)` (`gradientFlow`), exists for all times.  For `t ≥ 0` the
time-`t` map has the two properties that
`measure_tendsto_symmetricUnstable_eq_zero` asks of a map:

* it is `C¹` (`contDiff_gradientFlow`), by `GlobalFlow.IsFlowField.contDiff_flow`;
* its derivative is invertible everywhere (`det_fderiv_gradientFlow_ne_zero`).
  The flow back from time `t` to time `0` is the flow of the reversed field
  from time `-t` to time `0` (`GlobalFlow.IsFlowField.flow_reverse`), hence
  `C¹`, and it is a left inverse (`det_fderiv_flow_ne_zero`, for any flow
  field with a jointly continuous derivative).

A critical point `z` is fixed by the flow (`gradientFlow_eq_self`), so the
variational equation along its trajectory is the linear equation
`J' = -∇²f(z) J`, solved by `exp (-τ ∇²f(z))`, the derivative of the time-`t`
map at `z` (`hasFDerivAt_gradientFlow_of_gradient_eq_zero`).  A negative
eigenvalue `λ` of the Hessian gives the eigenvalue `e^{-λ} > 1` of the
derivative of the time-one map, so a strict saddle is a symmetric unstable
fixed point of the time-one map
(`IsStrictSaddle.isSymmetricUnstable_gradientFlow`).

These are the two facts that the proof of `l:nosaddleconv` in Appendix A of
arXiv:2312.10794v5 takes from Shub's book: `Φ^t` is a diffeomorphism, and the
Jacobian of a gradient field at a critical point is the Hessian.
-/

import Transformer.SaddleAvoidance.LinearFlow
import Transformer.SaddleAvoidance.GradientDescent
import Transformer.GlobalFlow.Variational

open NormedSpace
open scoped NNReal

namespace Transformer.SaddleAvoidance

open GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {M : ℝ}

/-- The gradient flow `ẋ = -∇f(x)` of `f`: `gradientFlow f t x` is the
position at time `t` of the trajectory through `x` at time `0`. -/
noncomputable def gradientFlow (f : E → ℝ) (t : ℝ) (x : E) : E :=
  flow (fun _ y => -gradient f y) 0 t x

/-- **`-∇f` is a flow field** when `∇f` is `L`-Lipschitz and bounded by `M`. -/
theorem isFlowField_neg_gradient (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) : IsFlowField (fun _ x => -gradient f x) L M where
  continuous := (hL.continuous.comp continuous_snd).neg
  lipschitz _ := hL.neg
  bound _ x := by rw [norm_neg]; exact hM x

/-- The trajectories of `gradientFlow f` solve `ẋ = -∇f(x)`. -/
theorem hasDerivAt_gradientFlow (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) (t : ℝ) (x : E) :
    HasDerivAt (fun t => gradientFlow f t x) (-gradient f (gradientFlow f t x)) t :=
  (isFlowField_neg_gradient hL hM).hasDerivAt_flow 0 t x

/-- **A critical point is fixed by the gradient flow**: the constant
trajectory through it solves `ẋ = -∇f(x)`. -/
theorem gradientFlow_eq_self (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) {z : E} (hz : gradient f z = 0) (t : ℝ) :
    gradientFlow f t z = z :=
  (isFlowField_neg_gradient hL hM).flow_eq (γ := fun _ => z)
    (fun τ => by simpa [hz] using hasDerivAt_const τ z) 0 t

variable [FiniteDimensional ℝ E]

/-- **A flow map has an invertible derivative.**  Let `w` be a flow field
with a jointly continuous spatial derivative `A`, and `s ≤ t`.  The flow
from time `t` back to time `s` is the flow of the reversed field `-w(-τ, x)`
from time `-t` to time `-s` (`flow_reverse`), so it is `C¹`, and it is a left
inverse of `flow w s t` (`flow_flow`).  The derivative of `flow w s t`
therefore has a left inverse, and its determinant is nonzero. -/
theorem det_fderiv_flow_ne_zero {w : ℝ → E → E} {A : ℝ → E → E →L[ℝ] E}
    (hw : IsFlowField w L M) (hA : Continuous (Function.uncurry A))
    (hwA : ∀ t x, HasFDerivAt (w t) (A t x) x) {s t : ℝ} (hst : s ≤ t) (x : E) :
    (fderiv ℝ (flow w s t) x).det ≠ 0 := by
  have hback : ContDiff ℝ 1 (flow w t s) := by
    have h := hw.reverse.contDiff_flow (A := fun τ y => -A (-τ) y)
      (hA.comp (continuous_fst.neg.prodMk continuous_snd)).neg (fun τ y => (hwA (-τ) y).neg)
      (neg_le_neg hst)
    rwa [show flow (fun τ y => -w (-τ) y) (-t) (-s) = flow w t s from
      funext (hw.flow_reverse t s)] at h
  have hinv : flow w t s ∘ flow w s t = id :=
    funext fun y => (hw.flow_flow s t s y).trans (hw.flow_self s y)
  have hcomp := (hback.differentiable one_ne_zero (flow w s t x)).hasFDerivAt.comp x
    ((hw.contDiff_flow hA hwA hst).differentiable one_ne_zero x).hasFDerivAt
  rw [hinv] at hcomp
  have hdet := congrArg ContinuousLinearMap.det (hcomp.unique (hasFDerivAt_id x))
  simp only [ContinuousLinearMap.det, ContinuousLinearMap.toLinearMap_comp, LinearMap.det_comp,
    ContinuousLinearMap.coe_id, LinearMap.det_id] at hdet
  exact right_ne_zero_of_mul_eq_one hdet

/-- The spatial derivative of the field `-∇f` is `-∇²f`. -/
theorem hasFDerivAt_neg_gradient (hf : ContDiff ℝ 2 f) (x : E) :
    HasFDerivAt (fun y => -gradient f y) (-fderiv ℝ (gradient f) x) x :=
  ((contDiff_gradient hf).differentiable one_ne_zero x).hasFDerivAt.neg

/-- The derivative `-∇²f` of the field `-∇f` is jointly continuous in time
and space, for `f` of class `C²`. -/
theorem continuous_neg_hessian (hf : ContDiff ℝ 2 f) :
    Continuous (Function.uncurry fun (_ : ℝ) (x : E) => -fderiv ℝ (gradient f) x) :=
  ((contDiff_gradient hf).continuous_fderiv one_ne_zero).neg.comp continuous_snd

/-- **The time-`t` map of the gradient flow is `C¹`**, for `t ≥ 0`. -/
theorem contDiff_gradientFlow (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) {t : ℝ} (ht : 0 ≤ t) : ContDiff ℝ 1 (gradientFlow f t) :=
  (isFlowField_neg_gradient hL hM).contDiff_flow (continuous_neg_hessian hf)
    (fun _ x => hasFDerivAt_neg_gradient hf x) ht

/-- **The time-`t` map of the gradient flow has an invertible derivative**,
for `t ≥ 0`. -/
theorem det_fderiv_gradientFlow_ne_zero (hf : ContDiff ℝ 2 f)
    (hL : LipschitzWith L (gradient f)) (hM : ∀ x, ‖gradient f x‖ ≤ M) {t : ℝ} (ht : 0 ≤ t)
    (x : E) : (fderiv ℝ (gradientFlow f t) x).det ≠ 0 :=
  det_fderiv_flow_ne_zero (isFlowField_neg_gradient hL hM) (continuous_neg_hessian hf)
    (fun _ y => hasFDerivAt_neg_gradient hf y) ht x

/-- **At a critical point the derivative of the time-`t` map is
`exp (-t ∇²f(z))`**, for `t ≥ 0`.  The trajectory through `z` is constant
(`gradientFlow_eq_self`), so the variational equation along it is
`J' = -∇²f(z) J`, solved by `J τ = exp (-τ ∇²f(z))` (`hasDerivAt_exp_smul`). -/
theorem hasFDerivAt_gradientFlow_of_gradient_eq_zero (hf : ContDiff ℝ 2 f)
    (hL : LipschitzWith L (gradient f)) (hM : ∀ x, ‖gradient f x‖ ≤ M) {z : E}
    (hz : gradient f z = 0) {t : ℝ} (ht : 0 ≤ t) :
    HasFDerivAt (gradientFlow f t) (exp (t • -fderiv ℝ (gradient f) z)) z := by
  refine (isFlowField_neg_gradient hL hM).hasFDerivAt_flow (continuous_neg_hessian hf)
    (fun _ x => hasFDerivAt_neg_gradient hf x) ht z
    (J := fun τ => exp (τ • -fderiv ℝ (gradient f) z)) (exp_zero_smul _) (continuous_exp_smul _)
    fun τ _ => ?_
  rw [show flow (fun _ y => -gradient f y) 0 τ z = z from gradientFlow_eq_self hL hM hz τ]
  exact hasDerivAt_exp_smul _ τ

/-- **A strict saddle is a symmetric unstable fixed point of the time-one
map.**  The derivative of the time-one map at a strict saddle `z` is
`exp (-∇²f(z))`, symmetric since the Hessian is (`isSymmetric_exp`), and a
negative eigenvalue `λ` of the Hessian gives its eigenvalue `e^{-λ} > 1`
(`hasEigenvalue_exp`). -/
theorem IsStrictSaddle.isSymmetricUnstable_gradientFlow (hf : ContDiff ℝ 2 f)
    (hL : LipschitzWith L (gradient f)) (hM : ∀ x, ‖gradient f x‖ ≤ M) {z : E}
    (hz : IsStrictSaddle f z) : IsSymmetricUnstable (gradientFlow f 1) z := by
  obtain ⟨hcrit, lam, hlam, hev⟩ := hz
  have hd : fderiv ℝ (gradientFlow f 1) z = exp (-fderiv ℝ (gradient f) z) := by
    have h := (hasFDerivAt_gradientFlow_of_gradient_eq_zero hf hL hM hcrit zero_le_one).fderiv
    rwa [one_smul] at h
  have hsymm : ((-fderiv ℝ (gradient f) z : E →L[ℝ] E) : E →ₗ[ℝ] E).IsSymmetric := by
    intro u v
    have hs := isSymmetric_fderiv_gradient hf z u v
    simp only [ContinuousLinearMap.coe_coe] at hs ⊢
    simp only [neg_apply, inner_neg_left, inner_neg_right, hs]
  have hneg : Module.End.HasEigenvalue
      ((-fderiv ℝ (gradient f) z : E →L[ℝ] E) : E →ₗ[ℝ] E) (-lam) := by
    obtain ⟨w, hw⟩ := hev.exists_hasEigenvector
    refine Module.End.hasEigenvalue_of_hasEigenvector (x := w) ⟨?_, hw.2⟩
    have hHw : fderiv ℝ (gradient f) z w = lam • w := hw.apply_eq_smul
    simp [hHw]
  refine ⟨gradientFlow_eq_self hL hM hcrit 1, ?_, Real.exp (-lam), ?_, ?_⟩
  · rw [hd]
    exact isSymmetric_exp hsymm
  · rw [abs_of_pos (Real.exp_pos _), Real.one_lt_exp_iff]
    linarith
  · rw [hd]
    exact hasEigenvalue_exp hneg

/-- The hypotheses of the theorems above hold for `f = cos` on the line, with
`L = M = 1`, the times `s = 0 ≤ t = 1`, the field `w = -∇f` and its derivative
`A = -∇²f`, at the strict saddle `z = 0`, a maximum: `∇f = -sin` is
`1`-Lipschitz and bounded by `1`, and the Hessian at `0` is `-cos 0 = -1`. -/
example : ContDiff ℝ 2 Real.cos ∧ LipschitzWith 1 (gradient Real.cos) ∧
    (∀ x, ‖gradient Real.cos x‖ ≤ 1) ∧ (0 : ℝ) ≤ 1 ∧ gradient Real.cos 0 = 0 ∧
    IsStrictSaddle Real.cos 0 ∧ IsFlowField (fun _ x => -gradient Real.cos x) 1 1 ∧
    Continuous (Function.uncurry fun (_ : ℝ) (x : ℝ) => -fderiv ℝ (gradient Real.cos) x) ∧
    ∀ t x : ℝ, HasFDerivAt ((fun _ y => -gradient Real.cos y) t)
      (-fderiv ℝ (gradient Real.cos) x) x := by
  have hgrad : gradient Real.cos = fun x => -Real.sin x :=
    funext fun x => (Real.hasDerivAt_cos x).hasGradientAt'.gradient
  have hL : LipschitzWith 1 (gradient Real.cos) := by
    rw [hgrad]
    exact Real.lipschitzWith_sin.neg
  have hM : ∀ x, ‖gradient Real.cos x‖ ≤ 1 := fun x => by
    rw [hgrad, norm_neg, Real.norm_eq_abs]
    exact Real.abs_sin_le_one x
  have h0 : gradient Real.cos 0 = 0 := by simp [hgrad]
  have hH : fderiv ℝ (gradient Real.cos) 0 =
      ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (-Real.cos 0) := by
    rw [hgrad]
    exact (Real.hasDerivAt_sin 0).neg.hasFDerivAt.fderiv
  refine ⟨Real.contDiff_cos, hL, hM, zero_le_one, h0, ⟨h0, -1, by norm_num, ?_⟩,
    isFlowField_neg_gradient hL hM, continuous_neg_hessian Real.contDiff_cos,
    fun _ x => hasFDerivAt_neg_gradient Real.contDiff_cos x⟩
  refine Module.End.hasEigenvalue_of_hasEigenvector (x := 1) ⟨?_, one_ne_zero⟩
  simp [hH]

end Transformer.SaddleAvoidance
