module

public import Transformer.Landscape.SaddleAvoidance.TimeOne
public import Transformer.Landscape.SaddleAvoidance.Symmetrizable
public import Transformer.Landscape.SaddleAvoidance.SquareRoot

/-!
# Saddle avoidance — the time-one map of `ẋ = -G(x) ∇f(x)`

Let `f` be a `C²` function and `G` a `C¹` field of linear maps on a
finite-dimensional real inner product space such that `G ∇f` is
`L`-Lipschitz and bounded by `M`. Then `-G ∇f` is a flow field
(`isFlowField_neg_metricGradient`) whose flow from time `0`
(`metricGradientFlow`) exists for all times: where `G(x)` is symmetric
positive definite, the gradient flow of `f` in the metric `⟪G(x)⁻¹ u, v⟫`.
As for `gradientFlow` (`Landscape.SaddleAvoidance.TimeOne`), for `t ≥ 0` the time-`t`
map is `C¹` (`contDiff_metricGradientFlow`) with an invertible derivative
(`det_fderiv_metricGradientFlow_ne_zero`). A critical point `z` is fixed
(`metricGradientFlow_eq_self`); there the derivative of `G ∇f` is
`G(z) ∇²f(z)`, the term `DG(z)[·] ∇f(z)` vanishing
(`fderiv_metricGradient_of_gradient_eq_zero`), and the time-`t` map has the
derivative `exp (-t G(z) ∇²f(z))`
(`hasFDerivAt_metricGradientFlow_of_gradient_eq_zero`).

`G(z) ∇²f(z)` need not be symmetric. When `G(z)` is symmetric positive
definite it is conjugate to a symmetric map, and a negative eigenvalue of the
Hessian gives it a negative eigenvalue `κ`
(`exists_conj_isSymmetric_and_hasEigenvalue_neg`). So a strict saddle is a
symmetrizable unstable fixed point of the time-one map, with the eigenvalue
`e^{-κ} > 1` (`IsStrictSaddle.isSymmetrizableUnstable_metricGradientFlow`).
-/

@[expose] public section

open NormedSpace
open scoped NNReal

namespace Transformer.Landscape.SaddleAvoidance

open Landscape.GlobalFlow

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {G : E → E →L[ℝ] E} {f : E → ℝ} {L : ℝ≥0} {M : ℝ}

/-- The flow `ẋ = -G(x) ∇f(x)`: `metricGradientFlow G f t x` is the position
at time `t` of the trajectory through `x` at time `0`. Where `G(x)` is
symmetric positive definite, `G(x) ∇f(x)` is the gradient of `f` in the
metric `⟪G(x)⁻¹ u, v⟫`. -/
noncomputable def metricGradientFlow (G : E → E →L[ℝ] E) (f : E → ℝ) (t : ℝ) (x : E) : E :=
  flow (fun _ y => -G y (gradient f y)) 0 t x

/-- **`-G ∇f` is a flow field** when `G ∇f` is `L`-Lipschitz and bounded. -/
theorem isFlowField_neg_metricGradient (hL : LipschitzWith L fun x => G x (gradient f x))
    (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M) :
    IsFlowField (fun _ x => -G x (gradient f x)) L M where
  continuous := (hL.continuous.comp continuous_snd).neg
  lipschitz _ := hL.neg
  bound _ x := by rw [norm_neg]; exact hM x

/-- **A critical point is fixed by the flow**: the constant trajectory
through it solves `ẋ = -G(x) ∇f(x)`. -/
theorem metricGradientFlow_eq_self (hL : LipschitzWith L fun x => G x (gradient f x))
    (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M) {z : E} (hz : gradient f z = 0) (t : ℝ) :
    metricGradientFlow G f t z = z :=
  (isFlowField_neg_metricGradient hL hM).flow_eq (γ := fun _ => z)
    (fun τ => by simpa [hz] using hasDerivAt_const τ z) 0 t

variable [FiniteDimensional ℝ E]

/-- The field `G ∇f` is `C¹` for `f` of class `C²` and `G` of class `C¹`. -/
theorem contDiff_metricGradient (hf : ContDiff ℝ 2 f) (hG : ContDiff ℝ 1 G) :
    ContDiff ℝ 1 fun x => G x (gradient f x) :=
  hG.clm_apply (contDiff_gradient hf)

/-- The derivative of the field `-G ∇f` is jointly continuous. -/
theorem continuous_neg_fderiv_metricGradient (hf : ContDiff ℝ 2 f) (hG : ContDiff ℝ 1 G) :
    Continuous (Function.uncurry fun (_ : ℝ) (x : E) =>
      -fderiv ℝ (fun y => G y (gradient f y)) x) :=
  ((contDiff_metricGradient hf hG).continuous_fderiv one_ne_zero).neg.comp continuous_snd

/-- The spatial derivative of the field `-G ∇f`. -/
theorem hasFDerivAt_neg_metricGradient (hf : ContDiff ℝ 2 f) (hG : ContDiff ℝ 1 G) (x : E) :
    HasFDerivAt (fun y => -G y (gradient f y)) (-fderiv ℝ (fun y => G y (gradient f y)) x) x :=
  ((contDiff_metricGradient hf hG).differentiable one_ne_zero x).hasFDerivAt.neg

/-- **The time-`t` map is `C¹`**, for `t ≥ 0`. -/
theorem contDiff_metricGradientFlow (hf : ContDiff ℝ 2 f) (hG : ContDiff ℝ 1 G)
    (hL : LipschitzWith L fun x => G x (gradient f x)) (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M)
    {t : ℝ} (ht : 0 ≤ t) : ContDiff ℝ 1 (metricGradientFlow G f t) :=
  (isFlowField_neg_metricGradient hL hM).contDiff_flow
    (continuous_neg_fderiv_metricGradient hf hG) (fun _ x => hasFDerivAt_neg_metricGradient hf hG x)
    ht

/-- **At a critical point the field `G ∇f` has the derivative `G(z) ∇²f(z)`**:
the term `DG(z)[·] ∇f(z)` of the product rule vanishes with `∇f(z)`. -/
theorem fderiv_metricGradient_of_gradient_eq_zero (hf : ContDiff ℝ 2 f) (hG : ContDiff ℝ 1 G)
    {z : E} (hz : gradient f z = 0) :
    fderiv ℝ (fun y => G y (gradient f y)) z = (G z).comp (fderiv ℝ (gradient f) z) := by
  rw [((hG.differentiable one_ne_zero z).hasFDerivAt.clm_apply
    ((contDiff_gradient hf).differentiable one_ne_zero z).hasFDerivAt).fderiv, hz, map_zero,
    add_zero]

/-- **At a critical point the derivative of the time-`t` map is
`exp (-t G(z) ∇²f(z))`**, for `t ≥ 0`: the trajectory through `z` is
constant, so the variational equation along it is `J' = -G(z) ∇²f(z) J`. -/
theorem hasFDerivAt_metricGradientFlow_of_gradient_eq_zero (hf : ContDiff ℝ 2 f)
    (hG : ContDiff ℝ 1 G) (hL : LipschitzWith L fun x => G x (gradient f x))
    (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M) {z : E} (hz : gradient f z = 0) {t : ℝ} (ht : 0 ≤ t) :
    HasFDerivAt (metricGradientFlow G f t)
      (exp (t • -(G z).comp (fderiv ℝ (gradient f) z))) z := by
  refine (isFlowField_neg_metricGradient hL hM).hasFDerivAt_flow
    (continuous_neg_fderiv_metricGradient hf hG)
    (fun _ x => hasFDerivAt_neg_metricGradient hf hG x) ht z
    (J := fun τ => exp (τ • -(G z).comp (fderiv ℝ (gradient f) z))) (exp_zero_smul _)
    (continuous_exp_smul _) fun τ _ => ?_
  rw [show flow (fun _ y => -G y (gradient f y)) 0 τ z = z from
    metricGradientFlow_eq_self hL hM hz τ, fderiv_metricGradient_of_gradient_eq_zero hf hG hz]
  exact hasDerivAt_exp_smul _ τ

/-- **The time-`t` map has an invertible derivative**, for `t ≥ 0`
(`det_fderiv_flow_ne_zero`). -/
theorem det_fderiv_metricGradientFlow_ne_zero (hf : ContDiff ℝ 2 f) (hG : ContDiff ℝ 1 G)
    (hL : LipschitzWith L fun x => G x (gradient f x)) (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M)
    {t : ℝ} (ht : 0 ≤ t) (x : E) : (fderiv ℝ (metricGradientFlow G f t) x).det ≠ 0 :=
  det_fderiv_flow_ne_zero (isFlowField_neg_metricGradient hL hM)
    (continuous_neg_fderiv_metricGradient hf hG)
    (fun _ y => hasFDerivAt_neg_metricGradient hf hG y) ht x

/-- **A strict saddle is a symmetrizable unstable fixed point of the time-one
map** when `G(z)` is symmetric positive definite. The derivative there is
`exp (-A)`, `A = G(z) ∇²f(z)`. For `T` with `T A T⁻¹` symmetric,
`T exp (-A) T⁻¹ = exp (-T A T⁻¹)` is symmetric (`conj_exp`,
`isSymmetric_exp`), and a negative eigenvalue `κ` of `A` gives the eigenvalue
`e^{-κ} > 1` of `exp (-A)` (`hasEigenvalue_exp`). -/
theorem IsStrictSaddle.isSymmetrizableUnstable_metricGradientFlow (hf : ContDiff ℝ 2 f)
    (hG : ContDiff ℝ 1 G) (hL : LipschitzWith L fun x => G x (gradient f x))
    (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M) {z : E} (hz : IsStrictSaddle f z)
    (hGs : (G z : E →ₗ[ℝ] E).IsSymmetric) (hGp : ∀ v, v ≠ 0 → 0 < ⟪G z v, v⟫) :
    IsSymmetrizableUnstable (metricGradientFlow G f 1) z := by
  obtain ⟨hcrit, lam, hlam, hev⟩ := hz
  set A := (G z).comp (fderiv ℝ (gradient f) z) with hA
  have hd : fderiv ℝ (metricGradientFlow G f 1) z = exp (-A) := by
    have h := (hasFDerivAt_metricGradientFlow_of_gradient_eq_zero hf hG hL hM hcrit
      zero_le_one).fderiv
    rwa [one_smul] at h
  obtain ⟨T, hT, κ, hκ, hevκ⟩ := exists_conj_isSymmetric_and_hasEigenvalue_neg hGs hGp
    (isSymmetric_fderiv_gradient hf z) hlam hev
  refine ⟨metricGradientFlow_eq_self hL hM hcrit 1, ⟨T, ?_⟩, Real.exp (-κ), ?_, ?_⟩
  · rw [hd, conj_exp]
    refine isSymmetric_exp fun u v => ?_
    have h := hT u v
    simp only [hA, ContinuousLinearMap.coe_coe, ContinuousLinearMap.comp_apply,
      neg_apply, ContinuousLinearEquiv.coe_coe, map_neg, inner_neg_left,
      inner_neg_right] at h ⊢
    rw [h]
  · rw [abs_of_pos (Real.exp_pos _), Real.one_lt_exp_iff]
    linarith
  · rw [hd]
    refine hasEigenvalue_exp ?_
    obtain ⟨w, hw⟩ := hevκ.exists_hasEigenvector
    refine Module.End.hasEigenvalue_of_hasEigenvector (x := w) ⟨?_, hw.2⟩
    have hAw : A w = κ • w := hw.apply_eq_smul
    rw [Module.End.mem_eigenspace_iff, ContinuousLinearMap.coe_coe,
      neg_apply, hAw, neg_smul]

/-- The hypotheses of the theorems above hold for `f = cos` on the line in the
constant field `G = 2`, with `L = M = 2`, `t = 1` and the strict saddle
`z = 0`, a maximum: `G ∇f = -2 sin` is `2`-Lipschitz and bounded by `2`, the
Hessian at `0` is `-1`, and `G(0) = 2` is symmetric positive definite. -/
example : let G : ℝ → ℝ →L[ℝ] ℝ := fun _ => (2 : ℝ) • ContinuousLinearMap.id ℝ ℝ
    ContDiff ℝ 2 Real.cos ∧ ContDiff ℝ 1 G ∧
      LipschitzWith 2 (fun x => G x (gradient Real.cos x)) ∧
      (∀ x, ‖G x (gradient Real.cos x)‖ ≤ 2) ∧ (0 : ℝ) ≤ 1 ∧ gradient Real.cos 0 = 0 ∧
      IsStrictSaddle Real.cos 0 ∧ (G 0 : ℝ →ₗ[ℝ] ℝ).IsSymmetric ∧
      ∀ v, v ≠ 0 → 0 < ⟪G 0 v, v⟫ := by
  intro G
  have hgrad : gradient Real.cos = fun x => -Real.sin x :=
    funext fun x => (Real.hasDerivAt_cos x).hasGradientAt'.gradient
  have hH : fderiv ℝ (gradient Real.cos) 0 =
      ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (-Real.cos 0) := by
    rw [hgrad]
    exact (Real.hasDerivAt_sin 0).neg.hasFDerivAt.fderiv
  have h0 : gradient Real.cos 0 = 0 := by simp [hgrad]
  have hGx : ∀ x, G x (gradient Real.cos x) = 2 * -Real.sin x := fun x => by simp [G, hgrad]
  refine ⟨Real.contDiff_cos, contDiff_const, ?_, fun x => ?_, zero_le_one, h0,
    ⟨h0, -1, by norm_num, ?_⟩, fun u v => ?_, fun v hv => ?_⟩
  · have h := (lipschitzWith_smul (2 : ℝ)).comp Real.lipschitzWith_sin.neg
    convert h using 1
    · simp
    · funext x
      simp [G, hgrad]
  · rw [hGx, norm_mul, norm_neg, Real.norm_eq_abs, Real.norm_eq_abs, abs_two]
    linarith [Real.abs_sin_le_one x]
  · refine Module.End.hasEigenvalue_of_hasEigenvector (x := 1) ⟨?_, one_ne_zero⟩
    simp [hH]
  · simp [G]
    ring
  · change 0 < ⟪(2 : ℝ) • v, v⟫
    rw [real_inner_smul_left]
    exact mul_pos two_pos (real_inner_self_pos.2 hv)

end Transformer.Landscape.SaddleAvoidance
