module

public import Transformer.Landscape.SaddleAvoidance.Global
public import Mathlib.Analysis.Calculus.FDeriv.Symmetric
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Analysis.SpecificLimits.Normed

/-!
# Saddle avoidance — gradient descent

Let `f` be a `C²` function on a finite-dimensional real inner product space
whose Hessian has operator norm at most `L`, and let `0 < η`, `η L < 1`.
Gradient descent `x ↦ x - η ∇f(x)` converges to a strict saddle of `f` only
from a null set of starts (`measure_tendsto_strictSaddle_eq_zero`).

The step map is `C¹` with derivative `1 - η ∇²f(x)`, whose determinant is
nonzero because `‖η ∇²f(x)‖ ≤ η L < 1` (`det_fderiv_gradientStep_ne_zero`). At
a strict saddle `z` the Hessian is symmetric (`isSymmetric_fderiv_gradient`)
and has a negative eigenvalue `λ`, so the derivative of the step is symmetric
with the eigenvalue `1 - ηλ > 1`, and `z` is a symmetric unstable fixed
point (`IsStrictSaddle.isSymmetricUnstable`).

This is the gradient-descent corollary in §5.1 of arXiv:1710.07406, and, for a
single strict saddle and a start drawn from a law with a density, `thm:main` of
arXiv:1602.04915 (`measure_tendsto_eq_zero_of_isStrictSaddle`).
-/

@[expose] public section

open Filter MeasureTheory InnerProductSpace
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- The gradient of a `C²` function is `C¹`. -/
theorem contDiff_gradient {f : E → ℝ} (hf : ContDiff ℝ 2 f) : ContDiff ℝ 1 (gradient f) :=
  (toDual ℝ E).symm.contDiff.comp (hf.fderiv_right (m := 1) (by norm_num))

/-- **The Hessian of a `C²` function is symmetric**: pairing the derivative
of the gradient with `v` gives the second derivative applied to `v`, which is
symmetric (`ContDiffAt.isSymmSndFDerivAt`). -/
theorem isSymmetric_fderiv_gradient {f : E → ℝ} (hf : ContDiff ℝ 2 f) (z : E) :
    (fderiv ℝ (gradient f) z : E →ₗ[ℝ] E).IsSymmetric := by
  have hsnd := (hf.contDiffAt (x := z)).isSymmSndFDerivAt
    (by simp [minSmoothness_of_isRCLikeNormedField])
  have hpair : ∀ u v, ⟪v, fderiv ℝ (gradient f) z u⟫ = fderiv ℝ (fderiv ℝ f) z u v := by
    intro u v
    have h1 := (innerSL ℝ v).hasFDerivAt.comp z
      ((contDiff_gradient hf).differentiable one_ne_zero z).hasFDerivAt
    have h2 := (ContinuousLinearMap.apply ℝ ℝ v).hasFDerivAt.comp z
      ((hf.fderiv_right (m := 1) (by norm_num)).differentiable one_ne_zero z).hasFDerivAt
    have hfun : (innerSL ℝ v) ∘ gradient f = (ContinuousLinearMap.apply ℝ ℝ v) ∘ fderiv ℝ f := by
      funext x
      simp only [Function.comp_apply, innerSL_apply_apply, gradient,
        ContinuousLinearMap.apply_apply]
      rw [real_inner_comm, toDual_symm_apply]
    rw [hfun] at h1
    simpa using congrArg (fun T => T u) (h1.unique h2)
  intro u v
  simp only [ContinuousLinearMap.coe_coe]
  rw [real_inner_comm v, hpair u v, hpair v u]
  exact hsnd u v

theorem hasFDerivAt_gradientStep {f : E → ℝ} (hf : ContDiff ℝ 2 f) (η : ℝ) (x : E) :
    HasFDerivAt (gradientStep f η) (ContinuousLinearMap.id ℝ E - η • fderiv ℝ (gradient f) x) x :=
  (hasFDerivAt_id x).sub
    (((contDiff_gradient hf).differentiable one_ne_zero x).hasFDerivAt.const_smul η)

theorem contDiff_gradientStep {f : E → ℝ} (hf : ContDiff ℝ 2 f) (η : ℝ) :
    ContDiff ℝ 1 (gradientStep f η) :=
  contDiff_id.sub ((contDiff_gradient hf).const_smul η)

/-- **The step map has invertible derivative at a point** where
`‖∇²f(x)‖ ≤ L`, when `0 ≤ η` and `η L < 1`: `1 - η ∇²f(x)` is a unit since
`‖η ∇²f(x)‖ < 1`. -/
theorem det_fderiv_gradientStep_ne_zero_of_norm_le {f : E → ℝ} (hf : ContDiff ℝ 2 f)
    {L η : ℝ} {x : E} (hx : ‖fderiv ℝ (gradient f) x‖ ≤ L) (hη : 0 ≤ η) (hηL : η * L < 1) :
    (fderiv ℝ (gradientStep f η) x).det ≠ 0 := by
  rw [(hasFDerivAt_gradientStep hf η x).fderiv]
  have hnorm : ‖η • fderiv ℝ (gradient f) x‖ < 1 := by
    rw [norm_smul, Real.norm_of_nonneg hη]
    exact (mul_le_mul_of_nonneg_left hx hη).trans_lt hηL
  have hunit : IsUnit (ContinuousLinearMap.id ℝ E - η • fderiv ℝ (gradient f) x) := by
    simpa [Units.val_oneSub, ContinuousLinearMap.one_def] using (Units.oneSub _ hnorm).isUnit
  exact ((LinearMap.isUnit_iff_isUnit_det _).1
    (hunit.map ContinuousLinearMap.toLinearMapRingHom)).ne_zero

/-- **The step map has invertible derivative** when `η L < 1`
(`prop:gd-det-not-0` of arXiv:1710.07406, whose step size `α < 1/L` is
tacitly positive; `0 ≤ η` suffices here). -/
theorem det_fderiv_gradientStep_ne_zero {f : E → ℝ} (hf : ContDiff ℝ 2 f) {L η : ℝ}
    (hL : ∀ x, ‖fderiv ℝ (gradient f) x‖ ≤ L) (hη : 0 ≤ η) (hηL : η * L < 1) (x : E) :
    (fderiv ℝ (gradientStep f η) x).det ≠ 0 :=
  det_fderiv_gradientStep_ne_zero_of_norm_le hf (hL x) hη hηL

/-- **A strict saddle is a symmetric unstable fixed point of gradient
descent** (`prop:gd-unstable` of arXiv:1710.07406): a negative eigenvalue `λ`
of the Hessian gives the eigenvalue `1 - ηλ > 1` of the step's derivative. -/
theorem IsStrictSaddle.isSymmetricUnstable {f : E → ℝ} (hf : ContDiff ℝ 2 f) {η : ℝ}
    (hη : 0 < η) {z : E} (hz : IsStrictSaddle f z) : IsSymmetricUnstable (gradientStep f η) z := by
  obtain ⟨hcrit, lam, hlam, hev⟩ := hz
  have hd := (hasFDerivAt_gradientStep hf η z).fderiv
  refine ⟨by simp [gradientStep, hcrit], ?_, 1 - η * lam, ?_, ?_⟩
  · rw [hd]
    intro u v
    have hs := isSymmetric_fderiv_gradient hf z u v
    simp only [ContinuousLinearMap.coe_coe] at hs
    simp only [ContinuousLinearMap.coe_coe, sub_apply, smul_apply, ContinuousLinearMap.id_apply,
      inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right, hs]
  · rw [abs_of_pos (by nlinarith)]
    nlinarith
  · obtain ⟨w, hw⟩ := hev.exists_hasEigenvector
    refine Module.End.hasEigenvalue_of_hasEigenvector (x := w) ⟨?_, hw.2⟩
    have hHw : fderiv ℝ (gradient f) z w = lam • w := hw.apply_eq_smul
    simp [hd, hHw, sub_smul, smul_smul]

variable [MeasurableSpace E] [BorelSpace E]

/-- **Gradient descent almost never converges to a strict saddle.**  For `f`
of class `C²` with `‖∇²f(x)‖ ≤ L` everywhere and a step size `0 < η` with
`η L < 1`, the starts from which gradient descent converges to a strict
saddle form a null set: the gradient-descent corollary in §5.1 of
arXiv:1710.07406, whose step size `α`, tacitly positive, with `α < 1/L` is
`η` here. -/
theorem measure_tendsto_strictSaddle_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure] {f : E → ℝ}
    (hf : ContDiff ℝ 2 f) {L : ℝ} (hL : ∀ x, ‖fderiv ℝ (gradient f) x‖ ≤ L) {η : ℝ}
    (hη : 0 < η) (hηL : η * L < 1) :
    μ {x | ∃ z, IsStrictSaddle f z ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z)} =
      0 := by
  refine measure_mono_null ?_ (measure_tendsto_symmetricUnstable_eq_zero μ
    (contDiff_gradientStep hf η) (det_fderiv_gradientStep_ne_zero hf hL hη.le hηL))
  rintro x ⟨z, hz, hlim⟩
  exact ⟨z, hz.isSymmetricUnstable hf hη, hlim⟩

/-- **`thm:main` of arXiv:1602.04915.**  For `f` of class `C²` with an
`L`-Lipschitz gradient, a step size `0 < η` with `η L < 1` (the source's
`0 < α < 1/L`) and a strict saddle `z`, gradient descent started from a law
`ν` absolutely continuous with respect to Lebesgue measure converges to `z`
with probability zero. -/
theorem measure_tendsto_eq_zero_of_isStrictSaddle (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) {f : E → ℝ} (hf : ContDiff ℝ 2 f) {L : ℝ≥0}
    (hL : LipschitzWith L (gradient f)) {η : ℝ} (hη : 0 < η) (hηL : η * L < 1) {z : E}
    (hz : IsStrictSaddle f z) :
    ν {x | Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z)} = 0 := by
  refine hν (measure_mono_null ?_ (measure_tendsto_strictSaddle_eq_zero μ hf
    (fun _ => norm_fderiv_le_of_lipschitz ℝ hL) hη hηL))
  exact fun x hx => ⟨z, hz, hx⟩

/-- The hypotheses of the theorems above hold for `f(x) = -‖x‖² / 2` on the
line, with `L = 1`, `η = 1/2` and the restriction of Lebesgue measure to the
unit ball: the gradient of `f` is `-x`, its Hessian is `-1`, and `0` is a
strict saddle, a maximum with the negative eigenvalue `-1`. -/
example : let f := fun x : EuclideanSpace ℝ (Fin 1) => -(1 / 2 : ℝ) * ‖x‖ ^ 2
    ContDiff ℝ 2 f ∧ (∀ x, ‖fderiv ℝ (gradient f) x‖ ≤ 1) ∧ LipschitzWith 1 (gradient f) ∧
      (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * 1 < 1 ∧ IsStrictSaddle f 0 ∧
      (volume : Measure (EuclideanSpace ℝ (Fin 1))).restrict (Metric.ball 0 1) ≪ volume := by
  intro f
  have hgrad : gradient f = fun x => -x := by
    funext y
    apply HasGradientAt.gradient
    rw [hasGradientAt_iff_hasFDerivAt]
    convert (hasStrictFDerivAt_norm_sq y).hasFDerivAt.const_mul (-(1 / 2 : ℝ)) using 1
    ext z
    simp only [map_neg, neg_apply, toDual_apply_apply, smul_apply, innerSL_apply_apply,
      smul_eq_mul]
    ring
  have hd : ∀ x, fderiv ℝ (gradient f) x = -ContinuousLinearMap.id ℝ _ := fun x => by
    rw [hgrad]
    exact (hasFDerivAt_id x).neg.fderiv
  refine ⟨contDiff_const.mul (contDiff_norm_sq ℝ), fun x => ?_, ?_, by norm_num, by norm_num,
    ⟨by simp [hgrad], -1, by norm_num, ?_⟩,
    Measure.absolutelyContinuous_of_le Measure.restrict_le_self⟩
  · rw [hd, norm_neg]
    exact ContinuousLinearMap.norm_id_le
  · rw [hgrad]
    exact LipschitzWith.id.neg
  · refine Module.End.hasEigenvalue_of_hasEigenvector (x := EuclideanSpace.single 0 1) ⟨?_, ?_⟩
    · simp [hd]
    · simp

end Transformer.Landscape.SaddleAvoidance
