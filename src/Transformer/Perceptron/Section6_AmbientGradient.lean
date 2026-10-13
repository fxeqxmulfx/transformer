/-
# Actual interaction gradients and their ambient analyticity

ArXiv:2601.21366v2, §2.3, `eq: fulltrans.stat`, §5.3,
and §6, `prop: unified.log` and `rem:unifiedlog-B`.
The first integrated Taylor coefficient is the original kernel's
actual derivative. Currying that coefficient commutes with the
original probability integral, giving the weight's actual derivative.

The derivative equals the inner-product functional of the actual
vector integral. Fréchet-Riesz representation then transfers its
proved analyticity to that vector field. Passing the actual tangent
projection through the integral identifies the original field
`attentionGradMap`, whose ambient analyticity follows as well.
No density, atomicity or stationary-field identity is assumed.
-/

import Transformer.Perceptron.Section6_AmbientWeight
import Mathlib.Analysis.Calculus.FDeriv.Analytic
import Mathlib.Analysis.InnerProductSpace.Dual

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The actual integral of the original kernel derivatives.
Source: arXiv:2601.21366v2, §6, the first-variation derivative. -/
noncomputable def attentionCovectorMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : EucSpace d →L[ℝ] ℝ :=
  ∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • innerSL ℝ (B (y : EucSpace d))
    ∂(μ : Measure (SSphere d))

/-- The actual ambient vector integral before the tangent projection.
Source: arXiv:2601.21366v2, §2.3 and §6, `eq: fulltrans.stat`. -/
noncomputable def ambientAttentionGradMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : EucSpace d :=
  ∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • B (y : EucSpace d)
    ∂(μ : Measure (SSphere d))

/-- The original vector integrand is integrable against the original
probability law. Source: arXiv:2601.21366v2, §6, the first variation. -/
theorem integrable_ambientAttentionKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    Integrable (fun y : SSphere d => Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
      B (y : EucSpace d)) (μ : Measure (SSphere d)) := by
  have hB : Continuous (fun y : SSphere d => B (y : EucSpace d)) :=
    B.continuous_of_finiteDimensional.comp continuous_subtype_val
  have hf : Continuous (fun y : SSphere d => Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
      B (y : EucSpace d)) := (Real.continuous_exp.comp (continuous_const.inner hB)).smul hB
  exact hf.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)

/-- The first true Taylor coefficient is the original kernel derivative.
Source: arXiv:2601.21366v2, §5.3 and §6, differentiating the exponential.
Uniqueness compares the proved Taylor derivative with the direct chain rule. -/
theorem curry_attentionKernelSeries_one {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x : EucSpace d) (y : SSphere d) :
    continuousMultilinearCurryFin1 ℝ (EucSpace d) ℝ (attentionKernelSeries B x y 1) =
      Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • innerSL ℝ (B (y : EucSpace d)) := by
  have h := ((innerSL ℝ (B (y : EucSpace d))).hasFDerivAt (x := x)).exp
  have heq : (fun z : EucSpace d => Real.exp (innerSL ℝ (B (y : EucSpace d)) z)) =
      fun z => Real.exp (inner (𝕜 := ℝ) z (B (y : EucSpace d))) := by
    funext z
    rw [innerSL_apply_apply, real_inner_comm z (B (y : EucSpace d))]
  rw [heq, innerSL_apply_apply, real_inner_comm x (B (y : EucSpace d))] at h
  exact (hasFPowerSeriesOnBall_attentionKernel B x y).hasFPowerSeriesAt.hasFDerivAt.unique h

/-- The actual derivative of the original weight is the actual
covector integral. Source: arXiv:2601.21366v2, §5.3 and §6,
the analytic first-variation field in `eq: fulltrans.stat`. -/
theorem hasFDerivAt_attentionWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    HasFDerivAt (attentionWeightMap B μ) (attentionCovectorMap B μ x) x := by
  apply (hasFPowerSeriesOnBall_attentionWeightMap B μ x).hasFPowerSeriesAt.hasFDerivAt.congr_fderiv
  let C := (continuousMultilinearCurryFin1 ℝ (EucSpace d) ℝ).toContinuousLinearEquiv
  change C (∫ y, attentionKernelSeries B x y 1 ∂(μ : Measure (SSphere d))) =
    ∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • innerSL ℝ (B (y : EucSpace d))
      ∂(μ : Measure (SSphere d))
  rw [← C.integral_comp_comm (fun y => attentionKernelSeries B x y 1)]
  exact integral_congr_ae (ae_of_all _ (fun y => curry_attentionKernelSeries_one B x y))

/-- The total Fréchet derivative is the proved actual covector integral.
Source: arXiv:2601.21366v2, §6, the derivative used in `H_log`. -/
theorem fderiv_attentionWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    fderiv ℝ (attentionWeightMap B μ) x = attentionCovectorMap B μ x :=
  (hasFDerivAt_attentionWeightMap B μ x).fderiv

/-- The actual derivative functional is represented by the original
vector integral. Source: arXiv:2601.21366v2, §6, the first-variation gradient.
The real inner-product map commutes with the actual integrable vector field. -/
theorem attentionCovectorMap_eq_innerSL_ambientGradMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    attentionCovectorMap B μ x = innerSL ℝ (ambientAttentionGradMap B μ x) := by
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  change (∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
    innerSL ℝ (B (y : EucSpace d)) ∂(μ : Measure (SSphere d))) =
      L (∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
        B (y : EucSpace d) ∂(μ : Measure (SSphere d)))
  simpa only [L.map_smul] using L.integral_comp_comm (integrable_ambientAttentionKernel B μ x)

/-- The actual covector integral is analytic, since it is the derivative
of the proved analytic weight. Source: arXiv:2601.21366v2, §5.3 and §6,
analyticity of the first-variation field. -/
theorem analyticOnNhd_attentionCovectorMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) : AnalyticOnNhd ℝ (attentionCovectorMap B μ) univ := by
  have heq : fderiv ℝ (attentionWeightMap B μ) = attentionCovectorMap B μ :=
    funext (fderiv_attentionWeightMap B μ)
  rw [← heq]
  exact (analyticOnNhd_attentionWeightMap B μ).fderiv

/-- The original ambient vector gradient is analytic.
Source: arXiv:2601.21366v2, §5.3 and §6, the analytic first variation.
Fréchet-Riesz representation is applied to its proved actual derivative. -/
theorem analyticOnNhd_ambientAttentionGradMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) : AnalyticOnNhd ℝ (ambientAttentionGradMap B μ) univ := by
  let R : (EucSpace d →L[ℝ] ℝ) →L[ℝ] EucSpace d :=
    (InnerProductSpace.toDual ℝ (EucSpace d)).symm.toContinuousLinearEquiv.toContinuousLinearMap
  have heq : ambientAttentionGradMap B μ = fun x => R (attentionCovectorMap B μ x) := by
    funext x
    apply (InnerProductSpace.toDual ℝ (EucSpace d)).injective
    change (InnerProductSpace.toDual ℝ (EucSpace d)) (ambientAttentionGradMap B μ x) =
      (InnerProductSpace.toDual ℝ (EucSpace d))
        ((InnerProductSpace.toDual ℝ (EucSpace d)).symm (attentionCovectorMap B μ x))
    rw [LinearIsometryEquiv.apply_symm_apply]
    ext v
    rw [InnerProductSpace.toDual_apply_apply, attentionCovectorMap_eq_innerSL_ambientGradMap,
      innerSL_apply_apply]
  rw [heq]
  exact fun x hx => (R.analyticAt _).comp (analyticOnNhd_attentionCovectorMap B μ x hx)

/-- Passing the actual linear projection through the original integral
gives exactly the original spherical attention field.
Source: arXiv:2601.21366v2, §2.3 and §6, `eq: fulltrans.stat`. -/
theorem attentionGradMap_eq_proj_ambientGradMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    attentionGradMap B μ x = proj d x (ambientAttentionGradMap B μ x) := by
  let P : EucSpace d →L[ℝ] EucSpace d :=
    ContinuousLinearMap.id ℝ (EucSpace d) - (innerSL ℝ x).smulRight x
  have hP (v : EucSpace d) : P v = proj d x v := rfl
  change (∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
      proj d x (B (y : EucSpace d)) ∂(μ : Measure (SSphere d))) =
    P (∫ y, Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) •
      B (y : EucSpace d) ∂(μ : Measure (SSphere d)))
  rw [← P.integral_comp_comm (integrable_ambientAttentionKernel B μ x)]
  apply integral_congr_ae
  exact ae_of_all _ (fun y => by
    change Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • proj d x (B (y : EucSpace d)) =
      P (Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • B (y : EucSpace d))
    rw [P.map_smul, hP])

/-- The original projected field is analytic in the actual ambient
space. Source: arXiv:2601.21366v2, §5.3 and §6, the analytic
stationary field. The projection uses the same original point. -/
theorem analyticOnNhd_attentionGradMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) : AnalyticOnNhd ℝ (attentionGradMap B μ) univ := by
  have heq : attentionGradMap B μ = fun x => proj d x (ambientAttentionGradMap B μ x) :=
    funext (attentionGradMap_eq_proj_ambientGradMap B μ)
  rw [heq]
  intro x hx
  let L : EucSpace d →L[ℝ] (EucSpace d →L[ℝ] ℝ) := innerSL ℝ
  have ha := analyticOnNhd_ambientAttentionGradMap B μ x hx
  have hi : AnalyticAt ℝ (fun z => inner (𝕜 := ℝ) z (ambientAttentionGradMap B μ z)) x :=
    (L.analyticAt_bilinear (x, ambientAttentionGradMap B μ x)).comp
      (f := fun z => (z, ambientAttentionGradMap B μ z)) (analyticAt_id.prod ha)
  exact ha.sub (hi.smul analyticAt_id)

/-- For actual identity attention and a nonempty Dirac law, the
original ambient vector gradient is the atom multiplied by its actual
kernel value. Source: arXiv:2601.21366v2, §6, the original first variation. -/
example (x : EucSpace 2) : ambientAttentionGradMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
    (Perspective.diracProb 2 (basePoint 1)) x =
      Real.exp (inner (𝕜 := ℝ) x ((basePoint 1 : SSphere 2) : EucSpace 2)) •
        ((basePoint 1 : SSphere 2) : EucSpace 2) := by
  change (∫ y, Real.exp (inner (𝕜 := ℝ) x (y : EucSpace 2)) • (y : EucSpace 2)
    ∂Measure.dirac (basePoint 1)) = _
  rw [integral_dirac]

end Transformer.Perceptron
