/-
# Analyticity of the actual circle interaction weight

ArXiv:2601.21366v2, §6, proof of `prop: unified.log` and
`rem:unifiedlog-B`. The actual weight `∫ exp(xᵀBy) dμ(y)` is
strictly positive and real-analytic in the circle angle.

Its complex extension uses the two coordinates of `By` together
with complex cosine and sine. The extension and its actual derivative
are jointly continuous, so the preceding dominated-integral argument
applies. The trace equality identifies the integral with the original
real interaction. The actual angular derivative is also computed
directly against the original probability measure.
-/

import Transformer.Perceptron.Section6_AnalyticIntegral
import Transformer.Perceptron.Section5_MapKernel
import Transformer.Perceptron.Section5_DominatedVariation
import Transformer.Perceptron.NormalizedMap
import Mathlib.Analysis.Calculus.FDeriv.Analytic

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The complex extension of the original matrix kernel in the angle.
Source: arXiv:2601.21366v2, §6, the analytic interaction in
`prop: unified.log` and `rem:unifiedlog-B`. -/
noncomputable def complexCircleMapKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (z : ℂ) (y : SSphere 2) : ℂ :=
  Complex.exp (Complex.cos z * ((B (y : EucSpace 2)) 0 : ℂ) +
    Complex.sin z * ((B (y : EucSpace 2)) 1 : ℂ))

/-- Its actual complex derivative integrand. Source: arXiv:2601.21366v2,
§6, the analyticity calculation for the original matrix interaction. -/
noncomputable def complexCircleMapForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (z : ℂ) (y : SSphere 2) : ℂ :=
  complexCircleMapKernel B z y *
    (-Complex.sin z * ((B (y : EucSpace 2)) 0 : ℂ) +
      Complex.cos z * ((B (y : EucSpace 2)) 1 : ℂ))

/-- The original interaction weight at the actual circle point.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`. -/
noncomputable def circleAngularWeightMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) : ℝ :=
  attentionWeightMap B μ (circlePoint θ : EucSpace 2)

/-- The complex extension is jointly continuous in its actual arguments.
Source: arXiv:2601.21366v2, §6, the analytic interaction calculation. -/
theorem continuous_complexCircleMapKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) :
    Continuous (fun q : ℂ × SSphere 2 => complexCircleMapKernel B q.1 q.2) := by
  have hB := B.continuous_of_finiteDimensional
  unfold complexCircleMapKernel
  fun_prop

/-- Its actual derivative is jointly continuous as well. Source:
arXiv:2601.21366v2, §6, domination for the analytic interaction. -/
theorem continuous_complexCircleMapForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) :
    Continuous (fun q : ℂ × SSphere 2 => complexCircleMapForce B q.1 q.2) := by
  have hB := B.continuous_of_finiteDimensional
  have hK := continuous_complexCircleMapKernel B
  unfold complexCircleMapForce
  fun_prop

/-- The supplied complex derivative is the actual derivative of the
extension. Source: arXiv:2601.21366v2, §6, the exponential interaction. -/
theorem hasDerivAt_complexCircleMapKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (y : SSphere 2) (z : ℂ) :
    HasDerivAt (fun w => complexCircleMapKernel B w y) (complexCircleMapForce B z y) z := by
  have hc := (Complex.hasDerivAt_cos z).mul_const ((B (y : EucSpace 2)) 0 : ℂ)
  have hs := (Complex.hasDerivAt_sin z).mul_const ((B (y : EucSpace 2)) 1 : ℂ)
  exact (hc.fun_add hs).cexp

/-- At a real angle the extension is exactly the original real kernel.
Source: arXiv:2601.21366v2, §6, the actual interaction weight.
This equality checks the original matrix and sphere coordinates. -/
theorem complexCircleMapKernel_real (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (y : SSphere 2) (θ : ℝ) :
    complexCircleMapKernel B (θ : ℂ) y = (circleMapPairKernel B (circlePoint θ) y : ℂ) := by
  have hinner : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) (B (y : EucSpace 2)) =
      Real.cos θ * (B (y : EucSpace 2)) 0 + Real.sin θ * (B (y : EucSpace 2)) 1 := by
    simp [circlePoint, greatCircle, inner_add_left, real_inner_smul_left,
      basePoint, secondAxis, EuclideanSpace.inner_single_left]
  unfold complexCircleMapKernel circleMapPairKernel
  rw [hinner, ← Complex.ofReal_cos, ← Complex.ofReal_sin,
    ← Complex.ofReal_mul, ← Complex.ofReal_mul, ← Complex.ofReal_add, ← Complex.ofReal_exp]

/-- The actual interaction weight is real-analytic in every angle.
Source: arXiv:2601.21366v2, §6, the analytic first variation used in
`prop: unified.log` and `rem:unifiedlog-B`. Arbitrary probability
measures are allowed; no finite-support or density input is used. -/
theorem analyticOnNhd_circleAngularWeightMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (μ : Perspective.ProbSphere 2) : AnalyticOnNhd ℝ (circleAngularWeightMap B μ) univ := by
  change AnalyticOnNhd ℝ (fun θ => ∫ y, circleMapPairKernel B (circlePoint θ) y
    ∂(μ : Measure (SSphere 2))) univ
  exact analyticOnNhd_integral_real_trace (μ : Measure (SSphere 2))
    (complexCircleMapKernel B) (complexCircleMapForce B)
    (continuous_complexCircleMapKernel B) (continuous_complexCircleMapForce B)
    (fun z y => hasDerivAt_complexCircleMapKernel B y z)
    (fun θ y => circleMapPairKernel B (circlePoint θ) y)
    (fun θ y => complexCircleMapKernel_real B y θ)

/-- The actual weight derivative is real-analytic in every angle.
Source: arXiv:2601.21366v2, §6, the analytic stationary field
used in the proof of `prop: unified.log`. -/
theorem analyticOnNhd_deriv_circleAngularWeightMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (μ : Perspective.ProbSphere 2) :
    AnalyticOnNhd ℝ (deriv (circleAngularWeightMap B μ)) univ :=
  (analyticOnNhd_circleAngularWeightMap B μ).deriv

/-- The actual angular derivative of the original weight is its force
integral. Source: arXiv:2601.21366v2, §6, the gradient in
`eq: fulltrans.stat`. Domination follows from compact pair continuity. -/
theorem hasDerivAt_circleAngularWeightMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    HasDerivAt (circleAngularWeightMap B μ)
      (∫ y, circleMapPairForce B (circlePoint θ) y ∂(μ : Measure (SSphere 2))) θ := by
  obtain ⟨C, -, hC⟩ := exists_pos_bound_continuous_compact _ (continuous_circleMapPairForce B)
  exact hasDerivAt_integral_on_compact (μ : Measure (SSphere 2))
    (fun r y => circleMapPairKernel B (circlePoint r) y)
    (fun r y => circleMapPairForce B (circlePoint r) y)
    (fun r => (continuous_circleMapPairKernel B).comp (continuous_const.prodMk continuous_id))
    (fun r => (continuous_circleMapPairForce B).comp (continuous_const.prodMk continuous_id))
    C (fun r y => hC (circlePoint r, y)) (fun r y => hasDerivAt_circleMapPairKernel B y r) θ

/-- The interaction weight is strictly positive at every actual angle.
Source: arXiv:2601.21366v2, §6, the strict positivity used in
`prop: unified.log`. The entire original probability measure is retained. -/
theorem circleAngularWeightMap_pos (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) : 0 < circleAngularWeightMap B μ θ := by
  have hI : Integrable (circleMapPairKernel B (circlePoint θ)) (μ : Measure (SSphere 2)) :=
    ((continuous_circleMapPairKernel B).comp
      (continuous_const.prodMk continuous_id)).integrable_of_hasCompactSupport
        (HasCompactSupport.of_compactSpace _)
  apply (integral_pos_iff_support_of_nonneg (fun y => (Real.exp_pos _).le) hI).mpr
  have hsupport : Function.support (circleMapPairKernel B (circlePoint θ)) = univ := by
    ext y
    simp only [Function.mem_support, mem_univ, iff_true]
    exact (Real.exp_pos _).ne'
  change 0 < (μ : Measure (SSphere 2)) (Function.support (circleMapPairKernel B (circlePoint θ)))
  rw [hsupport]
  simp

/-- At `B=id` the actual Dirac weight is the source's exponential cosine
kernel. Source: arXiv:2601.21366v2, §6 and §5.2, the original circle kernel.
This checks the complex trace against an actual probability measure. -/
theorem circleAngularWeightMap_dirac_id (θ φ : ℝ) :
    circleAngularWeightMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (Perspective.diracProb 2 (circlePoint φ)) θ = Real.exp (Real.cos (θ - φ)) := by
  change (∫ y, circleMapPairKernel (LinearMap.id (R := ℝ) (M := EucSpace 2))
    (circlePoint θ) y ∂Measure.dirac (circlePoint φ)) = _
  rw [integral_dirac, circleMapPairKernel, LinearMap.id_apply, inner_circlePoint_circlePoint]

end Transformer.Perceptron
