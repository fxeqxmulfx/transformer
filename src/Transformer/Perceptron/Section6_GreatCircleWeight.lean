module

public import Transformer.Perceptron.Section6_AnalyticIntegral
public import Transformer.Perceptron.NormalizedMap
public import Transformer.Perceptron.Analytic
public import Transformer.Perceptron.Section5_CircleGeometry
public import Mathlib.Analysis.Calculus.FDeriv.Analytic

/-!
# Analytic interaction weights along great circles in every dimension

ArXiv:2601.21366v2, §6, proof of `prop: unified.log` and
`rem:unifiedlog-B`, and §5.3, the analytic first variation.
The original exponential interaction restricted to any great circle
is analytic in its real angle and has its actual differentiated integral.

This extends the two-dimensional coordinate calculation to arbitrary
ambient dimension. The complex extension uses the real inner products
with the two actual vectors spanning the curve. Neither orthonormality
nor a finite-support presentation is needed for this analytic calculation.
The original probability measure is retained in the weight and derivative.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Complex extension of the actual kernel along the two-vector curve.
Source: arXiv:2601.21366v2, §6, the analytic exponential first variation. -/
noncomputable def complexGreatCircleMapKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) (z : ℂ) (y : SSphere d) : ℂ :=
  Complex.exp (Complex.cos z * (inner (𝕜 := ℝ) u (B (y : EucSpace d)) : ℂ) +
    Complex.sin z * (inner (𝕜 := ℝ) w (B (y : EucSpace d)) : ℂ))

/-- Its actual complex angular derivative. Source: arXiv:2601.21366v2,
§6, the analytic first-variation field in `prop: unified.log`. -/
noncomputable def complexGreatCircleMapForce {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) (z : ℂ) (y : SSphere d) : ℂ :=
  complexGreatCircleMapKernel B u w z y *
    (-Complex.sin z * (inner (𝕜 := ℝ) u (B (y : EucSpace d)) : ℂ) +
      Complex.cos z * (inner (𝕜 := ℝ) w (B (y : EucSpace d)) : ℂ))

/-- The original real interaction weight restricted to the actual curve.
Source: arXiv:2601.21366v2, §6, the first variation in `H_log`. -/
noncomputable def greatCircleWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (u w : EucSpace d) (θ : ℝ) : ℝ :=
  attentionWeightMap B μ (greatCircle u w θ)

/-- The actual real differentiated kernel along that curve.
Source: arXiv:2601.21366v2, §6, the angular first-variation derivative. -/
noncomputable def greatCircleMapWeightForce {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) (θ : ℝ) (y : SSphere d) : ℝ :=
  Real.exp (inner (𝕜 := ℝ) (greatCircle u w θ) (B (y : EucSpace d))) *
    inner (𝕜 := ℝ) (-Real.sin θ • u + Real.cos θ • w) (B (y : EucSpace d))

/-- The complex kernel is jointly continuous in its actual arguments.
Source: arXiv:2601.21366v2, §6, domination of the analytic interaction. -/
theorem continuous_complexGreatCircleMapKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) :
    Continuous (fun q : ℂ × SSphere d => complexGreatCircleMapKernel B u w q.1 q.2) := by
  have hB := B.continuous_of_finiteDimensional
  unfold complexGreatCircleMapKernel
  fun_prop

/-- Its actual derivative is jointly continuous as well.
Source: arXiv:2601.21366v2, §6, local domination of the derivative. -/
theorem continuous_complexGreatCircleMapForce {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) :
    Continuous (fun q : ℂ × SSphere d => complexGreatCircleMapForce B u w q.1 q.2) := by
  have hB := B.continuous_of_finiteDimensional
  have hK := continuous_complexGreatCircleMapKernel B u w
  unfold complexGreatCircleMapForce
  fun_prop

/-- The supplied complex force is proved to be the actual derivative.
Source: arXiv:2601.21366v2, §6, the analytic exponential kernel. -/
theorem hasDerivAt_complexGreatCircleMapKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) (y : SSphere d) (z : ℂ) :
    HasDerivAt (fun r => complexGreatCircleMapKernel B u w r y)
      (complexGreatCircleMapForce B u w z y) z := by
  exact (((Complex.hasDerivAt_cos z).mul_const (inner (𝕜 := ℝ) u (B (y : EucSpace d)) : ℂ)).fun_add
    ((Complex.hasDerivAt_sin z).mul_const (inner (𝕜 := ℝ) w (B (y : EucSpace d)) : ℂ))).cexp

/-- The complex trace equals the original real kernel at the actual point.
Source: arXiv:2601.21366v2, §6, the original interaction in `H_log`. -/
theorem complexGreatCircleMapKernel_real {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) (y : SSphere d) (θ : ℝ) :
    complexGreatCircleMapKernel B u w (θ : ℂ) y =
      (Real.exp (inner (𝕜 := ℝ) (greatCircle u w θ) (B (y : EucSpace d))) : ℂ) := by
  simp only [complexGreatCircleMapKernel, greatCircle, inner_add_left,
    real_inner_smul_left, Complex.ofReal_exp, Complex.ofReal_add,
    Complex.ofReal_mul, Complex.ofReal_cos, Complex.ofReal_sin]

/-- The complex derivative's trace is the actual original real derivative.
Source: arXiv:2601.21366v2, §6, differentiating the first variation. -/
theorem complexGreatCircleMapForce_real {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (u w : EucSpace d) (y : SSphere d) (θ : ℝ) :
    complexGreatCircleMapForce B u w (θ : ℂ) y = (greatCircleMapWeightForce B u w θ y : ℂ) := by
  simp only [complexGreatCircleMapForce, complexGreatCircleMapKernel_real,
    greatCircleMapWeightForce, inner_add_left, real_inner_smul_left,
    Complex.ofReal_mul, Complex.ofReal_add, Complex.ofReal_neg,
    Complex.ofReal_sin, Complex.ofReal_cos]

/-- The original weight is analytic along every actual great circle in
every dimension. Source: arXiv:2601.21366v2, §6, the analytic first
variation in `prop: unified.log`. Arbitrary probability measures are allowed. -/
theorem analyticOnNhd_greatCircleWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (u w : EucSpace d) :
    AnalyticOnNhd ℝ (greatCircleWeightMap B μ u w) univ := by
  exact analyticOnNhd_integral_real_trace (μ : Measure (SSphere d))
    (complexGreatCircleMapKernel B u w) (complexGreatCircleMapForce B u w)
    (continuous_complexGreatCircleMapKernel B u w) (continuous_complexGreatCircleMapForce B u w)
    (fun z y => hasDerivAt_complexGreatCircleMapKernel B u w y z)
    (fun θ y => Real.exp (inner (𝕜 := ℝ) (greatCircle u w θ) (B (y : EucSpace d))))
    (fun θ y => complexGreatCircleMapKernel_real B u w y θ)

/-- The actual derivative is the integral of the original differentiated
kernel, with the original measure. Source: arXiv:2601.21366v2, §6,
the normalized first-variation equation `eq: fulltrans.stat`. -/
theorem hasDerivAt_greatCircleWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (u w : EucSpace d) (θ : ℝ) :
    HasDerivAt (greatCircleWeightMap B μ u w)
      (∫ y, greatCircleMapWeightForce B u w θ y ∂(μ : Measure (SSphere d))) θ := by
  have h := hasDerivAt_real_part_complex_integral (μ : Measure (SSphere d))
    (complexGreatCircleMapKernel B u w) (complexGreatCircleMapForce B u w)
    (continuous_complexGreatCircleMapKernel B u w) (continuous_complexGreatCircleMapForce B u w)
    (fun z y => hasDerivAt_complexGreatCircleMapKernel B u w y z) θ
  have heq : (fun r : ℝ => (∫ y, complexGreatCircleMapKernel B u w (r : ℂ) y
      ∂(μ : Measure (SSphere d))).re) = greatCircleWeightMap B μ u w := by
    funext r
    simp only [complexGreatCircleMapKernel_real, integral_complex_ofReal, Complex.ofReal_re,
      greatCircleWeightMap, attentionWeightMap]
  rw [heq] at h
  simpa only [complexGreatCircleMapForce_real, integral_complex_ofReal, Complex.ofReal_re] using h

/-- Its actual derivative is itself globally analytic in the angle.
Source: arXiv:2601.21366v2, §6, the analytic first-variation field. -/
theorem analyticOnNhd_deriv_greatCircleWeightMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (u w : EucSpace d) :
    AnalyticOnNhd ℝ (deriv (greatCircleWeightMap B μ u w)) univ :=
  (analyticOnNhd_greatCircleWeightMap B μ u w).deriv

/-- The actual Dirac weight along the genuine circle is `exp(cos θ)`.
Source: arXiv:2601.21366v2, §5.2 and §6, the original exponential cosine
kernel. This verifies the higher-dimensional construction at actual source data. -/
example (θ : ℝ) : greatCircleWeightMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
    (Perspective.diracProb 2 (circlePoint 0)) ((basePoint 1 : SSphere 2) : EucSpace 2)
    secondAxis θ = Real.exp (Real.cos θ) := by
  change (∫ y, Real.exp (inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2)
    (y : EucSpace 2)) ∂Measure.dirac (circlePoint 0)) = _
  rw [integral_dirac, inner_circlePoint_circlePoint, sub_zero]

end Transformer.Perceptron
