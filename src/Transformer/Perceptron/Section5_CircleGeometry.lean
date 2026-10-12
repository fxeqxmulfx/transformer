/-
# The angular chart and orthonormal tangent frame on the circle

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear`, and §5.5,
`rem:strictSOPD-perceptron`. The source writes the circle as
`x(θ)=(cos θ,sin θ)` and its unit tangent as `x'(θ)`.
These statements connect that chart to the actual Euclidean sphere
used by the measure and geodesic definitions.

Every sphere point has a representative in `[-π,π]`, obtained from
the argument of its two coordinates as a complex number of norm one.
The chart is continuous, its tangent is orthogonal and has norm one,
and the pair interaction is precisely `exp(β cos(θ-φ))`.
No analyticity or atomicity claim is included in a definition.
The unit tangent frame also identifies each tangent vector with
its scalar angular speed for the subsequent Hessian calculation.
-/

import Transformer.Perceptron.Atoms
import Mathlib.Analysis.SpecialFunctions.Complex.Arg

open scoped BigOperators
open Real

namespace Transformer.Perceptron

/-- The first coordinate of the source's angular chart.
Source: arXiv:2601.21366v2, §5.2, `x(θ)=(cos θ,sin θ)`. -/
@[simp] theorem circlePoint_coord_zero (θ : ℝ) :
    (circlePoint θ : EucSpace 2) 0 = Real.cos θ := by
  simp [circlePoint, greatCircle, basePoint, secondAxis]

/-- The second coordinate of the source's angular chart.
Source: arXiv:2601.21366v2, §5.2, the same parametrization. -/
@[simp] theorem circlePoint_coord_one (θ : ℝ) :
    (circlePoint θ : EucSpace 2) 1 = Real.sin θ := by
  simp [circlePoint, greatCircle, basePoint, secondAxis]

/-- The first coordinate of the unit angular tangent.
Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`, `x'(θ)`. -/
@[simp] theorem circleVel_coord_zero (θ : ℝ) :
    circleVel θ 0 = -Real.sin θ := by
  simp [circleVel, basePoint, secondAxis]

/-- The second coordinate of the unit angular tangent.
Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`, `x'(θ)`. -/
@[simp] theorem circleVel_coord_one (θ : ℝ) :
    circleVel θ 1 = Real.cos θ := by
  simp [circleVel, basePoint, secondAxis]

/-- **The angular chart has unit tangent at every angle.**
Source: arXiv:2601.21366v2, §5.2, the scalar tangent velocity
in `eq:Hess-bilinear`, and `rem:strictSOPD-perceptron`.
The norm is the actual Euclidean norm, not the sup norm on functions. -/
theorem norm_circleVel (θ : ℝ) : ‖circleVel θ‖ = 1 := by
  have hsq : ‖circleVel θ‖ ^ 2 = 1 := by
    rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two,
      circleVel_coord_zero, circleVel_coord_one, neg_sq]
    exact Real.sin_sq_add_cos_sq θ
  nlinarith [norm_nonneg (circleVel θ)]

/-- **The angular tangent is orthogonal to its sphere point.**
Source: arXiv:2601.21366v2, §2.2 and §5.2, the spherical tangent space
in `eq:otto.tangent` and `eq:Hess-bilinear`. -/
theorem inner_circlePoint_circleVel (θ : ℝ) :
    inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) (circleVel θ) = 0 := by
  have hfirst : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2)
      ((basePoint 1 : SSphere 2) : EucSpace 2) = Real.cos θ := by
    exact (real_inner_comm (circlePoint θ : EucSpace 2)
      ((basePoint 1 : SSphere 2) : EucSpace 2)).symm.trans (inner_basePoint_circlePoint θ)
  have hsecond : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) secondAxis = Real.sin θ := by
    simp [circlePoint, greatCircle, inner_add_left, real_inner_smul_left]
  rw [inner_circleVel, hfirst, hsecond]
  ring

/-- **The cosine pair kernel in the angular chart is the actual
spherical inner product.** Source: arXiv:2601.21366v2, §5.2,
`eq:Hess-bilinear`, and §5.5, `K_β(θ)=exp(β cos θ)`. -/
theorem inner_circlePoint_circlePoint (θ φ : ℝ) :
    inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) (circlePoint φ : EucSpace 2) =
      Real.cos (θ - φ) := by
  have hfirst : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2)
      ((basePoint 1 : SSphere 2) : EucSpace 2) = Real.cos θ := by
    exact (real_inner_comm (circlePoint θ : EucSpace 2)
      ((basePoint 1 : SSphere 2) : EucSpace 2)).symm.trans (inner_basePoint_circlePoint θ)
  have hsecond : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) secondAxis = Real.sin θ := by
    simp [circlePoint, greatCircle, inner_add_left, real_inner_smul_left]
  rw [inner_circlePoint, hfirst, hsecond, Real.cos_sub]
  ring

/-- **The scalar tangent component of a second circle point.**
Source: arXiv:2601.21366v2, §5.2, the first angular derivative
of the pair kernel in `eq:Hess-bilinear`.
The order of the angles fixes the sign of the sine. -/
theorem inner_circlePoint_circleVel_sub (φ θ : ℝ) :
    inner (𝕜 := ℝ) (circlePoint φ : EucSpace 2) (circleVel θ) = Real.sin (φ - θ) := by
  have hfirst : inner (𝕜 := ℝ) (circlePoint φ : EucSpace 2)
      ((basePoint 1 : SSphere 2) : EucSpace 2) = Real.cos φ := by
    exact (real_inner_comm (circlePoint φ : EucSpace 2)
      ((basePoint 1 : SSphere 2) : EucSpace 2)).symm.trans (inner_basePoint_circlePoint φ)
  have hsecond : inner (𝕜 := ℝ) (circlePoint φ : EucSpace 2) secondAxis = Real.sin φ := by
    simp [circlePoint, greatCircle, inner_add_left, real_inner_smul_left]
  rw [inner_circleVel, hfirst, hsecond, Real.sin_sub]
  ring

/-- **The chart is continuous as a map into the actual sphere.**
Source: arXiv:2601.21366v2, §5.2, the angular parametrization
used by `eq:Hess-bilinear`. Its ambient differentiability was already
proved in `hasDerivAt_circlePoint`. -/
theorem continuous_circlePoint : Continuous circlePoint := by
  apply Continuous.subtype_mk
  exact continuous_iff_continuousAt.mpr fun θ => (hasDerivAt_circlePoint θ).continuousAt

/-- **Every sphere point has an angle in the compact interval `[-π,π]`.**
Source: arXiv:2601.21366v2, §5.2, the identification of `𝕊¹`
with angular coordinates. The interval chosen here differs from
`[0,2π)` only by a periodic representative; its closedness is useful
for the compactness argument for stationary support points.
The angle is constructed from the two coordinates, not assumed. -/
theorem exists_circlePoint_eq (x : SSphere 2) :
    ∃ θ : ℝ, θ ∈ Set.Icc (-π) π ∧ circlePoint θ = x := by
  let z : ℂ := ⟨(x : EucSpace 2) 0, (x : EucSpace 2) 1⟩
  have hx : ‖(x : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hxsq : ((x : EucSpace 2) 0) ^ 2 + ((x : EucSpace 2) 1) ^ 2 = 1 := by
    have h := EuclideanSpace.real_norm_sq_eq (x : EucSpace 2)
    rw [hx, Fin.sum_univ_two] at h
    simpa only [one_pow] using h.symm
  have hz : ‖z‖ = 1 := by
    rw [Complex.norm_def, Complex.normSq_mk]
    have heq : (x : EucSpace 2) 0 * (x : EucSpace 2) 0 +
        (x : EucSpace 2) 1 * (x : EucSpace 2) 1 = 1 := by nlinarith
    rw [heq, Real.sqrt_one]
  have hz0 : z ≠ 0 := by
    intro h
    rw [h, norm_zero] at hz
    norm_num at hz
  have hc : Real.cos z.arg = (x : EucSpace 2) 0 := by
    simpa only [hz, div_one] using Complex.cos_arg hz0
  have hs : Real.sin z.arg = (x : EucSpace 2) 1 := by
    simpa only [hz, div_one] using Complex.sin_arg z
  refine ⟨z.arg, ⟨(Complex.arg_mem_Ioc z).1.le, (Complex.arg_mem_Ioc z).2⟩, ?_⟩
  apply Subtype.ext
  ext i
  fin_cases i
  · exact (circlePoint_coord_zero _).trans hc
  · exact (circlePoint_coord_one _).trans hs

/-- **The circle parametrization is surjective.**
Source: arXiv:2601.21366v2, §5.2, the angular treatment of
arbitrary measures on `𝕊¹`. This follows from an actual compact
representative for every point, rather than a restriction on support. -/
theorem circlePoint_surjective : Function.Surjective circlePoint := by
  intro x
  obtain ⟨θ, -, hθ⟩ := exists_circlePoint_eq x
  exact ⟨θ, hθ⟩

/-- A representative witness with a nonempty circle: the first
standard unit vector is the chart point at angle zero. -/
example : circlePoint 0 = (basePoint 1 : SSphere 2) ∧
    (0 : ℝ) ∈ Set.Icc (-π) π := by
  refine ⟨?_, ⟨by linarith [Real.pi_pos], Real.pi_pos.le⟩⟩
  apply Subtype.ext
  simp [circlePoint, greatCircle]

end Transformer.Perceptron
