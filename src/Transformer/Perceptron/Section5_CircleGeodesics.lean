/-
# Angular speeds along the actual spherical exponential map

ArXiv:2601.21366v2, §2.2, the exponential map defining the Monge
geodesic, and §5.2, `eq:Hess-bilinear.std`. The source identifies
a tangent vector on the circle with its scalar angular speed.
The orthonormal pair `x(θ),x'(θ)` makes this identification exact
for the actual Euclidean vector fields used in the formalization.

The spherical exponential of a tangent velocity is precisely
`x(θ+t s)`, including zero and negative speeds. Combining this
identity with the actual singleton-weight integration formula gives
the energy along every finite-support Monge geodesic as a finite
sum of pair kernels and perceptron potentials.
No Hessian is defined by that sum; the next calculus step must
differentiate the actual energy expression.
-/

import Transformer.Perceptron.Section5_CircleGeometry
import Transformer.Perceptron.Section5_AtomicIntegrals
import Transformer.Perceptron.GeodesicCurve
import Transformer.Perceptron.Kernel

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- **The circle point and unit tangent are an orthonormal frame.**
Source: arXiv:2601.21366v2, §5.2, `x(θ)=(cos θ,sin θ)` and
the scalar velocities in `eq:Hess-bilinear.std`. This decomposes
every actual two-dimensional vector, before imposing tangency. -/
theorem circle_frame_decomposition (θ : ℝ) (v : EucSpace 2) :
    v = inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) v • (circlePoint θ : EucSpace 2) +
      inner (𝕜 := ℝ) (circleVel θ) v • circleVel θ := by
  have hp : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) v =
      Real.cos θ * v 0 + Real.sin θ * v 1 := by
    rw [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_two]
    change v 0 * (circlePoint θ : EucSpace 2) 0 +
      v 1 * (circlePoint θ : EucSpace 2) 1 = _
    rw [circlePoint_coord_zero, circlePoint_coord_one]
    ring
  have hv : inner (𝕜 := ℝ) (circleVel θ) v =
      -Real.sin θ * v 0 + Real.cos θ * v 1 := by
    rw [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Fin.sum_univ_two]
    change v 0 * circleVel θ 0 + v 1 * circleVel θ 1 = _
    rw [circleVel_coord_zero, circleVel_coord_one]
    ring
  rw [hp, hv]
  ext i
  fin_cases i
  · change v 0 = (Real.cos θ * v 0 + Real.sin θ * v 1) *
        (circlePoint θ : EucSpace 2) 0 +
      (-Real.sin θ * v 0 + Real.cos θ * v 1) * circleVel θ 0
    rw [circlePoint_coord_zero, circleVel_coord_zero]
    linear_combination -(v 0) * Real.sin_sq_add_cos_sq θ
  · change v 1 = (Real.cos θ * v 0 + Real.sin θ * v 1) *
        (circlePoint θ : EucSpace 2) 1 +
      (-Real.sin θ * v 0 + Real.cos θ * v 1) * circleVel θ 1
    rw [circlePoint_coord_one, circleVel_coord_one]
    linear_combination -(v 1) * Real.sin_sq_add_cos_sq θ

/-- **A tangent vector equals its scalar angular speed times `x'(θ)`.**
Source: arXiv:2601.21366v2, §2.2, `eq:otto.tangent`, and §5.2,
the scalar tangent field in `eq:Hess-bilinear.std`. -/
theorem tangent_eq_circleVel_smul (θ : ℝ) (v : EucSpace 2)
    (hv : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) v = 0) :
    v = inner (𝕜 := ℝ) (circleVel θ) v • circleVel θ := by
  have h := circle_frame_decomposition θ v
  simpa only [hv, zero_smul, zero_add] using h

/-- **The angular speed squared is the actual tangent norm squared.**
Source: arXiv:2601.21366v2, §5.2 and §5.5, the `L²(μ)` norm
in `eq:Hess-bilinear.std` and `rem:strictSOPD-perceptron`. -/
theorem norm_sq_tangent_circle (θ : ℝ) (v : EucSpace 2)
    (hv : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) v = 0) :
    ‖v‖ ^ 2 = (inner (𝕜 := ℝ) (circleVel θ) v) ^ 2 := by
  conv_lhs => rw [tangent_eq_circleVel_smul θ v hv]
  rw [norm_smul, norm_circleVel, mul_one, Real.norm_eq_abs, sq_abs]

/-- The tangent hypotheses are jointly realized by the actual unit
angular velocity at every point of the nonempty circle. -/
example (θ : ℝ) : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) (circleVel θ) = 0 ∧
    ‖circleVel θ‖ = 1 := ⟨inner_circlePoint_circleVel θ, norm_circleVel θ⟩

/-- **The angular addition formula in the actual sphere chart.**
Source: arXiv:2601.21366v2, §2.2 and §5.2, the great circles
used to compute `eq:Hess-bilinear.std`. -/
theorem circlePoint_add_frame (θ r : ℝ) :
    (circlePoint (θ + r) : EucSpace 2) =
      Real.cos r • (circlePoint θ : EucSpace 2) + Real.sin r • circleVel θ := by
  ext i
  fin_cases i
  · change (circlePoint (θ + r) : EucSpace 2) 0 =
      Real.cos r * (circlePoint θ : EucSpace 2) 0 + Real.sin r * circleVel θ 0
    rw [circlePoint_coord_zero, circlePoint_coord_zero, circleVel_coord_zero, Real.cos_add]
    ring
  · change (circlePoint (θ + r) : EucSpace 2) 1 =
      Real.cos r * (circlePoint θ : EucSpace 2) 1 + Real.sin r * circleVel θ 1
    rw [circlePoint_coord_one, circlePoint_coord_one, circleVel_coord_one, Real.sin_add]
    ring

/-- **The actual spherical exponential follows the angular chart.**
Source: arXiv:2601.21366v2, §2.2, the exponential-map footnote,
and §5.2, `eq:Hess-bilinear.std`. Every real speed is included. -/
theorem sphereExp_circleVel (θ r : ℝ) :
    sphereExp (circlePoint θ : EucSpace 2) (r • circleVel θ) =
      (circlePoint (θ + r) : EucSpace 2) := by
  rw [sphereExp_smul, norm_circleVel, mul_one, div_one, circlePoint_add_frame]

/-- **A tangent velocity gives the chart shift by its angular speed.**
Source: arXiv:2601.21366v2, §2.2 and §5.2, the Monge geodesic
whose energy is differentiated in `eq:Hess-bilinear.std`. -/
theorem sphereExp_tangent_circle (θ : ℝ) (v : EucSpace 2)
    (hv : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) v = 0) (t : ℝ) :
    sphereExp (circlePoint θ : EucSpace 2) (t • v) =
      (circlePoint (θ + t * inner (𝕜 := ℝ) (circleVel θ) v) : EucSpace 2) := by
  conv_lhs => rw [tangent_eq_circleVel_smul θ v hv, smul_smul]
  exact sphereExp_circleVel θ _

/-- **The energy along every actual finite-support Monge geodesic.**
Source: arXiv:2601.21366v2, §5.2, the finite-sum form of the
energy differentiated in `eq:Hess-bilinear.std`. The angles are
representatives of the original support points and the speeds are
the actual tangent-vector components, not independent new inputs. -/
theorem energy_geodesic_finite_angles (β : ℝ) (φ : ℝ → ℝ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hfinite : (μ : Measure (SSphere 2)).support.Finite)
    (ξ : SSphere 2 → EucSpace 2)
    (hξ : ∀ z : SSphere 2, inner (𝕜 := ℝ) (z : EucSpace 2) (ξ z) = 0)
    (ν : ℝ → Perspective.ProbSphere 2) (hν : IsGeodesicFrom ξ μ ν)
    (θ : SSphere 2 → ℝ) (hθ : ∀ z : SSphere 2, circlePoint (θ z) = z) (t : ℝ) :
    energy β φ ω a (ν t) = (2 * β)⁻¹ *
      (∑ x ∈ hfinite.toFinset, ((μ : Measure (SSphere 2)) {x}).toReal *
        ∑ y ∈ hfinite.toFinset, ((μ : Measure (SSphere 2)) {y}).toReal *
          kernelK β (θ x - θ y + t *
            (inner (𝕜 := ℝ) (circleVel (θ x)) (ξ x) -
              inner (𝕜 := ℝ) (circleVel (θ y)) (ξ y)))) +
      (1 / 2 : ℝ) * (∑ x ∈ hfinite.toFinset, ((μ : Measure (SSphere 2)) {x}).toReal *
        potential φ ω a (circlePoint
          (θ x + t * inner (𝕜 := ℝ) (circleVel (θ x)) (ξ x)) : EucSpace 2)) := by
  obtain ⟨T, hTm, hT, hmap⟩ := hν t
  have hangle (z : SSphere 2) : T z = circlePoint
      (θ z + t * inner (𝕜 := ℝ) (circleVel (θ z)) (ξ z)) := by
    apply Subtype.ext
    rw [hT]
    have ht : inner (𝕜 := ℝ) (circlePoint (θ z) : EucSpace 2) (ξ z) = 0 := by
      rw [hθ z]
      exact hξ z
    have h := sphereExp_tangent_circle (θ z) (ξ z) ht t
    rwa [hθ z] at h
  rw [energy_finite_support_map β φ ω a μ (ν t) hfinite T hTm hmap]
  simp_rw [hangle, inner_circlePoint_circlePoint]
  congr 2
  apply Finset.sum_congr rfl
  intro x hx
  congr 1
  apply Finset.sum_congr rfl
  intro y hy
  congr 1
  unfold kernelK
  rw [show (θ x + t * inner (𝕜 := ℝ) (circleVel (θ x)) (ξ x)) -
      (θ y + t * inner (𝕜 := ℝ) (circleVel (θ y)) (ξ y)) =
      θ x - θ y + t * (inner (𝕜 := ℝ) (circleVel (θ x)) (ξ x) -
        inner (𝕜 := ℝ) (circleVel (θ y)) (ξ y)) by ring]

/-- The angle-representation and geodesic hypotheses are jointly
realized by the zero tangent field and constant Dirac curve. -/
example : ∃ θ : SSphere 2 → ℝ, (∀ z : SSphere 2, circlePoint (θ z) = z) ∧
    (∀ z : SSphere 2, inner (𝕜 := ℝ) (z : EucSpace 2) (0 : EucSpace 2) = 0) ∧
    IsGeodesicFrom (fun _ : SSphere 2 => (0 : EucSpace 2))
      (Perspective.diracProb 2 (basePoint 1)) (fun _ => Perspective.diracProb 2 (basePoint 1)) := by
  classical
  choose θ hθ using circlePoint_surjective
  exact ⟨θ, hθ, fun z => by simp, fun t =>
    ⟨id, measurable_id, fun z => by simp, Measure.map_id.symm⟩⟩

end Transformer.Perceptron
