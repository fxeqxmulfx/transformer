/-
# Strict lower bound for the actual finite-support geodesic Hessian

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`, and §5.5,
`rem:strictSOPD-perceptron` and `eq:strictSOPD-sufficient`.
The angular chart and finite-support integral formulas identify the
actual energy along each Monge geodesic with the finite angular sum.
Its twice-differentiated value is therefore the actual second
derivative of that energy curve.

The exact weighted pair-square identity and the sharp kernel bound
give the source's Hessian lower bound with coefficient `κ`.
The potential regularity is supplied by `φ'=2σ` and the actual
derivative of `σ`; no continuity of `σ'` is added.
Stationarity is not needed for this Hessian estimate itself.
It is needed earlier to deduce finite support from strict curvature,
and later for the source's definition of a strict SOPD critical point.
-/

import Transformer.Perceptron.Section5_CircleGeodesics
import Transformer.Perceptron.Section5_FiniteVariation
import Transformer.Perceptron.Section5_WeightedVariance
import Transformer.Perceptron.Section5_FirstVariationCurvature

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- **The actual geodesic second derivative is the source's finite Hessian.**
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`.
Every Monge geodesic is included and the weights are the actual
singleton masses of the starting probability measure. -/
theorem hasDerivAt_deriv_energy_geodesic_finite_angles (β : ℝ) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r : ℝ, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hfinite : (μ : Measure (SSphere 2)).support.Finite)
    (ξ : SSphere 2 → EucSpace 2)
    (hξ : ∀ z : SSphere 2, inner (𝕜 := ℝ) (z : EucSpace 2) (ξ z) = 0)
    (ν : ℝ → Perspective.ProbSphere 2) (hν : IsGeodesicFrom ξ μ ν)
    (θ : SSphere 2 → ℝ) (hθ : ∀ z : SSphere 2, circlePoint (θ z) = z) :
    let F := hfinite.toFinset
    let m := fun z : SSphere 2 => ((μ : Measure (SSphere 2)) {z}).toReal
    let s := fun z : SSphere 2 => inner (𝕜 := ℝ) (circleVel (θ z)) (ξ z)
    let V := fun r : ℝ => potential φ ω a (circlePoint r : EucSpace 2)
    HasDerivAt (deriv (fun t => energy β φ ω a (ν t)))
      ((2 * β)⁻¹ * (∑ i ∈ F, ∑ j ∈ F,
        m i * m j * kernelK2 β (θ i - θ j) * (s i - s j) ^ 2) +
        (1 / 2 : ℝ) * (∑ i ∈ F, m i * deriv^[2] V (θ i) * s i ^ 2)) 0 := by
  intro F m s V
  have hV : Differentiable ℝ V := fun r =>
    (hasDerivAt_potential_circlePoint φ σ hφ ω a r).differentiableAt
  have hV' : Differentiable ℝ (deriv V) := fun r =>
    differentiableAt_angular_potential_derivative φ σ σ' hφ hσ ω a r
  have hE : (fun t => energy β φ ω a (ν t)) = finiteAngularEnergy β V F m θ s := by
    funext t
    exact energy_geodesic_finite_angles β φ ω a μ hfinite ξ hξ ν hν θ hθ t
  rw [hE]
  exact hasDerivAt_deriv_finiteAngularEnergy β V hV hV' F m θ s

/-- **The actual second variation exists on finite support.**
Source: arXiv:2601.21366v2, §2.2, the well-defined Hessian in
`eq:2order`, and §5.2, `eq:Hess-bilinear.std`. This verifies
existence along every Monge geodesic, including zero velocities,
before the lower-bound theorem quantifies over derivative values. -/
theorem exists_secondDeriv_energy_geodesic_finite (β : ℝ) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r : ℝ, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hfinite : (μ : Measure (SSphere 2)).support.Finite)
    (ξ : SSphere 2 → EucSpace 2)
    (hξ : ∀ z : SSphere 2, inner (𝕜 := ℝ) (z : EucSpace 2) (ξ z) = 0)
    (ν : ℝ → Perspective.ProbSphere 2) (hν : IsGeodesicFrom ξ μ ν) :
    ∃ H : ℝ, HasDerivAt (deriv (fun t => energy β φ ω a (ν t))) H 0 := by
  classical
  choose θ hθ using circlePoint_surjective
  exact ⟨_, hasDerivAt_deriv_energy_geodesic_finite_angles β φ σ σ' hφ hσ
    ω a μ hfinite ξ hξ ν hν θ hθ⟩

/-- **The curvature condition bounds the actual geodesic Hessian below.**
Source: arXiv:2601.21366v2, §5.5, the Hessian inequality in
`rem:strictSOPD-perceptron` and `eq:strictSOPD-sufficient`.
The result takes a finite-support measure, any actual tangent field,
and any actual second-derivative value along its Monge geodesic. -/
theorem geodesic_hessian_ge_of_finite_support (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r : ℝ, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hfinite : (μ : Measure (SSphere 2)).support.Finite) (κ : ℝ)
    (hcurv : ∀ r : ℝ, circlePoint r ∈ (μ : Measure (SSphere 2)).support →
      2 * Real.exp β + 2 * κ ≤
        deriv^[2] (fun t => potential φ ω a (circlePoint t : EucSpace 2)) r)
    (ξ : SSphere 2 → EucSpace 2)
    (hξ : ∀ z : SSphere 2, inner (𝕜 := ℝ) (z : EucSpace 2) (ξ z) = 0)
    (ν : ℝ → Perspective.ProbSphere 2) (hν : IsGeodesicFrom ξ μ ν)
    (H : ℝ) (hH : HasDerivAt (deriv (fun t => energy β φ ω a (ν t))) H 0) :
    κ * ∫ z, ‖ξ z‖ ^ 2 ∂(μ : Measure (SSphere 2)) ≤ H := by
  classical
  choose θ hθ using circlePoint_surjective
  let F := hfinite.toFinset
  let m := fun z : SSphere 2 => ((μ : Measure (SSphere 2)) {z}).toReal
  let s := fun z : SSphere 2 => inner (𝕜 := ℝ) (circleVel (θ z)) (ξ z)
  let V := fun r : ℝ => potential φ ω a (circlePoint r : EucSpace 2)
  have hD := hasDerivAt_deriv_energy_geodesic_finite_angles β φ σ σ' hφ hσ
    ω a μ hfinite ξ hξ ν hν θ hθ
  have hvalue := hD.unique hH
  have hnorm : (∫ z, ‖ξ z‖ ^ 2 ∂(μ : Measure (SSphere 2))) =
      ∑ z ∈ F, m z * s z ^ 2 := by
    rw [integral_finite_support μ hfinite]
    apply Finset.sum_congr rfl
    intro z hz
    rw [smul_eq_mul]
    congr 1
    exact norm_sq_tangent_circle (θ z) (ξ z) (by rw [hθ z]; exact hξ z)
  rw [hnorm, ← hvalue]
  apply weighted_hessian_lower F m s (fun i j => kernelK2 β (θ i - θ j))
    (fun i => deriv^[2] V (θ i)) β κ hβ
  · exact fun z hz => singleton_mass_toReal_nonneg μ z
  · exact sum_singleton_mass_toReal μ hfinite
  · exact fun i hi j hj => neg_le_of_abs_le (abs_kernelK2_le β hβ (θ i - θ j))
  · intro z hz
    apply hcurv (θ z)
    rw [hθ z]
    exact hfinite.mem_toFinset.mp hz

/-- The complete regularity, curvature, finite-support and geodesic
inputs are jointly realized by the actual quadratic Dirac witness
and the zero tangent field. Its actual second derivative is zero. -/
example : let μ := Perspective.diracProb 2 (basePoint 1)
    let ω := Pi.single 0 (-4 : ℝ)
    let a := Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2))
    (0 : ℝ) < 1 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) r) r) ∧
    (μ : Measure (SSphere 2)).support.Finite ∧
    (∀ r : ℝ, circlePoint r ∈ (μ : Measure (SSphere 2)).support →
      2 * Real.exp 1 + 2 * 1 ≤ deriv^[2]
        (fun t => potential (fun u : ℝ => u ^ 2) ω a (circlePoint t : EucSpace 2)) r) ∧
    (∀ z : SSphere 2, inner (𝕜 := ℝ) (z : EucSpace 2) (0 : EucSpace 2) = 0) ∧
    IsGeodesicFrom (fun _ : SSphere 2 => (0 : EucSpace 2)) μ (fun _ => μ) ∧
    HasDerivAt (deriv (fun _ : ℝ => energy 1 (fun u : ℝ => u ^ 2) ω a μ)) 0 0 ∧
    ∃ θ : SSphere 2 → ℝ, ∀ z : SSphere 2, circlePoint (θ z) = z := by
  classical
  intro μ ω a
  obtain ⟨hβ, hφ, hσ, -, -, hcurv⟩ := strict_curvature_quadratic_dirac_witness
  refine ⟨hβ, hφ, hσ, ?_, hcurv, fun z => by simp, ?_, ?_, circlePoint_surjective.hasRightInverse⟩
  · exact finite_support_of_finitely_atomic μ (isFinitelyAtomic_diracProb (basePoint 1))
  · exact fun t => ⟨id, measurable_id, fun z => by simp, Measure.map_id.symm⟩
  · have hd : deriv (fun _ : ℝ => energy 1 (fun u : ℝ => u ^ 2) ω a μ) =
        fun _ => (0 : ℝ) := by
      funext t
      exact (hasDerivAt_const t _).deriv
    rw [hd]
    exact hasDerivAt_const 0 0

end Transformer.Perceptron
