module

public import Transformer.Perceptron.Section5_AtomicForce
public import Transformer.Perceptron.Section5_PotentialForce
public import Transformer.Perceptron.Section5_CircleGeodesics
public import Transformer.Perceptron.Section5_AngularFirstVariation

/-!
# Actual stationary quadratic two-atom configurations

ArXiv:2601.21366v2, §2.3, `eq: primitive.field` and
`eq: steady.state`, and §5.4, the hypotheses of `thm: bound`.
Two equal masses at the circle points with angles `r,-r` are
stationary for the linear activation when the single neuron at
the first axis has weight `-exp(β cos(2r))`.

The interaction force and actual potential force cancel at both
support points. The actual Wasserstein gradient is tangent to
the circle, so its vanishing scalar tangent component makes the
entire vector vanish. This gives genuine stationary close-atom
configurations for the first-order strengthening of the mass bound.
No SOPD condition is asserted for these configurations.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- **Zero actual angular force makes the actual spherical gradient zero.**
Source: arXiv:2601.21366v2, §2.3, `eq: steady.state`, in the
circle chart of §5.2. The gradient is already tangent; the actual
orthonormal frame identifies its remaining scalar component. -/
theorem energyGrad_circle_eq_zero_of_force_zero (β : ℝ) (hβ : β ≠ 0)
    (φ σ : ℝ → ℝ) (hφ : ∀ r : ℝ, HasDerivAt φ (2 * σ r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ)
    (hforce : (∫ z, circleInteractionForce β θ z ∂(μ : Measure (SSphere 2))) +
      angularPotentialForce φ ω a θ = 0) :
    energyGrad β σ ω a μ (circlePoint θ : EucSpace 2) = 0 := by
  let g := energyGrad β σ ω a μ (circlePoint θ : EucSpace 2)
  have hradial : inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) g = 0 := by
    dsimp [g]
    rw [energyGrad_eq_proj]
    exact inner_proj_eq_zero (norm_circlePoint_eq_one θ) _
  have htangent : inner (𝕜 := ℝ) g (circleVel θ) = 0 := by
    have h1 := deriv_angularFirstVariation β hβ φ σ hφ ω a μ θ
    have h2 := deriv_angularFirstVariation_eq_force β hβ φ σ hφ ω a μ θ
    change (∫ z, circleInteractionForce β θ z ∂(μ : Measure (SSphere 2))) +
      (1 / 2 : ℝ) * deriv (fun t => potential φ ω a (circlePoint t : EucSpace 2)) θ = 0 at hforce
    change inner (𝕜 := ℝ) (energyGrad β σ ω a μ (circlePoint θ : EucSpace 2)) (circleVel θ) = 0
    linarith
  have hframe := circle_frame_decomposition θ g
  rw [hradial, real_inner_comm g (circleVel θ), htangent, zero_smul, zero_smul, zero_add] at hframe
  exact hframe

/-- The actual one-neuron quadratic potential is `-W cos²θ`.
Source: arXiv:2601.21366v2, §2.3, `eq: primitive.field`.
This checks the potential whose force is used in the construction. -/
theorem potential_quadratic_axis_circle (W θ : ℝ) :
    potential (fun s : ℝ => s ^ 2) (Pi.single 0 (-W))
      (Pi.single 0 ((basePoint 1 : SSphere 2) : EucSpace 2)) (circlePoint θ : EucSpace 2) =
      -W * Real.cos θ ^ 2 := by
  rw [potential, Fin.sum_univ_two]
  simp only [Pi.single_eq_same, Pi.single_eq_of_ne (show (1 : Idx 2) ≠ 0 by decide),
    inner_zero_left, zero_mul, add_zero, inner_basePoint_circlePoint]

/-- **The actual quadratic pin has angular force `W cos θ sin θ`.**
Source: arXiv:2601.21366v2, §2.3, `eq: primitive.field`, and
§5.4, the force corresponding to `eq:T-field-bound`.
The primitive is `s²`, activation is `id`, and the weight is `-W`. -/
theorem angularPotentialForce_quadratic_pin (W θ : ℝ) :
    angularPotentialForce (fun s : ℝ => s ^ 2) (Pi.single 0 (-W))
      (Pi.single 0 ((basePoint 1 : SSphere 2) : EucSpace 2)) θ =
      W * Real.cos θ * Real.sin θ := by
  have hφ : ∀ r : ℝ, HasDerivAt (fun s : ℝ => s ^ 2) (2 * id r) r :=
    fun r => by simpa using hasDerivAt_pow 2 r
  have ht : inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2) (circleVel θ) =
      -Real.sin θ := by
    rw [inner_circleVel, real_inner_self_eq_norm_sq, norm_basePoint_one,
      inner_basePoint_secondAxis]
    ring
  rw [angularPotentialForce_eq_sum _ id hφ, Fin.sum_univ_two]
  simp only [Pi.single_eq_same, Pi.single_eq_of_ne (show (1 : Idx 2) ≠ 0 by decide),
    id_eq, inner_zero_left, zero_mul, add_zero, inner_basePoint_circlePoint, ht]
  ring

/-- **The actual interaction force of the symmetric pair at the first atom.**
Source: arXiv:2601.21366v2, §2.3, the first-variation interaction
gradient. Self-interaction is radial and the other mass is one half. -/
theorem integral_force_twoAtom_symmetric (β r : ℝ) :
    (∫ z, circleInteractionForce β r z ∂(twoAtomProb (circlePoint r) (circlePoint (-r)) :
      Measure (SSphere 2))) = -(1 / 2 : ℝ) * Real.exp (β * Real.cos (2 * r)) * Real.sin (2 * r) := by
  rw [integral_twoAtomProb]
  simp only [smul_eq_mul, circleInteractionForce_self, mul_zero, zero_add]
  rw [circleInteractionForce, inner_circlePoint_circlePoint, inner_circlePoint_circleVel_sub,
    show -r - r = -(2 * r) by ring, Real.cos_neg, Real.sin_neg]
  ring

/-- **The corresponding actual interaction force at the second atom.**
Source: arXiv:2601.21366v2, §2.3, the same first-variation
gradient. Its sign is reversed by the reflection symmetry. -/
theorem integral_force_twoAtom_symmetric_neg (β r : ℝ) :
    (∫ z, circleInteractionForce β (-r) z ∂(twoAtomProb (circlePoint r) (circlePoint (-r)) :
      Measure (SSphere 2))) = (1 / 2 : ℝ) * Real.exp (β * Real.cos (2 * r)) * Real.sin (2 * r) := by
  rw [integral_twoAtomProb]
  simp only [smul_eq_mul, circleInteractionForce_self, mul_zero, add_zero]
  rw [circleInteractionForce, inner_circlePoint_circlePoint, inner_circlePoint_circleVel_sub,
    show r - -r = 2 * r by ring]
  ring

/-- Reflection reverses the actual interaction force of this pair.
Source: arXiv:2601.21366v2, §2.3, the symmetric interaction
in the first-variation gradient. The two integrals use the same measure. -/
theorem integral_force_twoAtom_reflection (β r : ℝ) :
    (∫ z, circleInteractionForce β (-r) z ∂(twoAtomProb (circlePoint r) (circlePoint (-r)) :
      Measure (SSphere 2))) =
      -(∫ z, circleInteractionForce β r z ∂(twoAtomProb (circlePoint r) (circlePoint (-r)) :
        Measure (SSphere 2))) := by
  rw [integral_force_twoAtom_symmetric_neg, integral_force_twoAtom_symmetric]
  ring

/-- **The symmetric quadratic pair is an actual stationary measure.**
Source: arXiv:2601.21366v2, §2.3, `eq: steady.state`, and §5.4,
the stationary hypotheses of `thm: bound`. This construction uses
weight `-exp(β cos(2r))`, not half that value, because `φ'=2σ`
makes the potential's contribution one half of its derivative. -/
theorem isStationary_twoAtom_quadratic_pin (β : ℝ) (hβ : β ≠ 0) (r : ℝ) :
    IsStationary β id (Pi.single 0 (-Real.exp (β * Real.cos (2 * r))))
      (Pi.single 0 ((basePoint 1 : SSphere 2) : EucSpace 2))
      (twoAtomProb (circlePoint r) (circlePoint (-r))) := by
  have hφ : ∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id s) s :=
    fun s => by simpa using hasDerivAt_pow 2 s
  intro x hx
  rcases mem_support_twoAtomProb hx with rfl | rfl
  · apply energyGrad_circle_eq_zero_of_force_zero β hβ (fun s : ℝ => s ^ 2) id hφ
    rw [integral_force_twoAtom_symmetric, angularPotentialForce_quadratic_pin, Real.sin_two_mul]
    ring
  · apply energyGrad_circle_eq_zero_of_force_zero β hβ (fun s : ℝ => s ^ 2) id hφ
    rw [integral_force_twoAtom_symmetric_neg, angularPotentialForce_quadratic_pin,
      Real.cos_neg, Real.sin_neg, Real.sin_two_mul]
    ring

/-- All nonzero-temperature and primitive hypotheses above are
jointly realized at `β=1` by the actual quadratic primitive.
The zero-force input is realized on a nonempty sphere by a Dirac
mass at angle zero with zero perceptron weights. -/
example : (1 : ℝ) ≠ 0 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * id r) r) ∧
    (∫ z, circleInteractionForce 1 0 z ∂(Perspective.diracProb 2 (circlePoint 0) :
      Measure (SSphere 2))) + angularPotentialForce (fun t : ℝ => t ^ 2) 0 0 0 = 0 := by
  refine ⟨one_ne_zero, fun r => by simpa using hasDerivAt_pow 2 r, ?_⟩
  have hI : (∫ z, circleInteractionForce 1 0 z ∂(Perspective.diracProb 2 (circlePoint 0) :
      Measure (SSphere 2))) = 0 := by
    change (∫ z, circleInteractionForce 1 0 z ∂Measure.dirac (circlePoint 0)) = 0
    rw [integral_dirac, circleInteractionForce_self]
  rw [hI, zero_add, angularPotentialForce_eq_sum _ id
    (fun r => by simpa using hasDerivAt_pow 2 r)]
  simp

end Transformer.Perceptron
