module

public import Transformer.Perceptron.Section5_InteractionIntegral
public import Transformer.Perceptron.FirstVariation
public import Transformer.Perceptron.CircleDeriv
public import Mathlib.MeasureTheory.Function.L2Space

/-!
# The first variation in actual angular coordinates

ArXiv:2601.21366v2, §2.1--2.3, the first variation and
`eq: steady.state`, and §5.5, `rem:strictSOPD-perceptron`.
The angular first variation is the restriction of the actual
ambient first variation to the standard circle chart.

Its derivative is the scalar tangent component of the actual
Wasserstein gradient. Consequently stationarity on the actual
measure support makes every angular representative critical.
The derivative also splits into the justified interaction
integral and one half of the potential's angular derivative.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- The actual first variation restricted to the angular chart.
Source: arXiv:2601.21366v2, §2.1--2.3, the first variation,
and §5.5, the angular curvature in `rem:strictSOPD-perceptron`. -/
noncomputable def angularFirstVariation (β : ℝ) (φ : ℝ → ℝ) (ω : Idx 2 → ℝ)
    (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) : ℝ :=
  firstVariation β φ ω a μ (circlePoint θ : EucSpace 2)

/-- **The actual first variation of a zero-weight Dirac measure.**
Source: arXiv:2601.21366v2, §2.1--2.3, the first variation
and its exponential pair kernel. This closed form checks the
temperature factor and the interaction against an actual measure. -/
theorem angularFirstVariation_dirac_zero_weights (β θ φ : ℝ) :
    angularFirstVariation β 0 0 0 (Perspective.diracProb 2 (circlePoint φ)) θ =
      β⁻¹ * Real.exp (β * Real.cos (θ - φ)) := by
  change β⁻¹ * (∫ y : SSphere 2,
      Real.exp (β * inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) (y : EucSpace 2))
        ∂Measure.dirac (circlePoint φ)) + (2 : ℝ)⁻¹ *
      potential 0 0 0 (circlePoint θ : EucSpace 2) = _
  rw [integral_dirac, inner_circlePoint_circlePoint]
  simp [potential]

/-- **Its actual derivative is the tangent component of the actual
Wasserstein gradient.** Source: arXiv:2601.21366v2, §2.3,
the first-variation gradient and `eq: steady.state`.
The radial component drops out by orthogonality to `x'(θ)`. -/
theorem hasDerivAt_angularFirstVariation (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    HasDerivAt (angularFirstVariation β φ ω a μ)
      (inner (𝕜 := ℝ) (energyGrad β σ ω a μ (circlePoint θ : EucSpace 2)) (circleVel θ)) θ := by
  let G : EucSpace 2 :=
    (∫ y, Real.exp (β * inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) (y : EucSpace 2))
      • (y : EucSpace 2) ∂(μ : Measure (SSphere 2))) +
      ∑ j : Idx 2, (ω j * σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2))) • a j
  have hraw := (hasGradientAt_firstVariation hβ φ σ hφ ω a μ
    (circlePoint θ : EucSpace 2)).hasFDerivAt.comp_hasDerivAt θ (hasDerivAt_circlePoint θ)
  change HasDerivAt (angularFirstVariation β φ ω a μ)
    (inner (𝕜 := ℝ) G (circleVel θ)) θ at hraw
  apply hraw.congr_deriv
  rw [energyGrad_eq_proj]
  change inner (𝕜 := ℝ) G (circleVel θ) =
    inner (𝕜 := ℝ) (proj 2 (circlePoint θ : EucSpace 2) G) (circleVel θ)
  rw [proj, inner_sub_left, real_inner_smul_left, inner_circlePoint_circleVel,
    mul_zero, sub_zero]

/-- **The angular derivative as a function value.**
Source: arXiv:2601.21366v2, §2.3, the angular stationarity
condition corresponding to `eq: steady.state`. -/
theorem deriv_angularFirstVariation (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    deriv (angularFirstVariation β φ ω a μ) θ =
      inner (𝕜 := ℝ) (energyGrad β σ ω a μ (circlePoint θ : EucSpace 2)) (circleVel θ) :=
  (hasDerivAt_angularFirstVariation β hβ φ σ hφ ω a μ θ).deriv

/-- **The interaction tangent component equals the actual scalar
force integral.** Source: arXiv:2601.21366v2, §2.1--2.3,
the first-variation interaction gradient in angular coordinates.
The inner product commutes with the integrable vector-valued
interaction, and its real symmetry fixes the argument order. -/
theorem inner_interaction_integral_circleVel (β : ℝ)
    (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    inner (𝕜 := ℝ)
      (∫ y, Real.exp (β * inner (𝕜 := ℝ) (circlePoint θ : EucSpace 2) (y : EucSpace 2))
        • (y : EucSpace 2) ∂(μ : Measure (SSphere 2))) (circleVel θ) =
      ∫ y, circleInteractionForce β θ y ∂(μ : Measure (SSphere 2)) := by
  rw [real_inner_comm (circleVel θ), ← integral_inner
    (Perspective.integrable_expInner_smul 2 β μ (circlePoint θ : EucSpace 2))]
  apply integral_congr_ae
  exact ae_of_all _ fun y => by
    dsimp only
    rw [real_inner_smul_right, circleInteractionForce,
      real_inner_comm (y : EucSpace 2) (circlePoint θ : EucSpace 2),
      real_inner_comm (y : EucSpace 2) (circleVel θ)]

/-- **The first angular derivative splits into interaction and
potential.** Source: arXiv:2601.21366v2, §2.3, its first variation,
and §5.5, the two terms differentiated in `rem:strictSOPD-perceptron`.
The factor one half is the source's convention `φ'=2σ`. -/
theorem deriv_angularFirstVariation_eq_force (β : ℝ) (hβ : β ≠ 0) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    deriv (angularFirstVariation β φ ω a μ) θ =
      (∫ y, circleInteractionForce β θ y ∂(μ : Measure (SSphere 2))) +
        (1 / 2 : ℝ) * deriv (fun t => potential φ ω a (circlePoint t : EucSpace 2)) θ := by
  rw [deriv_angularFirstVariation β hβ φ σ hφ, energyGrad_eq_proj,
    proj, inner_sub_left, real_inner_smul_left, inner_circlePoint_circleVel,
    mul_zero, sub_zero, inner_add_left, inner_interaction_integral_circleVel]
  have hlin : inner (𝕜 := ℝ)
      (∑ j : Idx 2, (ω j * σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2))) • a j)
      (circleVel θ) = ∑ j : Idx 2, ω j *
        (σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
          inner (𝕜 := ℝ) (a j) (circleVel θ)) := by
    rw [sum_inner]
    apply Finset.sum_congr rfl
    intro j _
    rw [real_inner_smul_left]
    ring
  rw [hlin, deriv_potential_circlePoint φ σ hφ]
  ring

/-- **Stationary support points are critical in every angular
representative.** Source: arXiv:2601.21366v2, §2.3,
`eq: steady.state`, as used by `rem:strictSOPD-perceptron`. -/
theorem deriv_angularFirstVariation_zero_on_support (β : ℝ) (hβ : β ≠ 0)
    (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β σ ω a μ) (θ : ℝ)
    (hθ : circlePoint θ ∈ (μ : Measure (SSphere 2)).support) :
    deriv (angularFirstVariation β φ ω a μ) θ = 0 := by
  rw [deriv_angularFirstVariation β hβ φ σ hφ, hμ _ hθ, inner_zero_left]

/-- Every nontrivial input of the derivative, force-splitting, and
stationarity consequences is realized at `β=1`, `φ=σ=0`, zero
weights, and the actual Dirac probability at the first basis vector.
Its support contains that actual point, which is `circlePoint 0`. -/
example : (1 : ℝ) ≠ 0 ∧
    (∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
    IsStationary 1 (0 : ℝ → ℝ) (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2)
      (Perspective.diracProb 2 (circlePoint 0)) ∧
    circlePoint 0 ∈ (Perspective.diracProb 2 (circlePoint 0) : Measure (SSphere 2)).support := by
  refine ⟨one_ne_zero, fun s => by simpa using hasDerivAt_const s (0 : ℝ), ?_, ?_⟩
  · exact isStationary_diracProb_of_radial 1 0 0 0 (circlePoint 0) 0 (by simp)
  · apply (Measure.mem_support_iff_forall _).mpr
    intro U hU
    have hx : circlePoint 0 ∈ U := mem_of_mem_nhds hU
    change 0 < Measure.dirac (circlePoint 0) U
    simp [hx]

end Transformer.Perceptron
