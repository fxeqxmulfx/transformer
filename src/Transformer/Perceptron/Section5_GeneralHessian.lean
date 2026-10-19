module

public import Transformer.Perceptron.Section5_InteractionVariation
public import Transformer.Perceptron.Section5_PotentialVariation
public import Transformer.Perspective.Section2_EnergyConvex

/-!
# The actual circle Hessian for arbitrary probability measures

ArXiv:2601.21366v2, §2.2, `eq:strict2order`, and §5.2,
`eq:Hess-bilinear.std`. The interaction and potential integrals
have now both been differentiated twice under domination.
Pulling the actual energy back along the actual Monge transport
therefore proves existence and the value of its second derivative
for any probability measure on the circle.

`circlePairCurvature` is `K_β''/β`, and
`circlePotentialCurvature` is one half of the actual potential's
angular second derivative. Thus the factors below are the source's
factors expressed in continuous sphere coordinates. A strict SOPD
point must satisfy the resulting inequality along every continuous
speed whose tangent field is a smooth spherical gradient.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- The actual transported energy integrates against the original
measure and its product. Source: arXiv:2601.21366v2, §2.2--2.3,
the Monge transport and energy underlying `eq:Hess-bilinear.std`. -/
theorem energy_circleRotation_eq (β : ℝ) (φ : ℝ → ℝ) (hφ : Continuous φ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    energy β φ ω a (μ.map (circleRotation s t)) =
      (2 * β)⁻¹ * (∫ p, rotatedPairKernel β s t p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
      (1 / 2 : ℝ) * (∫ x, rotatedPotential φ ω a s t x ∂(μ : Measure (SSphere 2))) := by
  have hT := (continuous_circleRotation s hs t).measurable
  unfold energy
  rw [Perspective.interactionEnergy_eq_integral_prod]
  simp only [ProbabilityMeasure.toMeasure_map]
  rw [Measure.map_prod_map _ _ hT hT,
    integral_map (hT.prodMap hT).aemeasurable
      (Perspective.continuous_expInner_prod β).aestronglyMeasurable,
    integral_map (f := fun x : SSphere 2 => potential φ ω a (x : EucSpace 2)) hT.aemeasurable
      ((continuous_potential hφ ω a).comp continuous_subtype_val).aestronglyMeasurable]
  simp only [rotatedPairKernel, rotatedPotential, one_div]
  rfl

/-- The actual second derivative of the full energy is the source's
Hessian formula, for arbitrary support. Source: arXiv:2601.21366v2,
§5.2, `eq:Hess-bilinear.std`. A continuous actual activation
derivative suffices for domination; real analyticity is stronger. -/
theorem hasDerivAt_deriv_energy_circleRotation (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (s : SSphere 2 → ℝ) (hs : Continuous s) :
    HasDerivAt (deriv (fun t => energy β φ ω a (μ.map (circleRotation s t))))
      ((1 / 2 : ℝ) * (∫ p, circlePairCurvature β p.1 p.2 * (s p.1 - s p.2) ^ 2
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
        (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2
          ∂(μ : Measure (SSphere 2)))) 0 := by
  let I : ℝ → ℝ := fun t => ∫ p, rotatedPairKernel β s t p
    ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))
  let P : ℝ → ℝ := fun t => ∫ x, rotatedPotential φ ω a s t x ∂(μ : Measure (SSphere 2))
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  have hI (t : ℝ) := hasDerivAt_integral_rotatedPairKernel β hβ μ s hs t
  have hP (t : ℝ) := hasDerivAt_integral_rotatedPotential φ σ hφ hσc ω a μ s hs t
  have hE : (fun t => energy β φ ω a (μ.map (circleRotation s t))) =
      fun t => (2 * β)⁻¹ * I t + (1 / 2 : ℝ) * P t := by
    funext t
    exact energy_circleRotation_eq β φ (continuous_of_hasDerivAt hφ) ω a μ s hs t
  have hfirst : deriv (fun t => energy β φ ω a (μ.map (circleRotation s t))) =
      fun t => (2 * β)⁻¹ * deriv I t + (1 / 2 : ℝ) * deriv P t := by
    rw [hE]
    funext t
    exact (((hI t).differentiableAt.hasDerivAt.const_mul ((2 * β)⁻¹)).fun_add
      ((hP t).differentiableAt.hasDerivAt.const_mul (1 / 2 : ℝ))).deriv
  rw [hfirst]
  have hDI := (hasDerivAt_deriv_integral_rotatedPairKernel β hβ μ s hs 0).const_mul ((2 * β)⁻¹)
  have hDP := (hasDerivAt_deriv_integral_rotatedPotential φ σ σ' hφ hσ hσ' ω a μ s hs 0).const_mul
    (1 / 2 : ℝ)
  apply (hDI.fun_add hDP).congr_deriv
  simp only [rotatedPairCurvature_zero, rotatedPotentialCurvature_zero,
    mul_assoc, integral_const_mul]
  field_simp

/-- Strict SOPD gives a uniform positive lower bound for the actual
angular Hessian on smooth gradient directions. Source:
arXiv:2601.21366v2, §2.2, `eq:strict2order`, and §5.2,
`eq:Hess-bilinear.std`. Existence of this second derivative has
already been proved without assuming finite support. -/
theorem strict_circle_hessian_coercive (β : ℝ) (hβ : 0 < β)
    (φ σ σ' : ℝ → ℝ) (hφ : ∀ r, HasDerivAt φ (2 * σ r) r)
    (hσ : ∀ r, HasDerivAt σ (σ' r) r) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStrictSOPD β φ σ ω a μ) :
    ∃ κ : ℝ, 0 < κ ∧ ∀ s : SSphere 2 → ℝ, Continuous s →
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) →
      κ * (∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))) ≤
        (1 / 2 : ℝ) * (∫ p, circlePairCurvature β p.1 p.2 * (s p.1 - s p.2) ^ 2
          ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
        (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2
          ∂(μ : Measure (SSphere 2))) := by
  obtain ⟨κ, hκ, hH⟩ := hμ.2
  refine ⟨κ, hκ, fun s hs hgrad => ?_⟩
  have h := hH _ hgrad (fun t => μ.map (circleRotation s t))
    (isGeodesicFrom_circleRotation μ s hs) _
    (hasDerivAt_deriv_energy_circleRotation β hβ φ σ σ' hφ hσ hσ' ω a μ s hs)
  have hnorm (x : SSphere 2) : ‖s x • circleTangent (x : EucSpace 2)‖ ^ 2 = s x ^ 2 := by
    rw [norm_smul, norm_circleTangent, mem_sphere_zero_iff_norm.mp x.2,
      mul_one, Real.norm_eq_abs, sq_abs]
  simpa only [hnorm] using h

/-- The pinned Dirac has actual second variation `1/2` along unit
angular rotation. Source: arXiv:2601.21366v2, §2.2 and §5.2,
`eq:Hess-bilinear.std`. Its self-interaction is constant, while
its actual potential is `-cos θ`. This nonzero value checks the
normalization of the full formula on an actual probability measure. -/
theorem hasDerivAt_deriv_energy_pin_rotation (β : ℝ) (hβ : 0 < β) :
    HasDerivAt (deriv (fun t => energy β (fun r => r) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      ((Perspective.diracProb 2 (basePoint 1)).map (circleRotation (fun _ => 1) t))))
      (1 / 2 : ℝ) 0 := by
  have hφ : ∀ r : ℝ, HasDerivAt (fun t : ℝ => t)
      (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r :=
    fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r)
  have hσ : ∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r :=
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹)
  have h := hasDerivAt_deriv_energy_circleRotation β hβ (fun t => t)
    (fun _ => (2 : ℝ)⁻¹) 0 hφ hσ continuous_const (Pi.single 0 (1 : ℝ))
    (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
    (Perspective.diracProb 2 (basePoint 1)) (fun _ => 1) continuous_const
  simpa [Perspective.diracProb, circlePotentialCurvature, circleTangent,
    Fin.sum_univ_two, basePoint, secondAxis] using h

/-- All primitive, derivative, continuity, positive-temperature and strict
SOPD hypotheses are jointly realized by the source's pinned Dirac witness:
`φ=id`, `σ=1/2`, `σ'=0`, one neuron pointing opposite the atom, and `β=1`.
Its potential is the nonconstant function `-⟪x,·⟫`. -/
example : (0 : ℝ) < 1 ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (0 : ℝ → ℝ) ∧ Continuous (fun t : ℝ => t) ∧
    Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧
    IsStrictSOPD 1 (fun t => t) (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) := by
  refine ⟨one_pos, fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r),
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const,
    continuous_id, continuous_const, isStrictSOPD_pin 1 0 (basePoint 1)⟩

end Transformer.Perceptron
