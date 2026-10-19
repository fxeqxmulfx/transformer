module

public import Transformer.Perceptron.Section5_MapForce
public import Transformer.Perceptron.Section5_GeneralHessian

/-!
# The actual transported full-energy Hessian for matrix attention

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2,
`eq:Hess-bilinear.std`, to the matrix interaction `E_B` of §6.
Both the interaction and perceptron are integrated against the
original measure after the actual Monge transport is pulled back.

Their proved dominated first derivatives identify the derivative
of the full energy as a function. Differentiating that function
again gives existence and the value of its actual second derivative.
Strict SOPD can consequently be applied to this actual curve for an
arbitrary measure, before any finite-support reduction is established.

Both halves of `E_{B,ϑ}` have prefactor one half; the scalar-temperature
prefactor of `E_{β,ϑ}` is not substituted into the matrix energy.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- Pull back the original matrix energy along its actual transport.
Source: arXiv:2601.21366v2, §2.2, the Monge geodesic, and §3.1,
`rem: ext` (ii), with `E_B` as defined in §6, `rem:unifiedlog-B`. -/
theorem energyMap_circleRotation_eq (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (s : SSphere 2 → ℝ) (hs : Continuous s) (t : ℝ) :
    energyMap B φ ω a (μ.map (circleRotation s t)) =
      (1 / 2 : ℝ) * (∫ p, rotatedMapPairKernel B s t p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
      (1 / 2 : ℝ) * (∫ x, rotatedPotential φ ω a s t x ∂(μ : Measure (SSphere 2))) := by
  have hT := (continuous_circleRotation s hs t).measurable
  have hI : Integrable (fun p : SSphere 2 × SSphere 2 => circleMapPairKernel B p.1 p.2)
      ((μ.map (circleRotation s t) : Measure (SSphere 2)).prod
        (μ.map (circleRotation s t) : Measure (SSphere 2))) :=
    (continuous_circleMapPairKernel B).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  unfold energyMap interactionEnergyMap
  change (2 : ℝ)⁻¹ * (∫ x, ∫ y, circleMapPairKernel B x y
      ∂(μ.map (circleRotation s t) : Measure (SSphere 2))
      ∂(μ.map (circleRotation s t) : Measure (SSphere 2))) +
    (2 : ℝ)⁻¹ * (∫ x, potential φ ω a (x : EucSpace 2)
      ∂(μ.map (circleRotation s t) : Measure (SSphere 2))) = _
  rw [integral_integral (f := fun x y => circleMapPairKernel B x y) hI]
  simp only [ProbabilityMeasure.toMeasure_map]
  rw [Measure.map_prod_map _ _ hT hT,
    integral_map (hT.prodMap hT).aemeasurable (continuous_circleMapPairKernel B).aestronglyMeasurable,
    integral_map (f := fun x : SSphere 2 => potential φ ω a (x : EucSpace 2)) hT.aemeasurable
      ((continuous_potential hφ ω a).comp continuous_subtype_val).aestronglyMeasurable]
  simp only [one_div, rotatedMapPairKernel, rotatedPotential]
  rfl

/-- The genuine second derivative of the actual matrix coupled energy,
including both its interaction and perceptron terms. Source:
arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2,
`eq:Hess-bilinear.std`. No support or density conclusion is assumed. -/
theorem hasDerivAt_deriv_energyMap_circleRotation (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (hσ' : Continuous σ') (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (s : SSphere 2 → ℝ) (hs : Continuous s) :
    HasDerivAt (deriv (fun t => energyMap B φ ω a (μ.map (circleRotation s t))))
      ((1 / 2 : ℝ) * (∫ p, rotatedMapPairCurvature B s 0 p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
        (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2
          ∂(μ : Measure (SSphere 2)))) 0 := by
  let I : ℝ → ℝ := fun t => ∫ p, rotatedMapPairKernel B s t p
    ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))
  let P : ℝ → ℝ := fun t => ∫ x, rotatedPotential φ ω a s t x ∂(μ : Measure (SSphere 2))
  have hσc : Continuous σ := continuous_iff_continuousAt.mpr fun r => (hσ r).continuousAt
  have hI (t : ℝ) := hasDerivAt_integral_rotatedMapPairKernel B hsymm μ s hs t
  have hP (t : ℝ) := hasDerivAt_integral_rotatedPotential φ σ hφ hσc ω a μ s hs t
  have hE : (fun t => energyMap B φ ω a (μ.map (circleRotation s t))) =
      fun t => (1 / 2 : ℝ) * I t + (1 / 2 : ℝ) * P t := by
    funext t
    exact energyMap_circleRotation_eq B φ (continuous_of_hasDerivAt hφ) ω a μ s hs t
  have hfirst : deriv (fun t => energyMap B φ ω a (μ.map (circleRotation s t))) =
      fun t => (1 / 2 : ℝ) * deriv I t + (1 / 2 : ℝ) * deriv P t := by
    rw [hE]
    funext t
    exact (((hI t).differentiableAt.hasDerivAt.const_mul (1 / 2 : ℝ)).fun_add
      ((hP t).differentiableAt.hasDerivAt.const_mul (1 / 2 : ℝ))).deriv
  rw [hfirst]
  have hDI := (hasDerivAt_deriv_integral_rotatedMapPairKernel B hsymm μ s hs 0).const_mul
    (1 / 2 : ℝ)
  have hDP := (hasDerivAt_deriv_integral_rotatedPotential φ σ σ' hφ hσ hσ' ω a μ s hs 0).const_mul
    (1 / 2 : ℝ)
  apply (hDI.fun_add hDP).congr_deriv
  simp only [rotatedPotentialCurvature_zero, mul_assoc, integral_const_mul]
  ring

/-- The actual iterated derivative has the same proved matrix Hessian
value. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
extending §5.2. Derivative existence is supplied by the preceding proof. -/
theorem secondDeriv_energyMap_circleRotation (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (hσ' : Continuous σ') (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (s : SSphere 2 → ℝ) (hs : Continuous s) :
    deriv^[2] (fun t => energyMap B φ ω a (μ.map (circleRotation s t))) 0 =
      (1 / 2 : ℝ) * (∫ p, rotatedMapPairCurvature B s 0 p
        ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
        (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2
          ∂(μ : Measure (SSphere 2))) :=
  (hasDerivAt_deriv_energyMap_circleRotation B hsymm φ σ σ' hφ hσ hσ' ω a μ s hs).deriv

/-- Strict SOPD bounds the actual matrix Hessian below on every continuous
speed whose field is an actual smooth spherical gradient. Source:
arXiv:2601.21366v2, §2.2, `eq:strict2order`, and §3.1, `rem: ext` (ii). -/
theorem strict_circle_map_hessian_coercive (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ r, HasDerivAt φ (2 * σ r) r) (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (hσ' : Continuous σ') (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (hμ : IsStrictSOPDMap B φ σ ω a μ) :
    ∃ κ : ℝ, 0 < κ ∧ ∀ s : SSphere 2 → ℝ, Continuous s →
      IsGradientField (fun x => s x • circleTangent (x : EucSpace 2)) →
      κ * (∫ x, s x ^ 2 ∂(μ : Measure (SSphere 2))) ≤
        (1 / 2 : ℝ) * (∫ p, rotatedMapPairCurvature B s 0 p
          ∂(μ : Measure (SSphere 2)).prod (μ : Measure (SSphere 2))) +
          (∫ x, circlePotentialCurvature σ σ' ω a x * s x ^ 2
            ∂(μ : Measure (SSphere 2))) := by
  obtain ⟨κ, hκ, hH⟩ := hμ.2
  refine ⟨κ, hκ, fun s hs hgrad => ?_⟩
  have h := hH _ hgrad (fun t => μ.map (circleRotation s t))
    (isGeodesicFrom_circleRotation μ s hs) _
    (hasDerivAt_deriv_energyMap_circleRotation B hsymm φ σ σ' hφ hσ hσ' ω a μ s hs)
  have hnorm (x : SSphere 2) : ‖s x • circleTangent (x : EucSpace 2)‖ ^ 2 = s x ^ 2 := by
    rw [norm_smul, norm_circleTangent, mem_sphere_zero_iff_norm.mp x.2,
      mul_one, Real.norm_eq_abs, sq_abs]
  simpa only [hnorm] using h

/-- At the identity matrix the pinned Dirac's full actual second
variation is one half. Source: arXiv:2601.21366v2, §2.2 and §3.1,
`rem: ext` (ii). This checks the two matrix energy prefactors on the
original nonconstant perceptron and actual probability measure. -/
theorem hasDerivAt_deriv_energyMap_pin_rotation :
    HasDerivAt (deriv (fun t => energyMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun r => r) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      ((Perspective.diracProb 2 (basePoint 1)).map (circleRotation (fun _ => 1) t))))
      (1 / 2 : ℝ) 0 := by
  simpa only [energyMap_id_eq] using hasDerivAt_deriv_energy_pin_rotation 1 one_pos

/-- All symmetry, primitive, activation derivative, continuity and strict
SOPD inputs are jointly realized at `B=id` by the actual pinned Dirac,
with primitive identity, constant activation one half, and derivative zero. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    (∀ r : ℝ, HasDerivAt (fun t : ℝ => t) (2 * (fun _ : ℝ => (2 : ℝ)⁻¹) r) r) ∧
    (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (0 : ℝ → ℝ) ∧ Continuous (fun t : ℝ => t) ∧
    Continuous (fun _ : SSphere 2 => (1 : ℝ)) ∧
    IsStrictSOPDMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun t => t)
      (fun _ => (2 : ℝ)⁻¹) (Pi.single 0 (1 : ℝ))
      (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨fun _ _ => rfl, fun r => by simpa using hasDerivAt_id' (𝕜 := ℝ) (x := r),
    fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const, continuous_id,
    continuous_const, (isStrictSOPDMap_id_iff _ _ _ _ _).mpr (isStrictSOPD_pin 1 0 _)⟩

end Transformer.Perceptron
