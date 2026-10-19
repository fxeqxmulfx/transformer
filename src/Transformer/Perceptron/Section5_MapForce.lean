module

public import Transformer.Perceptron.Section5_MapIntegral
public import Transformer.Perceptron.Section5_CirclePotential
public import Mathlib.MeasureTheory.Function.L2Space

/-!
# Actual stationary force and curvature for matrix attention on the circle

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending the
first-variation calculation of §2.3 and §5.2 to `E_B` of §6,
`rem:unifiedlog-B`. The scalar angular force is the actual tangent
component of `energyGradMap`, so stationarity makes it zero at
every representative of every actual support point.

Its actual derivative is the integrated diagonal matrix curvature
plus the perceptron curvature. The coefficient is continuous for
continuously differentiable activations. These statements are proved
before any finite-support conclusion or Dirac decomposition is used.
-/

@[expose] public section

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Perceptron

/-- A jointly continuous circle pair function has a continuous integral
in its first point. Source: arXiv:2601.21366v2, §5.2, the local
continuity estimate, extended to `E_B` by §3.1, `rem: ext` (ii). -/
theorem continuous_integral_circle_pair (μ : Perspective.ProbSphere 2)
    (F : SSphere 2 × SSphere 2 → ℝ) (hF : Continuous F) :
    Continuous (fun x => ∫ y, F (x, y) ∂(μ : Measure (SSphere 2))) := by
  obtain ⟨C, -, hC⟩ := exists_pos_bound_continuous_compact F hF
  apply continuous_of_dominated (bound := fun _ => C)
  · intro x
    exact (hF.comp (continuous_const.prodMk continuous_id)).aestronglyMeasurable
  · intro x
    exact ae_of_all _ fun y => by simpa only [Real.norm_eq_abs] using hC (x, y)
  · exact integrable_const C
  · exact ae_of_all _ fun y => hF.comp (continuous_id.prodMk continuous_const)

/-- The joint-continuity input is realized by a nonzero constant
pair function on the actual nonempty circle. -/
example : Continuous (fun _ : SSphere 2 × SSphere 2 => (1 : ℝ)) := continuous_const

/-- The actual matrix interaction's angular force plus the perceptron
force. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
extending the first-variation field in §2.3. -/
noncomputable def circleForceMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (σ : ℝ → ℝ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (x : SSphere 2) : ℝ :=
  (∫ y, circleMapPairForce B x y ∂(μ : Measure (SSphere 2))) + circlePotentialForce σ ω a x

/-- The actual angular force derivative coefficient, retaining the
original measure. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
extending the local coefficient of `eq:Hess-bilinear` in §5.2. -/
noncomputable def circleCurvatureMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (σ σ' : ℝ → ℝ)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (x : SSphere 2) : ℝ :=
  (∫ y, circleMapPairCurvature B x y ∂(μ : Measure (SSphere 2))) +
    circlePotentialCurvature σ σ' ω a x

/-- The radial part of a projected vector is orthogonal to the
actual circle tangent. Source: arXiv:2601.21366v2, §2.3,
the spherical first-variation gradient, and §5.2. -/
theorem inner_circleTangent_proj (x v : EucSpace 2) :
    inner (𝕜 := ℝ) (circleTangent x) (proj 2 x v) =
      inner (𝕜 := ℝ) (circleTangent x) v := by
  rw [proj, inner_sub_right, real_inner_smul_right,
    real_inner_comm x (circleTangent x), inner_circleTangent, mul_zero, sub_zero]

/-- The scalar force is the tangent component of the actual matrix
gradient field. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
extending §2.3, `eq: steady.state`. The actual vector integral is
integrable, so its inner product may be passed through the integral. -/
theorem circleForceMap_eq_inner_energyGradMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (σ : ℝ → ℝ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (x : SSphere 2) :
    circleForceMap B σ ω a μ x =
      inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2))
        (energyGradMap B σ ω a μ (x : EucSpace 2)) := by
  let F : SSphere 2 → EucSpace 2 := fun y =>
    circleMapPairKernel B x y • proj 2 (x : EucSpace 2) (B (y : EucSpace 2))
  have hFC : Continuous F := by
    have hB := B.continuous_of_finiteDimensional
    have hK : Continuous (fun y : SSphere 2 => circleMapPairKernel B x y) :=
      (continuous_circleMapPairKernel B).comp (continuous_const.prodMk continuous_id)
    dsimp only [F, proj]
    fun_prop
  have hFI : Integrable F (μ : Measure (SSphere 2)) :=
    hFC.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  rw [circleForceMap, energyGradMap, inner_add_right]
  change (∫ y, circleMapPairForce B x y ∂(μ : Measure (SSphere 2))) +
    circlePotentialForce σ ω a x =
      inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2)) (∫ y, F y ∂(μ : Measure (SSphere 2))) +
        inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2)) (drift σ ω a (x : EucSpace 2))
  rw [← integral_inner hFI]
  congr 1
  · apply integral_congr_ae
    exact ae_of_all _ fun y => by
      simp only [F, real_inner_smul_right, inner_circleTangent_proj,
        circleMapPairForce, circleMapPairKernel]
  · rw [drift, inner_circleTangent_proj, inner_sum]
    unfold circlePotentialForce
    apply Finset.sum_congr rfl
    intro j _
    rw [real_inner_smul_right, real_inner_comm (a j) (circleTangent (x : EucSpace 2))]
    ring

/-- Stationarity makes the actual scalar force zero on actual support.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending
`eq: steady.state` as used in §5.2. -/
theorem circleForceMap_zero_on_support (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (σ : ℝ → ℝ) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B σ ω a μ)
    (x : SSphere 2) (hx : x ∈ (μ : Measure (SSphere 2)).support) :
    circleForceMap B σ ω a μ x = 0 := by
  rw [circleForceMap_eq_inner_energyGradMap, hμ x hx, inner_zero_right]

/-- Continuity of the actual local curvature coefficient follows from
the actual activation and its derivative. Source: arXiv:2601.21366v2,
§3.1, `rem: ext` (ii), extending the local estimate of §5.2. -/
theorem continuous_circleCurvatureMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (σ σ' : ℝ → ℝ) (hσ : Continuous σ) (hσ' : Continuous σ')
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) :
    Continuous (circleCurvatureMap B σ σ' ω a μ) :=
  (continuous_integral_circle_pair μ _ (continuous_circleMapPairCurvature B)).add
    (continuous_circlePotentialCurvature σ σ' hσ hσ' ω a)

/-- The local coefficient is the actual derivative of the stationary
angular force, by dominated differentiation of the actual interaction.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
theorem hasDerivAt_circleForceMap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (σ σ' : ℝ → ℝ) (hσ : ∀ r, HasDerivAt σ (σ' r) r)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2) (θ : ℝ) :
    HasDerivAt (fun r => circleForceMap B σ ω a μ (circlePoint r))
      (circleCurvatureMap B σ σ' ω a μ (circlePoint θ)) θ := by
  obtain ⟨C, -, hC⟩ := exists_pos_bound_continuous_compact _ (continuous_circleMapPairCurvature B)
  have hI := hasDerivAt_integral_on_compact (μ : Measure (SSphere 2))
    (fun r y => circleMapPairForce B (circlePoint r) y)
    (fun r y => circleMapPairCurvature B (circlePoint r) y)
    (fun r => (continuous_circleMapPairForce B).comp (continuous_const.prodMk continuous_id))
    (fun r => (continuous_circleMapPairCurvature B).comp (continuous_const.prodMk continuous_id))
    C (fun r y => hC (circlePoint r, y)) (fun r y => hasDerivAt_circleMapPairForce B y r) θ
  exact hI.fun_add (hasDerivAt_circlePotentialForce σ σ' hσ ω a θ)

/-- All activation derivative, continuity, stationarity and support
inputs hold jointly at the actual pinned Dirac with `B=id`, constant
activation `1/2`, derivative zero, and a neuron opposite its atom. -/
example : (∀ r : ℝ, HasDerivAt (fun _ : ℝ => (2 : ℝ)⁻¹) ((0 : ℝ → ℝ) r) r) ∧
    Continuous (fun _ : ℝ => (2 : ℝ)⁻¹) ∧ Continuous (0 : ℝ → ℝ) ∧
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun _ => (2 : ℝ)⁻¹)
      (Pi.single 0 (1 : ℝ)) (Pi.single 0 (-(circlePoint 0 : EucSpace 2)))
      (Perspective.diracProb 2 (circlePoint 0)) ∧
    circlePoint 0 ∈ (Perspective.diracProb 2 (circlePoint 0) : Measure (SSphere 2)).support := by
  refine ⟨fun r => by simpa using hasDerivAt_const r ((2 : ℝ)⁻¹), continuous_const,
    continuous_const, (isStationaryMap_id_iff _ _ _ _).mpr (isStrictSOPD_pin 1 0 _).1, ?_⟩
  apply (Measure.mem_support_iff_forall _).mpr
  intro U hU
  have hx : circlePoint 0 ∈ U := mem_of_mem_nhds hU
  change 0 < Measure.dirac (circlePoint 0) U
  simp [hx]

end Transformer.Perceptron
