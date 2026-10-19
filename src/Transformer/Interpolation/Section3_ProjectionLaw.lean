module

public import Transformer.Interpolation.Section1_SpacetimeTests
public import Transformer.Interpolation.Section3_ProjectionTransport

/-!
# Constructing actual weak solutions of a constant projection field

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
`lem: first.quadrant`. Push the original probability forward by the
explicit sphere flow. The resulting curve solves the original integrated
weak equation whenever its full vector field is that constant projection.
The derivative of each original `C¹` test follows from compact uniform
differentiation, and its continuous flux gives the time-integrated equation.
Thus the construction does not assume existence of a measure-valued solution.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual transported original probability law.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
noncomputable def projectionLaw (w : SSphere d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    Perspective.ProbSphere d := ν.map (sphereProjectionFlow w t)

/-- Its underlying measure is the original pushforward.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem coe_projectionLaw (w : SSphere d) (ν : Perspective.ProbSphere d) (t : ℝ) :
    (projectionLaw w ν t : Measure (SSphere d)) =
      Measure.map (sphereProjectionFlow w t) (ν : Measure (SSphere d)) := rfl

/-- The constructed curve starts at its original input law.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
@[simp] theorem projectionLaw_zero (w : SSphere d) (ν : Perspective.ProbSphere d) :
    projectionLaw w ν 0 = ν := by
  apply Subtype.ext
  change (projectionLaw w ν 0 : Measure (SSphere d)) = (ν : Measure (SSphere d))
  rw [coe_projectionLaw, show sphereProjectionFlow w 0 = id from
    funext fun x => sphereProjectionFlow_zero w x, Measure.map_id]

/-- Original continuous tests integrate against the actual transported
points. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem integral_projectionLaw (w : SSphere d) (ν : Perspective.ProbSphere d)
    (φ : EucSpace d → ℝ) (hφ : Continuous φ) (t : ℝ) :
    (∫ x, φ (x : EucSpace d) ∂(projectionLaw w ν t : Measure (SSphere d))) =
      ∫ x, φ (sphereProjectionFlow w t x : EucSpace d) ∂(ν : Measure (SSphere d)) := by
  rw [coe_projectionLaw]
  exact integral_map
    (((continuous_sphereProjectionFlow w).comp
      (continuous_const.prodMk continuous_id)).measurable.aemeasurable)
    (hφ.comp continuous_subtype_val).measurable.aestronglyMeasurable

/-- The original `C¹` test integrals have the actual projection flux as
their derivative. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`,
and §3.1, `lem: first.quadrant`. -/
theorem hasDerivAt_projectionLaw_test (w : SSphere d) (ν : Perspective.ProbSphere d)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d)
      ∂(projectionLaw w ν s : Measure (SSphere d)))
      (∫ x, projectionTestFlux φ (w : EucSpace d) (x : EucSpace d)
        ∂(projectionLaw w ν t : Measure (SSphere d))) t := by
  let F (s : ℝ) (x : SSphere d) := φ (sphereProjectionFlow w s x : EucSpace d)
  let G (s : ℝ) (x : SSphere d) :=
    projectionTestFlux φ (w : EucSpace d) (sphereProjectionFlow w s x : EucSpace d)
  have hc : Continuous (fun q : ℝ × SSphere d =>
      (sphereProjectionFlow w q.1 q.2 : EucSpace d)) :=
    continuous_subtype_val.comp (continuous_sphereProjectionFlow w)
  have hF : Continuous (Function.uncurry F) := hφ.continuous.comp hc
  have hflux := continuous_projectionTestFlux φ hφ (w : EucSpace d)
  have hG : Continuous (Function.uncurry G) := hflux.comp hc
  have hD (s : ℝ) (x : SSphere d) : HasDerivAt (fun r => F r x) (G s x) s :=
    (hφ.differentiable (by decide) (sphereProjectionFlow w s x : EucSpace d)).hasFDerivAt
      |>.comp_hasDerivAt s (hasDerivAt_sphereProjectionFlow w x s)
  have hd := hasDerivAt_spherical_compact_test (fun _ => ν) F G hF hG hD t 0
    (hasDerivAt_const t _) continuousAt_const
  have hm : (fun s => ∫ x, φ (x : EucSpace d)
      ∂(projectionLaw w ν s : Measure (SSphere d))) =
      (fun s => ∫ x, F s x ∂(ν : Measure (SSphere d))) :=
    funext fun s => integral_projectionLaw w ν φ hφ.continuous s
  rw [hm, integral_projectionLaw w ν (projectionTestFlux φ (w : EucSpace d)) hflux t]
  simpa only [zero_add] using hd

/-- The actual flux is continuous, including for tests that are only `C¹`.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem continuous_projectionLaw_flux (w : SSphere d) (ν : Perspective.ProbSphere d)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    Continuous (fun s => ∫ x, projectionTestFlux φ (w : EucSpace d) (x : EucSpace d)
      ∂(projectionLaw w ν s : Measure (SSphere d))) := by
  have hflux := continuous_projectionTestFlux φ hφ (w : EucSpace d)
  have heq : (fun s => ∫ x, projectionTestFlux φ (w : EucSpace d) (x : EucSpace d)
      ∂(projectionLaw w ν s : Measure (SSphere d))) =
      (fun s => ∫ x, projectionTestFlux φ (w : EucSpace d)
        (sphereProjectionFlow w s x : EucSpace d) ∂(ν : Measure (SSphere d))) :=
    funext fun s => integral_projectionLaw w ν _ hflux s
  rw [heq]
  apply continuous_spherical_integral_of_joint_continuous ν
  exact hflux.comp (continuous_subtype_val.comp (continuous_sphereProjectionFlow w))

/-- The constructed curve solves the original integrated weak equation
for every original `C¹` test and every real time.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
the constant-field phases of `lem: first.quadrant`. -/
theorem cauchyPB_projectionLaw (θ : TimeParams d) (w : SSphere d)
    (ν : Perspective.ProbSphere d)
    (hfield : ∀ s : ℝ, ∀ x : SSphere d,
      fullVF d θ (projectionLaw w ν s) s (x : EucSpace d) =
        proj d (x : EucSpace d) (w : EucSpace d)) :
    cauchyPB d θ (projectionLaw w ν) := by
  intro φ hφ t
  have heq : (fun s => ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
      (fullVF d θ (projectionLaw w ν s) s (x : EucSpace d))
        ∂(projectionLaw w ν s : Measure (SSphere d))) =
      (fun s => ∫ x, projectionTestFlux φ (w : EucSpace d) (x : EucSpace d)
        ∂(projectionLaw w ν s : Measure (SSphere d))) := by
    funext s
    apply integral_congr_ae
    exact ae_of_all _ fun x => by
      dsimp only
      rw [hfield s x, ← projectionTestFlux_eq_inner]
  rw [heq]
  have hi : IntervalIntegrable (fun s => ∫ x,
      projectionTestFlux φ (w : EucSpace d) (x : EucSpace d)
        ∂(projectionLaw w ν s : Measure (SSphere d))) volume 0 t :=
    (continuous_projectionLaw_flux w ν φ hφ).intervalIntegrable 0 t
  refine ⟨hi, ?_⟩
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_projectionLaw_test w ν φ hφ s) hi
  linarith

/-- A constant speed factor gives the exact time rescaling needed to
place each phase inside a prescribed time budget.
Source: arXiv:2411.04551v3, §3.1, the time rescaling at the end
of the proof of `lem: first.quadrant`. The identity holds for every
real factor; the construction uses positive factors. -/
theorem cauchyPB_projectionLaw_scale (θ : TimeParams d) (w : SSphere d)
    (ν : Perspective.ProbSphere d) (κ : ℝ)
    (hfield : ∀ s : ℝ, ∀ x : SSphere d,
      fullVF d θ (projectionLaw w ν (κ * s)) s (x : EucSpace d) =
        κ • proj d (x : EucSpace d) (w : EucSpace d)) :
    cauchyPB d θ (fun s => projectionLaw w ν (κ * s)) := by
  intro φ hφ t
  let J (s : ℝ) := κ * ∫ x, projectionTestFlux φ (w : EucSpace d) (x : EucSpace d)
    ∂(projectionLaw w ν (κ * s) : Measure (SSphere d))
  have hJ : Continuous J := ((continuous_projectionLaw_flux w ν φ hφ).comp
    (continuous_const.mul continuous_id)).const_mul κ
  have heq : (fun s => ∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
      (fullVF d θ (projectionLaw w ν (κ * s)) s (x : EucSpace d))
        ∂(projectionLaw w ν (κ * s) : Measure (SSphere d))) = J := by
    funext s
    dsimp only [J]
    rw [← integral_const_mul]
    apply integral_congr_ae
    exact ae_of_all _ fun x => by
      dsimp only
      rw [hfield s x, real_inner_smul_right, ← projectionTestFlux_eq_inner]
  have hD (s : ℝ) : HasDerivAt (fun r => ∫ x, φ (x : EucSpace d)
      ∂(projectionLaw w ν (κ * r) : Measure (SSphere d))) (J s) s := by
    simpa only [Function.comp_def, id_eq, mul_one, smul_eq_mul] using
      (hasDerivAt_projectionLaw_test w ν φ hφ (κ * s)).scomp s
        ((hasDerivAt_id s).const_mul κ)
  rw [heq]
  have hi : IntervalIntegrable J volume 0 t := hJ.intervalIntegrable 0 t
  refine ⟨hi, ?_⟩
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt (fun s _ => hD s) hi
  linarith

/-- The actual law at the attracting Dirac remains that same nonzero law.
Source: arXiv:2411.04551v3, §3.1, the stationary pole in
`lem: first.quadrant`. -/
@[simp] theorem projectionLaw_dirac_axis (w : SSphere d) (t : ℝ) :
    projectionLaw w (Perspective.diracProb d w) t = Perspective.diracProb d w := by
  apply Subtype.ext
  change (projectionLaw w (Perspective.diracProb d w) t : Measure (SSphere d)) =
    (Perspective.diracProb d w : Measure (SSphere d))
  rw [coe_projectionLaw]
  change Measure.map (sphereProjectionFlow w t) (Measure.dirac w) = Measure.dirac w
  rw [Measure.map_dirac, sphereProjectionFlow_axis]

/-- The existence hypothesis is realized by actual nonzero data and an
original attention parameter curve: the constructed Dirac curve solves
the original weak equation. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`. -/
example : cauchyPB 2 (pureAttentionParams 2 0)
    (projectionLaw (basePoint 1) (Perspective.diracProb 2 (basePoint 1))) := by
  apply cauchyPB_projectionLaw
  intro s x
  rw [projectionLaw_dirac_axis, fullVF_pureAttention_zero]
  simp [Perspective.diracProb]

/-- A nonconstant original test realizes the `C¹` test hypothesis. -/
example : ContDiff ℝ 1 (fun x : EucSpace 2 => inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) x) :=
  contDiff_const.inner ℝ contDiff_id

end Transformer.Interpolation
