module

public import Transformer.Interpolation.Section3_MeanAxisClock
public import Transformer.Interpolation.Section1_AutonomousComparison

/-!
# Actual weak laws for rank-one mean attention

ArXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`, proof of
`lem: induction.barycenter`, Step 1. Push the original initial law
along the actual projection flow at its own scalar clock. The clock
coefficient is computed from this same transported probability law,
so its characteristics solve the original nonlinear vector field.

Compact differentiation of every original C1 test and the fundamental
theorem of calculus prove the original integrated weak equation at
all real times. Existence is constructed, without a well-posedness
assumption. A vanishing initial axial mean gives a stationary law.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual characteristics of the rank-one attention law.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
noncomputable def meanAxisFlow (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : SSphere d) : SSphere d := sphereProjectionFlow w (meanAxisClock w κ ν t) x

/-- The original probability transported at its own nonlinear clock.
Source: arXiv:2411.04551v3, §3.2, Step 1 of `lem: induction.barycenter`. -/
noncomputable def meanAxisLaw (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (t : ℝ) : Perspective.ProbSphere d := projectionLaw w ν (meanAxisClock w κ ν t)

/-- Characteristics start at the prescribed original point.
Source: arXiv:2411.04551v3, §1, `eq: ips`. -/
theorem meanAxisFlow_zero (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (x : SSphere d) :
    meanAxisFlow w κ ν 0 x = x := by rw [meanAxisFlow, meanAxisClock_zero, sphereProjectionFlow_zero]

/-- The constructed law has the original initial datum.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem meanAxisLaw_zero (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    meanAxisLaw w κ ν 0 = ν := by rw [meanAxisLaw, meanAxisClock_zero, projectionLaw_zero]

/-- The constructed law is the actual characteristic pushforward.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem coe_meanAxisLaw (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    (meanAxisLaw w κ ν t : Measure (SSphere d)) = Measure.map (meanAxisFlow w κ ν t) (ν : Measure (SSphere d)) := rfl

/-- The actual characteristics are jointly continuous.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem continuous_meanAxisFlow (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    Continuous (fun q : ℝ × SSphere d => meanAxisFlow w κ ν q.1 q.2) :=
  (continuous_sphereProjectionFlow w).comp
    (((continuous_meanAxisClock w κ ν).comp continuous_fst).prodMk continuous_snd)

/-- Each actual characteristic solves the original nonlinear field.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem hasDerivAt_meanAxisFlow (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (x : SSphere d) (t : ℝ) : HasDerivAt (fun s => (meanAxisFlow w κ ν s x : EucSpace d))
      (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (meanAxisLaw w κ ν t) t
        (meanAxisFlow w κ ν t x)) t := by
  rw [fullVF_meanAxisParams]
  exact (hasDerivAt_sphereProjectionFlow w x (meanAxisClock w κ ν t)).scomp t (hasDerivAt_meanAxisClock w κ ν t)

/-- Original characteristics have C1 time regularity.
Auxiliary for arXiv:2411.04551v3, §3.2, the weak formulation. -/
theorem contDiff_meanAxisFlow_time (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (x : SSphere d) :
    ContDiff ℝ 1 (fun s => (meanAxisFlow w κ ν s x : EucSpace d)) := by
  have hF : Continuous (fun y : EucSpace d => proj d y (w : EucSpace d)) :=
    continuous_const.sub ((continuous_id.inner (𝕜 := ℝ) continuous_const).smul continuous_id)
  have hC := contDiff_spherical_particle _ hF (fun q => sphereProjectionFlow w q x)
    (hasDerivAt_sphereProjectionFlow w x)
  exact hC.comp (contDiff_meanAxisClock w κ ν)

/-- Every original C1 test has the actual nonlinear projection flux
as derivative. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_meanAxisLaw_test (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, φ x ∂(meanAxisLaw w κ ν s : Measure (SSphere d)))
      ((κ * meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν t)) *
        ∫ x : SSphere d, projectionTestFlux φ (w : EucSpace d) x
          ∂(meanAxisLaw w κ ν t : Measure (SSphere d))) t := by
  have hd := (hasDerivAt_projectionLaw_test w ν φ hφ (meanAxisClock w κ ν t)).scomp t
    (hasDerivAt_meanAxisClock w κ ν t)
  simpa only [Function.comp_def, smul_eq_mul, meanAxisClockSpeed, meanAxisLaw] using hd

/-- The transported law solves the original integrated weak equation.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem cauchyPB_meanAxisLaw (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) (meanAxisLaw w κ ν) := by
  intro φ hφ t
  let J (s : ℝ) := (κ * meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν s)) *
    ∫ x : SSphere d, projectionTestFlux φ (w : EucSpace d) x ∂(meanAxisLaw w κ ν s : Measure (SSphere d))
  have hc : Continuous (fun s => meanAxisMoment (w : EucSpace d) (meanAxisLaw w κ ν s)) :=
    (continuous_meanAxisClockSpeed w ν).comp (continuous_meanAxisClock w κ ν)
  have hJ : Continuous J := (continuous_const.mul hc).mul
    ((continuous_projectionLaw_flux w ν φ hφ).comp (continuous_meanAxisClock w κ ν))
  have heq : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (meanAxisLaw w κ ν s) s x)
      ∂(meanAxisLaw w κ ν s : Measure (SSphere d))) = J := by
    funext s
    dsimp only [J]
    rw [← integral_const_mul]
    apply integral_congr_ae
    exact ae_of_all _ fun x => by
      change inner ℝ (gradient φ (x : EucSpace d))
        (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (meanAxisLaw w κ ν s) s x) = _
      rw [fullVF_meanAxisParams, real_inner_smul_right, ← projectionTestFlux_eq_inner]
  rw [heq]
  have hi : IntervalIntegrable J volume 0 t := hJ.intervalIntegrable 0 t
  refine ⟨hi, ?_⟩
  have hFTC := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun s _ => hasDerivAt_meanAxisLaw_test w κ ν φ hφ s) hi
  linarith

/-- An actual global weak solution exists for every initial law.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_cauchyPB_meanAxis (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ :=
  ⟨meanAxisLaw w κ ν, meanAxisLaw_zero w κ ν, cauchyPB_meanAxisLaw w κ ν⟩

/-- The constructed law is invariant when its initial mean is
orthogonal to the axis. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisLaw_of_moment_eq_zero (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (h0 : meanAxisMoment (w : EucSpace d) ν = 0) : meanAxisLaw w κ ν = fun _ => ν := by
  funext t
  rw [meanAxisLaw, meanAxisClock_of_moment_eq_zero w κ ν h0, projectionLaw_zero]

/-- The zero control gives the original stationary probability law.
Auxiliary for arXiv:2411.04551v3, §3.2, time rescaling. -/
theorem meanAxisLaw_of_zero_factor (w : SSphere d) (ν : Perspective.ProbSphere d) :
    meanAxisLaw w 0 ν = fun _ => ν := by
  funext t
  rw [meanAxisLaw, meanAxisClock_of_zero_factor w ν, projectionLaw_zero]

/-- A transverse nonzero probability law and a nonconstant linear test
realize the zero-mean and C1 hypotheses for an actual original weak law. -/
example : ∃ (w : SSphere 2) (ν : Perspective.ProbSphere 2) (φ : EucSpace 2 → ℝ)
    (μ : ℝ → Perspective.ProbSphere 2),
    meanAxisMoment (w : EucSpace 2) ν = 0 ∧ ContDiff ℝ 1 φ ∧ μ 0 = ν ∧
    cauchyPB 2 (fun _ => meanAxisParams (w : EucSpace 2) 1) μ := by
  let w : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let ν := Perspective.diracProb 2 z
  refine ⟨w, ν, fun x => inner ℝ (w : EucSpace 2) x, meanAxisLaw w 1 ν, ?_,
    contDiff_const.inner ℝ contDiff_id, meanAxisLaw_zero w 1 ν, cauchyPB_meanAxisLaw w 1 ν⟩
  rw [meanAxisMoment_eq_integral]
  change (∫ y : SSphere 2, inner ℝ (w : EucSpace 2) (y : EucSpace 2) ∂Measure.dirac z) = 0
  rw [integral_dirac]
  simp [w, z, basePoint, EuclideanSpace.inner_single_left]

end Transformer.Interpolation
