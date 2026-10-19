module

public import Transformer.Interpolation.Section3_MeanAxisLaw
public import Transformer.Interpolation.Section1_TimeDependentWeakTests

/-!
# Differential weak tests and clocks for every rank-one mean law

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.2,
`eq: cauchy.pb.reloaded`. Every original rank-one weak law has
continuous C1-test fluxes, hence the actual fixed-test derivative.
Compact differentiation then gives C1 spacetime tests. In particular,
the original mean satisfies the source's homogeneous moment equation.

The time integral of the actual axial mean defines a C1 intrinsic
clock for any original weak law. No characteristic representation or
uniqueness assumption enters these statements.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every original C1-test flux is continuous along every actual weak law.
Source: arXiv:2411.04551v3, §1 and §3.2, `eq: cauchy.pb.reloaded`. -/
theorem continuous_meanAxis_cauchyPB_flux (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) :
    Continuous (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (μ s) s x)
      ∂(μ s : Measure (SSphere d))) := by
  have he : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (μ s) s x)
      ∂(μ s : Measure (SSphere d))) = fun s => (κ * meanAxisMoment (w : EucSpace d) (μ s)) *
      ∫ x : SSphere d, projectionTestFlux φ (w : EucSpace d) x ∂(μ s : Measure (SSphere d)) := by
    funext s
    simp_rw [fullVF_meanAxisParams, real_inner_smul_right, ← projectionTestFlux_eq_inner]
    exact integral_const_mul _ _
  rw [he]
  exact (continuous_const.mul (continuous_meanAxisMoment_of_cauchyPB w κ μ hμ)).mul
    (continuous_meanAxis_cauchyPB_test w κ μ hμ _ (continuous_projectionTestFlux φ hφ _))

/-- The original integrated weak equation gives the actual C1-test
derivative. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem hasDerivAt_meanAxis_cauchyPB_test (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, φ x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
        (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (μ t) t x)
        ∂(μ t : Measure (SSphere d))) t :=
  hasDerivAt_cauchyPB_test _ μ hμ φ hφ (continuous_meanAxis_cauchyPB_flux w κ μ hμ φ hφ) t

/-- Every original rank-one weak law differentiates C1 spacetime tests.
Auxiliary for arXiv:2411.04551v3, §3.2, the characteristic argument. -/
theorem hasDerivAt_meanAxis_cauchyPB_spacetime (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (A : ℝ → EucSpace d → ℝ) (hA : ContDiff ℝ 1 (Function.uncurry A)) (t : ℝ) :
    HasDerivAt (fun s => ∫ x : SSphere d, A s x ∂(μ s : Measure (SSphere d)))
      (∫ x : SSphere d, fderiv ℝ (Function.uncurry A) (t, (x : EucSpace d))
        (1, fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (μ t) t x)
        ∂(μ t : Measure (SSphere d))) t := by
  simp_rw [fullVF_meanAxisParams]
  let F (s : ℝ) (x : EucSpace d) := (κ * meanAxisMoment (w : EucSpace d) (μ s)) • proj d x (w : EucSpace d)
  have hP : Continuous (fun x : EucSpace d => proj d x (w : EucSpace d)) :=
    continuous_const.sub ((continuous_id.inner (𝕜 := ℝ) continuous_const).smul continuous_id)
  have hF (s : ℝ) : Continuous (F s) :=
    (continuous_const : Continuous (fun _ : EucSpace d => κ * meanAxisMoment (w : EucSpace d) (μ s))).smul hP
  have hweak (φ : EucSpace d → ℝ) (hφ : ContDiff ℝ 1 φ) (s : ℝ) :
      HasDerivAt (fun r => ∫ x : SSphere d, φ x ∂(μ r : Measure (SSphere d)))
        (∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d)) (F s x)
          ∂(μ s : Measure (SSphere d))) s := by
    simpa only [fullVF_meanAxisParams] using hasDerivAt_meanAxis_cauchyPB_test w κ μ hμ φ hφ s
  exact hasDerivAt_timeDependent_spacetime_test F hF μ
    (continuous_meanAxis_cauchyPB_test w κ μ hμ) hweak A hA t

/-- The axial mean of every original weak law satisfies the source
moment ODE. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem hasDerivAt_meanAxis_cauchyPB_moment (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) (t : ℝ) :
    HasDerivAt (fun s => meanAxisMoment (w : EucSpace d) (μ s))
      ((κ * meanAxisMoment (w : EucSpace d) (μ t)) *
        (1 - ∫ x : SSphere d, (inner ℝ (w : EucSpace d) (x : EucSpace d)) ^ 2
          ∂(μ t : Measure (SSphere d)))) t := by
  have he : (fun s => meanAxisMoment (w : EucSpace d) (μ s)) =
      fun s => ∫ x : SSphere d, inner ℝ (w : EucSpace d) (x : EucSpace d)
        ∂(μ s : Measure (SSphere d)) := funext fun s => meanAxisMoment_eq_integral _ _
  rw [he]
  apply (hasDerivAt_meanAxis_cauchyPB_test w κ μ hμ
    (fun x : EucSpace d => inner ℝ (w : EucSpace d) x)
    (contDiff_const.inner ℝ contDiff_id) t).congr_deriv
  have hp (x : SSphere d) : inner ℝ
      (gradient (fun y : EucSpace d => inner ℝ (w : EucSpace d) y) (x : EucSpace d))
      (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (μ t) t x) =
      (κ * meanAxisMoment (w : EucSpace d) (μ t)) *
        (1 - (inner ℝ (w : EucSpace d) (x : EucSpace d)) ^ 2) := by
    rw [gradient_inner_linear, fullVF_meanAxisParams, real_inner_smul_right, proj,
      inner_sub_right, real_inner_smul_right, real_inner_self_eq_norm_sq,
      mem_sphere_zero_iff_norm.mp w.2, real_inner_comm (x : EucSpace d) (w : EucSpace d)]
    ring
  have hi : Integrable (fun x : SSphere d => (inner ℝ (w : EucSpace d) (x : EucSpace d)) ^ 2)
      (μ t : Measure (SSphere d)) :=
    integrable_spherical_continuous (μ t) _ ((continuous_const.inner continuous_subtype_val).pow 2)
  rw [integral_congr_ae (ae_of_all _ hp), integral_const_mul]
  have hs := integral_sub (integrable_const (1 : ℝ) : Integrable (fun _ : SSphere d => (1 : ℝ))
    (μ t : Measure (SSphere d))) hi
  simp only [integral_const, probReal_univ, one_smul] at hs
  rw [hs]

/-- The intrinsic time integral of the actual weak law
in arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
noncomputable def meanAxisWeakClock (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (t : ℝ) : ℝ := ∫ s in 0..t, κ * meanAxisMoment (w : EucSpace d) (μ s)

/-- The actual weak-law clock starts at intrinsic time zero.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisWeakClock_zero (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d) :
    meanAxisWeakClock w κ μ 0 = 0 := intervalIntegral.integral_same

/-- The weak-law clock has the actual mean coefficient as derivative.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem hasDerivAt_meanAxisWeakClock (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) (t : ℝ) :
    HasDerivAt (meanAxisWeakClock w κ μ) (κ * meanAxisMoment (w : EucSpace d) (μ t)) t :=
  ((continuous_const.mul (continuous_meanAxisMoment_of_cauchyPB w κ μ hμ)).integral_hasStrictDerivAt 0 t).hasDerivAt

/-- The actual weak-law clock is continuous.
Auxiliary for arXiv:2411.04551v3, §3.2, the transported-test argument. -/
theorem continuous_meanAxisWeakClock (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) :
    Continuous (meanAxisWeakClock w κ μ) :=
  continuous_iff_continuousAt.mpr fun t => (hasDerivAt_meanAxisWeakClock w κ μ hμ t).continuousAt

/-- The actual weak-law clock is C1, without assuming law transport.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem contDiff_meanAxisWeakClock (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) :
    ContDiff ℝ 1 (meanAxisWeakClock w κ μ) := by
  refine contDiff_one_iff_deriv.mpr ⟨fun s => (hasDerivAt_meanAxisWeakClock w κ μ hμ s).differentiableAt, ?_⟩
  have he : deriv (meanAxisWeakClock w κ μ) = fun s => κ * meanAxisMoment (w : EucSpace d) (μ s) :=
    funext fun s => (hasDerivAt_meanAxisWeakClock w κ μ hμ s).deriv
  rw [he]
  exact continuous_const.mul (continuous_meanAxisMoment_of_cauchyPB w κ μ hμ)

/-- A constructed actual weak law and nonconstant original tests realize
the weak-law, C1 spatial-test and C1 spacetime-test hypotheses jointly. -/
example : ∃ (w : SSphere 2) (μ : ℝ → Perspective.ProbSphere 2)
    (φ : EucSpace 2 → ℝ) (A : ℝ → EucSpace 2 → ℝ),
    cauchyPB 2 (fun _ => meanAxisParams (w : EucSpace 2) 1) μ ∧
      ContDiff ℝ 1 φ ∧ ContDiff ℝ 1 (Function.uncurry A) := by
  let w : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  refine ⟨w, meanAxisLaw w 1 (Perspective.diracProb 2 z),
    (fun x => inner ℝ (w : EucSpace 2) x),
    (fun s x => sin s + inner ℝ (w : EucSpace 2) x), cauchyPB_meanAxisLaw w 1 _,
    contDiff_const.inner ℝ contDiff_id, ?_⟩
  exact (contDiff_sin.comp contDiff_fst).add (contDiff_const.inner ℝ contDiff_snd)

end Transformer.Interpolation
