module

public import Transformer.Interpolation.Section3_MeanAxisPullback
public import Transformer.Interpolation.Section1_TimeTranslation

/-!
# Transport, uniqueness and reversal for every rank-one mean law

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.2,
`eq: cauchy.pb.reloaded`, Steps 1 and 3 of `lem: induction.barycenter`.
Conservation of inverse-test integrals determines every original weak
law as the pushforward of its initial law at its own intrinsic clock.
The scalar clock uniqueness theorem then identifies it with the
constructed nonlinear mean-attention law, at every real time.

Existence, uniqueness, stationarity for zero axial mean, and support
transport therefore hold for arbitrary initial probabilities. Translating
and reversing actual original weak curves gives the probability flow's
group law and exact inverse, including for negative control factors.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Every original weak law is its actual characteristic pushforward.
Source: arXiv:2411.04551v3, §1 and §3.2, `eq: cauchy.pb.reloaded`. -/
theorem cauchyPB_meanAxis_map (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) (t : ℝ) :
    (μ t : Measure (SSphere d)) =
      Measure.map (sphereProjectionFlow w (meanAxisWeakClock w κ μ t)) (μ 0 : Measure (SSphere d)) := by
  have hm (r : ℝ) : Measurable (sphereProjectionFlow w r) :=
    ((continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id)).measurable
  have he := congrArg (Measure.map (sphereProjectionFlow w (meanAxisWeakClock w κ μ t)))
    (cauchyPB_meanAxis_inverse_map w κ μ hμ t)
  rw [Measure.map_map (hm _) (hm _)] at he
  have hi : sphereProjectionFlow w (meanAxisWeakClock w κ μ t) ∘
      sphereProjectionFlow w (-meanAxisWeakClock w κ μ t) = id :=
    funext fun x => sphereProjectionFlow_neg_right w x _
  rw [hi, Measure.map_id] at he
  exact he

/-- The actual weak law is a projection law at its own intrinsic clock.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem cauchyPB_meanAxis_eq_projectionLaw (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) (t : ℝ) :
    μ t = projectionLaw w (μ 0) (meanAxisWeakClock w κ μ t) :=
  Subtype.ext (cauchyPB_meanAxis_map w κ μ hμ t)

/-- Every actual weak-law clock solves the unique original scalar ODE.
Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisWeakClock_eq (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) :
    meanAxisWeakClock w κ μ = meanAxisClock w κ (μ 0) := by
  apply Eq.symm
  apply meanAxisClock_eq_of_solution w κ (μ 0) _ (meanAxisWeakClock_zero w κ μ)
  intro t
  have hd := hasDerivAt_meanAxisWeakClock w κ μ hμ t
  rw [cauchyPB_meanAxis_eq_projectionLaw w κ μ hμ t] at hd
  exact hd

/-- Every original weak law equals the constructed nonlinear law.
Source: arXiv:2411.04551v3, §1 and §3.2, well-posedness. -/
theorem cauchyPB_meanAxis_eq (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) :
    μ = meanAxisLaw w κ (μ 0) := by
  funext t
  rw [cauchyPB_meanAxis_eq_projectionLaw w κ μ hμ t, meanAxisWeakClock_eq w κ μ hμ]
  rfl

/-- Equal initial laws give equal original rank-one weak solutions.
Source: arXiv:2411.04551v3, §1 and §3.2, well-posedness. -/
theorem cauchyPB_meanAxis_unique (w : SSphere d) (κ : ℝ) (μ ν : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (hν : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) ν) (h0 : μ 0 = ν 0) : μ = ν := by
  rw [cauchyPB_meanAxis_eq w κ μ hμ, cauchyPB_meanAxis_eq w κ ν hν, h0]

/-- Every original weak law with zero initial axial mean is stationary.
Source: arXiv:2411.04551v3, §3.2, the invariant laws of Step 1. -/
theorem cauchyPB_meanAxis_stationary (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (h0 : meanAxisMoment (w : EucSpace d) (μ 0) = 0) : μ = fun _ => μ 0 := by
  exact (cauchyPB_meanAxis_eq w κ μ hμ).trans (meanAxisLaw_of_moment_eq_zero w κ (μ 0) h0)

/-- Actual global weak existence and uniqueness hold for every input law.
Source: arXiv:2411.04551v3, §1 and §3.2, well-posedness. -/
theorem exists_unique_cauchyPB_meanAxis (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = ν ∧ cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ := by
  refine ⟨meanAxisLaw w κ ν, ⟨meanAxisLaw_zero w κ ν, cauchyPB_meanAxisLaw w κ ν⟩, fun μ hμ => ?_⟩
  have he := cauchyPB_meanAxis_eq w κ μ hμ.2
  rwa [hμ.1] at he

/-- Original supports follow the actual nonlinear characteristic map.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem cauchyPB_meanAxis_support (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) (t : ℝ) :
    (μ t : Measure (SSphere d)).support ⊆
      sphereProjectionFlow w (meanAxisClock w κ (μ 0) t) '' (μ 0 : Measure (SSphere d)).support := by
  have he : (μ t : Measure (SSphere d)) = Measure.map (sphereProjectionFlow w (meanAxisClock w κ (μ 0) t))
      (μ 0 : Measure (SSphere d)) := by rw [← meanAxisWeakClock_eq w κ μ hμ]; exact cauchyPB_meanAxis_map w κ μ hμ t
  rw [he]
  exact support_map_subset_image _ _ ((continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id))

/-- Actual mean-attention probability maps form a complete flow.
Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem meanAxisLaw_add (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (s t : ℝ) :
    meanAxisLaw w κ (meanAxisLaw w κ ν s) t = meanAxisLaw w κ ν (s + t) := by
  have hshift : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ)
      (fun t => meanAxisLaw w κ ν (s + t)) :=
    cauchyPB_translateTimeParams s (fun _ => meanAxisParams (w : EucSpace d) κ) _ (cauchyPB_meanAxisLaw w κ ν)
  have he := cauchyPB_meanAxis_eq w κ _ hshift
  simpa only [add_zero] using (congrFun he t).symm

/-- Negating the original value matrix reverses the actual weak-law time.
Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem meanAxisLaw_neg_factor (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    meanAxisLaw w (-κ) ν t = meanAxisLaw w κ ν (-t) := by
  have hr := cauchyPB_reverseTimeParams 0 (fun _ => meanAxisParams (w : EucSpace d) κ)
    (meanAxisLaw w κ ν) (cauchyPB_meanAxisLaw w κ ν)
  have hparams : reverseTimeParams 0 (fun _ => meanAxisParams (w : EucSpace d) κ) =
      (fun _ => meanAxisParams (w : EucSpace d) (-κ)) := by
    funext s
    simp only [reverseTimeParams, rescaleTimeParams, translateTimeParams, meanAxisParams,
      smul_smul, neg_one_mul]
    congr 1
    ext x
    simp only [smul_apply, zero_apply, smul_zero]
  rw [hparams] at hr
  have he := cauchyPB_meanAxis_eq w (-κ) _ hr
  simpa only [zero_sub, neg_zero, meanAxisLaw_zero] using (congrFun he t).symm

/-- The reversed original control recovers the initial law exactly.
Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem meanAxisLaw_inverse (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    meanAxisLaw w (-κ) (meanAxisLaw w κ ν t) t = ν := by
  rw [meanAxisLaw_neg_factor, meanAxisLaw_add, add_neg_cancel, meanAxisLaw_zero]

/-- Every actual probability flow map is bijective, with the proved
reverse control as inverse. Source: arXiv:2411.04551v3, §3.2, Step 3. -/
theorem bijective_meanAxisLaw (w : SSphere d) (κ t : ℝ) :
    Function.Bijective (fun ν : Perspective.ProbSphere d => meanAxisLaw w κ ν t) := by
  constructor
  · intro ν η he
    have hh := congrArg (fun ρ => meanAxisLaw w (-κ) ρ t) he
    simpa only [meanAxisLaw_inverse] using hh
  · intro ν
    refine ⟨meanAxisLaw w (-κ) ν t, ?_⟩
    simpa only [neg_neg] using meanAxisLaw_inverse w (-κ) ν t

/-- A transverse Dirac datum realizes the stationary hypothesis. Two
actual original weak curves with that same datum realize the weak-law
and initial-equality hypotheses without assuming any well-posedness. -/
example : ∃ (w : SSphere 2) (μ ν : ℝ → Perspective.ProbSphere 2),
    cauchyPB 2 (fun _ => meanAxisParams (w : EucSpace 2) 1) μ ∧
    cauchyPB 2 (fun _ => meanAxisParams (w : EucSpace 2) 1) ν ∧
      μ 0 = ν 0 ∧ meanAxisMoment (w : EucSpace 2) (μ 0) = 0 := by
  let w : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let μ := meanAxisLaw w 1 (Perspective.diracProb 2 z)
  refine ⟨w, μ, μ, cauchyPB_meanAxisLaw w 1 _, cauchyPB_meanAxisLaw w 1 _, rfl, ?_⟩
  change meanAxisMoment (w : EucSpace 2) (meanAxisLaw w 1 (Perspective.diracProb 2 z) 0) = 0
  rw [meanAxisLaw_zero, meanAxisMoment_eq_integral]
  change (∫ x : SSphere 2, inner ℝ (w : EucSpace 2) (x : EucSpace 2) ∂Measure.dirac z) = 0
  rw [integral_dirac]
  simp [w, z, basePoint, EuclideanSpace.inner_single_left]

end Transformer.Interpolation
