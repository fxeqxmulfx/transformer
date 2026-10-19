module

public import Transformer.Interpolation.Section3_MeanAxisClockGrowth
public import Transformer.Interpolation.Section3_MeanAxisTransport
public import Transformer.Interpolation.Section1_TimeDependentFlowBounds

/-!
# Uniform control of rank-one attention near stationary poles

ArXiv:2411.04551v3, §3.2, Steps 1 and 3 of `lem: induction.barycenter`.
If the initial law lies within radius δ of a pole orthogonal to the
attention axis, its entire support stays within δ exp(|κ|t) of that pole.
The estimate holds for the original nonlinear law, including reversed
controls, and for every original weak solution.

The scalar clock estimate follows from its actual globally Lipschitz
coefficient and Grönwall. Projection characteristics have unit intrinsic
speed. Their combination gives a constant independent of δ, as needed
when choosing the clustering radius before applying the reverse phase.
Exact time scaling also puts each actual phase in a prescribed horizon.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual intrinsic clock grows at most proportionally to its
initial mean. Auxiliary for arXiv:2411.04551v3, §3.2, Steps 1 and 3. -/
theorem abs_meanAxisClock_le (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) (ht : 0 ≤ t) :
    |meanAxisClock w κ ν t| ≤ |meanAxisMoment (w : EucSpace d) ν| * (exp (|κ| * t) - 1) := by
  by_cases hk : κ = 0
  · subst κ
    rw [meanAxisClock_of_zero_factor]
    simp
  have hbound (s : ℝ) : ‖κ * meanAxisClockSpeed w ν (meanAxisClock w κ ν s)‖ ≤
      |κ| * ‖meanAxisClock w κ ν s‖ + |κ| * |meanAxisMoment (w : EucSpace d) ν| := by
    have hl := (lipschitzWith_meanAxisClockSpeed w ν).dist_le_mul (meanAxisClock w κ ν s) 0
    simp only [Real.dist_eq, NNReal.coe_one, one_mul, sub_zero, meanAxisClockSpeed_zero] at hl
    have hb : |meanAxisClockSpeed w ν (meanAxisClock w κ ν s)| ≤
        |meanAxisClock w κ ν s| + |meanAxisMoment (w : EucSpace d) ν| := by
      have hh := abs_add_le (meanAxisClockSpeed w ν (meanAxisClock w κ ν s) - meanAxisMoment (w : EucSpace d) ν)
        (meanAxisMoment (w : EucSpace d) ν)
      rw [sub_add_cancel] at hh
      linarith
    rw [Real.norm_eq_abs, abs_mul, Real.norm_eq_abs]
    nlinarith [mul_le_mul_of_nonneg_left hb (abs_nonneg κ)]
  have hg := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := 0) (K := |κ|) (ε := |κ| * |meanAxisMoment (w : EucSpace d) ν|)
    (continuous_meanAxisClock w κ ν).continuousOn
    (fun s _ => (hasDerivAt_meanAxisClock w κ ν s).hasDerivWithinAt)
    (by rw [meanAxisClock_zero, norm_zero]) (fun s _ => hbound s) t ⟨ht, le_rfl⟩
  rw [gronwallBound_of_K_ne_0 (abs_ne_zero.mpr hk)] at hg
  simpa only [sub_zero, zero_mul, zero_add, mul_div_cancel_left₀ _ (abs_ne_zero.mpr hk), Real.norm_eq_abs] using hg

/-- Projection characteristics have unit intrinsic speed.
Source: arXiv:2411.04551v3, §3.1, the characteristic equation. -/
theorem lipschitzWith_sphereProjectionFlow_time (w x : SSphere d) :
    LipschitzWith 1 (fun q => sphereProjectionFlow w q x) := by
  apply lipschitzWith_timeDependent_spherical_trajectory_time (fun _ y => proj d y (w : EucSpace d)) 1
  · intro t y
    simpa only [NNReal.coe_one, mem_sphere_zero_iff_norm.mp w.2] using
      norm_proj_le (mem_sphere_zero_iff_norm.mp y.2) (w : EucSpace d)
  · exact hasDerivAt_sphereProjectionFlow w x

/-- A law near an orthogonal stationary pole has small actual axial mean.
Auxiliary for arXiv:2411.04551v3, §3.2, Steps 1 and 3. -/
theorem abs_meanAxisMoment_le_radius (w z : SSphere d) (ν : Perspective.ProbSphere d) (δ : ℝ)
    (hwz : inner ℝ (w : EucSpace d) (z : EucSpace d) = 0)
    (hν : ∀ x ∈ (ν : Measure (SSphere d)).support, dist x z ≤ δ) :
    |meanAxisMoment (w : EucSpace d) ν| ≤ δ := by
  rw [meanAxisMoment_eq_integral, ← Real.norm_eq_abs]
  have hb : ∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)),
      ‖inner ℝ (w : EucSpace d) (x : EucSpace d)‖ ≤ δ := by
    filter_upwards [Measure.support_mem_ae (μ := (ν : Measure (SSphere d)))] with x hx
    have hh := abs_real_inner_le_norm (w : EucSpace d) ((x : EucSpace d) - z)
    rw [inner_sub_right, hwz, sub_zero, mem_sphere_zero_iff_norm.mp w.2, one_mul] at hh
    exact hh.trans (by simpa only [Subtype.dist_eq, dist_eq_norm] using hν x hx)
  simpa only [probReal_univ, mul_one] using norm_integral_le_of_norm_le_const hb

/-- Actual particles near an orthogonal stationary pole obey a uniform
exponential radius bound. Source: arXiv:2411.04551v3, §3.2, Steps 1 and 3. -/
theorem dist_meanAxisFlow_le_radius (w z : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (δ : ℝ) (hwz : inner ℝ (w : EucSpace d) (z : EucSpace d) = 0)
    (hν : ∀ x ∈ (ν : Measure (SSphere d)).support, dist x z ≤ δ)
    (x : SSphere d) (hx : dist x z ≤ δ) (t : ℝ) (ht : 0 ≤ t) :
    dist (meanAxisFlow w κ ν t x) z ≤ δ * exp (|κ| * t) := by
  have hc := (abs_meanAxisClock_le w κ ν t ht).trans
    (mul_le_mul_of_nonneg_right (abs_meanAxisMoment_le_radius w z ν δ hwz hν)
      (sub_nonneg.mpr (one_le_exp (mul_nonneg (abs_nonneg κ) ht))))
  have hd := (lipschitzWith_sphereProjectionFlow_time w x).dist_le_mul (meanAxisClock w κ ν t) 0
  rw [sphereProjectionFlow_zero] at hd
  simp only [NNReal.coe_one, one_mul, Real.dist_eq, sub_zero] at hd
  change dist (sphereProjectionFlow w (meanAxisClock w κ ν t) x) z ≤ _
  have hh := (dist_triangle (sphereProjectionFlow w (meanAxisClock w κ ν t) x) x z).trans (add_le_add hd hx)
  nlinarith

/-- Scaling the original value matrix exactly rescales its nonlinear
clock. Source: arXiv:2411.04551v3, §3.2, prescribed-time phases. -/
theorem meanAxisClock_scale (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) :
    meanAxisClock w κ ν = fun t => meanAxisClock w 1 ν (κ * t) := by
  apply meanAxisClock_eq_of_solution w κ ν _ (by rw [mul_zero, meanAxisClock_zero])
  intro s
  simpa only [Function.comp_def, one_mul, mul_one, smul_eq_mul] using
    (hasDerivAt_meanAxisClock w 1 ν (κ * s)).scomp s ((hasDerivAt_id s).const_mul κ)

/-- Original mean-attention laws obey the exact control time scaling.
Source: arXiv:2411.04551v3, §3.2, prescribed-time phases. -/
theorem meanAxisLaw_scale (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    meanAxisLaw w κ ν t = meanAxisLaw w 1 ν (κ * t) := by
  rw [meanAxisLaw, meanAxisClock_scale]
  rfl

/-- The entire constructed support obeys the actual radius estimate.
Source: arXiv:2411.04551v3, §3.2, Steps 1 and 3. -/
theorem support_meanAxisLaw_radius (w z : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (δ : ℝ) (hwz : inner ℝ (w : EucSpace d) (z : EucSpace d) = 0)
    (hν : ∀ x ∈ (ν : Measure (SSphere d)).support, dist x z ≤ δ) (t : ℝ) (ht : 0 ≤ t) :
    (meanAxisLaw w κ ν t : Measure (SSphere d)).support ⊆ Metric.closedBall z (δ * exp (|κ| * t)) := by
  have hs : (meanAxisLaw w κ ν t : Measure (SSphere d)).support ⊆
      meanAxisFlow w κ ν t '' (ν : Measure (SSphere d)).support := by
    rw [coe_meanAxisLaw]
    exact support_map_subset_image _ _
      ((continuous_meanAxisFlow w κ ν).comp (continuous_const.prodMk continuous_id))
  intro y hy
  obtain ⟨x, hx, rfl⟩ := hs hy
  exact dist_meanAxisFlow_le_radius w z κ ν δ hwz hν x (hν x hx) t ht

/-- Every original weak law obeys the same support-radius estimate.
Source: arXiv:2411.04551v3, §3.2, Steps 1 and 3. -/
theorem cauchyPB_meanAxis_support_radius (w z : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (δ : ℝ) (hwz : inner ℝ (w : EucSpace d) (z : EucSpace d) = 0)
    (h0 : ∀ x ∈ (μ 0 : Measure (SSphere d)).support, dist x z ≤ δ) (t : ℝ) (ht : 0 ≤ t) :
    (μ t : Measure (SSphere d)).support ⊆ Metric.closedBall z (δ * exp (|κ| * t)) := by
  have he := congrFun (cauchyPB_meanAxis_eq w κ μ hμ) t
  rw [he]
  exact support_meanAxisLaw_radius w z κ (μ 0) δ hwz h0 t ht

/-- A transverse Dirac law realizes the orthogonality and small-support
hypotheses, with a positive observation time and an actual reversed weak
control. Its initial point also realizes the particle-radius hypothesis. -/
example : ∃ (w z x : SSphere 2) (ν : Perspective.ProbSphere 2) (δ t : ℝ),
    inner ℝ (w : EucSpace 2) (z : EucSpace 2) = 0 ∧
    (∀ y ∈ (ν : Measure (SSphere 2)).support, dist y z ≤ δ) ∧ dist x z ≤ δ ∧ 0 ≤ t ∧
      cauchyPB 2 (fun _ => meanAxisParams (w : EucSpace 2) (-1)) (meanAxisLaw w (-1) ν) := by
  let w : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  refine ⟨w, z, z, Perspective.diracProb 2 z, 1 / 10, 1, ?_, ?_, ?_, zero_le_one,
    cauchyPB_meanAxisLaw w (-1) _⟩
  · simp [w, z, basePoint, EuclideanSpace.inner_single_left]
  · intro y hy
    have he : y = z := eq_of_mem_support_dirac hy
    rw [he, dist_self]
    norm_num
  · rw [dist_self]
    norm_num

end Transformer.Interpolation
