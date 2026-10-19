module

public import Transformer.Interpolation.Section3_MeanAxisField

/-!
# The scalar clock coefficient for rank-one mean attention

ArXiv:2411.04551v3, proof of `lem: induction.barycenter`, Step 1,
`eq: cauchy.pb.reloaded`. Push the initial law along the actual unit-axis
projection flow for an intrinsic time `q`. Its mean component along the
axis is the coefficient of the scalar clock equation.

The coefficient lies in `[-1,1]`. Its derivative is the integral of
`1 - ⟨w,x⟩²`, hence lies in `[0,1]`; it is monotone and globally
1-Lipschitz. Composing projection flows gives the exact coefficient
translation needed to reverse the original mean-attention phase.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original mean component after intrinsic projection time `q`.
Source: arXiv:2411.04551v3, §3.2, Step 1 of `lem: induction.barycenter`. -/
noncomputable def meanAxisClockSpeed (w : SSphere d) (ν : Perspective.ProbSphere d) (q : ℝ) : ℝ :=
  meanAxisMoment (w : EucSpace d) (projectionLaw w ν q)

/-- The clock coefficient starts at the prescribed initial mean.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem meanAxisClockSpeed_zero (w : SSphere d) (ν : Perspective.ProbSphere d) :
    meanAxisClockSpeed w ν 0 = meanAxisMoment (w : EucSpace d) ν := by
  rw [meanAxisClockSpeed, projectionLaw_zero]

/-- The coefficient is bounded independently of the original law.
Auxiliary for arXiv:2411.04551v3, §3.2, the complete clock construction. -/
theorem abs_meanAxisClockSpeed_le_one (w : SSphere d) (ν : Perspective.ProbSphere d) (q : ℝ) :
    |meanAxisClockSpeed w ν q| ≤ 1 := by
  simpa only [meanAxisClockSpeed, mem_sphere_zero_iff_norm.mp w.2] using
    abs_meanAxisMoment_le (w : EucSpace d) (projectionLaw w ν q)

/-- The actual intrinsic-time derivative is the averaged tangent
mobility. Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem hasDerivAt_meanAxisClockSpeed (w : SSphere d) (ν : Perspective.ProbSphere d) (q : ℝ) :
    HasDerivAt (meanAxisClockSpeed w ν)
      (∫ y : SSphere d, (1 - (inner ℝ (w : EucSpace d) (y : EucSpace d)) ^ 2)
        ∂(projectionLaw w ν q : Measure (SSphere d))) q := by
  have he : meanAxisClockSpeed w ν = fun s => ∫ y : SSphere d,
      inner ℝ (w : EucSpace d) (y : EucSpace d) ∂(projectionLaw w ν s : Measure (SSphere d)) :=
    funext fun s => meanAxisMoment_eq_integral (w : EucSpace d) _
  rw [he]
  have hd := hasDerivAt_projectionLaw_test w ν (fun x : EucSpace d => inner ℝ (w : EucSpace d) x)
    (contDiff_const.inner ℝ contDiff_id) q
  apply hd.congr_deriv
  apply integral_congr_ae
  exact ae_of_all _ fun y => by
    change projectionTestFlux (fun x : EucSpace d => inner ℝ (w : EucSpace d) x)
      (w : EucSpace d) (y : EucSpace d) = _
    rw [projectionTestFlux_eq_inner, gradient_inner_linear, proj, inner_sub_right,
      real_inner_smul_right, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp w.2,
      real_inner_comm (w : EucSpace d) (y : EucSpace d)]
    ring

/-- The actual clock coefficient is continuous.
Auxiliary for arXiv:2411.04551v3, §3.2, characteristics of Step 1. -/
theorem continuous_meanAxisClockSpeed (w : SSphere d) (ν : Perspective.ProbSphere d) :
    Continuous (meanAxisClockSpeed w ν) :=
  continuous_iff_continuousAt.mpr fun q => (hasDerivAt_meanAxisClockSpeed w ν q).continuousAt

/-- The averaged tangent mobility lies in `[0,1]`.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: induction.barycenter`. -/
theorem meanAxisClockSpeed_deriv_bounds (w : SSphere d) (ν : Perspective.ProbSphere d) (q : ℝ) :
    0 ≤ deriv (meanAxisClockSpeed w ν) q ∧ deriv (meanAxisClockSpeed w ν) q ≤ 1 := by
  rw [(hasDerivAt_meanAxisClockSpeed w ν q).deriv]
  have hpoint (y : SSphere d) : 0 ≤ 1 - (inner ℝ (w : EucSpace d) (y : EucSpace d)) ^ 2 ∧
      1 - (inner ℝ (w : EucSpace d) (y : EucSpace d)) ^ 2 ≤ 1 := by
    have hh := abs_real_inner_le_norm (w : EucSpace d) (y : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp w.2, mem_sphere_zero_iff_norm.mp y.2, mul_one] at hh
    have ha := abs_le.mp hh
    constructor <;> nlinarith [sq_nonneg (inner ℝ (w : EucSpace d) (y : EucSpace d))]
  have hi : Integrable (fun y : SSphere d => 1 - (inner ℝ (w : EucSpace d) (y : EucSpace d)) ^ 2)
      (projectionLaw w ν q : Measure (SSphere d)) :=
    integrable_spherical_continuous _ _ (continuous_const.sub ((continuous_const.inner continuous_subtype_val).pow 2))
  exact ⟨integral_nonneg fun y => (hpoint y).1, by
    have hh := integral_mono hi (integrable_const (1 : ℝ)) (fun y => (hpoint y).2)
    simpa only [integral_const, probReal_univ, one_smul] using hh⟩

/-- Positive-time projection increases the original mean component.
Source: arXiv:2411.04551v3, §3.2, the sign argument in Step 1. -/
theorem monotone_meanAxisClockSpeed (w : SSphere d) (ν : Perspective.ProbSphere d) :
    Monotone (meanAxisClockSpeed w ν) :=
  monotone_of_hasDerivAt_nonneg (hasDerivAt_meanAxisClockSpeed w ν)
    (fun q => by rw [← (hasDerivAt_meanAxisClockSpeed w ν q).deriv]; exact (meanAxisClockSpeed_deriv_bounds w ν q).1)

/-- A global Lipschitz modulus for the actual scalar clock coefficient.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: induction.barycenter`. -/
theorem lipschitzWith_meanAxisClockSpeed (w : SSphere d) (ν : Perspective.ProbSphere d) :
    LipschitzWith 1 (meanAxisClockSpeed w ν) := by
  apply lipschitzWith_of_nnnorm_deriv_le (𝕜 := ℝ)
    (fun q => (hasDerivAt_meanAxisClockSpeed w ν q).differentiableAt)
  intro q
  apply NNReal.coe_le_coe.mp
  rw [coe_nnnorm, Real.norm_eq_abs, abs_of_nonneg (meanAxisClockSpeed_deriv_bounds w ν q).1]
  exact (meanAxisClockSpeed_deriv_bounds w ν q).2

/-- The coefficient is C1, by continuity of its actual transported
quadratic test. Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem contDiff_meanAxisClockSpeed (w : SSphere d) (ν : Perspective.ProbSphere d) :
    ContDiff ℝ 1 (meanAxisClockSpeed w ν) := by
  refine contDiff_one_iff_deriv.mpr ⟨fun q => (hasDerivAt_meanAxisClockSpeed w ν q).differentiableAt, ?_⟩
  have he : deriv (meanAxisClockSpeed w ν) = fun q => ∫ y : SSphere d,
      (1 - (inner ℝ (w : EucSpace d) (y : EucSpace d)) ^ 2) ∂(projectionLaw w ν q : Measure (SSphere d)) :=
    funext fun q => (hasDerivAt_meanAxisClockSpeed w ν q).deriv
  rw [he]
  have hf : Continuous (fun x : EucSpace d => 1 - (inner ℝ (w : EucSpace d) x) ^ 2) :=
    continuous_const.sub ((continuous_const.inner continuous_id).pow 2)
  simp_rw [integral_projectionLaw w ν _ hf]
  apply continuous_spherical_integral_of_joint_continuous ν
  exact hf.comp (continuous_subtype_val.comp (continuous_sphereProjectionFlow w))

/-- Actual probability pushforwards inherit the group law of the
projection flow. Source: arXiv:2411.04551v3, §3.2, flow reversal. -/
theorem projectionLaw_add (w : SSphere d) (ν : Perspective.ProbSphere d) (p q : ℝ) :
    projectionLaw w (projectionLaw w ν q) p = projectionLaw w ν (p + q) := by
  apply Subtype.ext
  change (projectionLaw w (projectionLaw w ν q) p : Measure (SSphere d)) =
    (projectionLaw w ν (p + q) : Measure (SSphere d))
  simp only [coe_projectionLaw]
  have hm (s : ℝ) : Measurable (sphereProjectionFlow w s) :=
    ((continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id)).measurable
  rw [Measure.map_map (hm p) (hm q)]
  congr 1
  exact funext fun x => sphereProjectionFlow_add w x p q

/-- Transporting the initial law translates the actual clock
coefficient. Source: arXiv:2411.04551v3, §3.2, flow reversal. -/
theorem meanAxisClockSpeed_shift (w : SSphere d) (ν : Perspective.ProbSphere d) (p q : ℝ) :
    meanAxisClockSpeed w (projectionLaw w ν q) p = meanAxisClockSpeed w ν (p + q) := by
  simp only [meanAxisClockSpeed, projectionLaw_add]

/-- A transverse initial atom gives a genuinely varying coefficient:
its initial value is zero and its initial derivative is one. -/
example : ∃ (w : SSphere 2) (ν : Perspective.ProbSphere 2),
    meanAxisClockSpeed w ν 0 = 0 ∧ deriv (meanAxisClockSpeed w ν) 0 = 1 := by
  let w : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let ν := Perspective.diracProb 2 z
  have hwz : inner ℝ (w : EucSpace 2) (z : EucSpace 2) = 0 := by
    simp [w, z, basePoint, EuclideanSpace.inner_single_left]
  refine ⟨w, ν, ?_, ?_⟩
  · rw [meanAxisClockSpeed_zero, meanAxisMoment_eq_integral]
    change (∫ y : SSphere 2, inner ℝ (w : EucSpace 2) (y : EucSpace 2) ∂Measure.dirac z) = 0
    rw [integral_dirac, hwz]
  · rw [(hasDerivAt_meanAxisClockSpeed w ν 0).deriv, projectionLaw_zero]
    change (∫ y : SSphere 2, (1 - (inner ℝ (w : EucSpace 2) (y : EucSpace 2)) ^ 2) ∂Measure.dirac z) = 1
    rw [integral_dirac, hwz]
    norm_num

end Transformer.Interpolation
