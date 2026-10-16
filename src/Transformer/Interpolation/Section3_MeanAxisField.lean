/-
# The original rank-one mean-attention field

ArXiv:2411.04551v3, §3.2, `lem: induction.barycenter`, proof Step 1,
`eq: cauchy.pb.reloaded`, and §1, `eq: average.vf`.
With zero scores, no perceptron and value matrix `κ w wᵀ`, the actual
field is `κ ⟨w, mean(μ)⟩ Proj_x w`. Its scalar moment is the original
linear test integral. A unit axis bounds the speed by `|κ|`, uniformly
over all laws, so the actual weak equation gives continuity of every
continuous test integral and of the moment coefficient itself.

The factor `κ` includes the source's forward phase, its reverse phase,
and the constant rescaling needed for a prescribed horizon.
-/

import Transformer.Interpolation.Section3_ProjectionLaw
import Transformer.Interpolation.Section1_AutonomousWeakTests
import Transformer.Interpolation.Clustering

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The actual barycenter component used in `eq: cauchy.pb.reloaded`,
arXiv:2411.04551v3, proof of `lem: induction.barycenter`, Step 1. -/
noncomputable def meanAxisMoment (w : EucSpace d) (ν : Perspective.ProbSphere d) : ℝ :=
  inner ℝ w (barycenter d ν)

/-- Original constant controls with `V = κ w wᵀ`, `B = W = U = b = 0`.
Source: arXiv:2411.04551v3, proof of `lem: induction.barycenter`, Step 1. -/
noncomputable def meanAxisParams (w : EucSpace d) (κ : ℝ) : Params d :=
  { V := κ • (innerSL ℝ w).smulRight w
    B := 0
    W := 0
    U := 0
    b := 0 }

/-- The source's mean component is exactly the linear weak-test moment.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem meanAxisMoment_eq_integral (w : EucSpace d) (ν : Perspective.ProbSphere d) :
    meanAxisMoment w ν = ∫ y : SSphere d, inner ℝ w (y : EucSpace d) ∂(ν : Measure (SSphere d)) := by
  have hi : Integrable (fun y : SSphere d => (y : EucSpace d)) (ν : Measure (SSphere d)) :=
    continuous_subtype_val.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  exact (integral_inner (𝕜 := ℝ) hi w).symm

/-- A probability on the unit sphere bounds the actual mean component.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: induction.barycenter`. -/
theorem abs_meanAxisMoment_le (w : EucSpace d) (ν : Perspective.ProbSphere d) :
    |meanAxisMoment w ν| ≤ ‖w‖ := by
  rw [meanAxisMoment_eq_integral, ← Real.norm_eq_abs]
  have hbound (y : SSphere d) : ‖inner ℝ w (y : EucSpace d)‖ ≤ ‖w‖ := by
    rw [Real.norm_eq_abs]
    simpa only [mem_sphere_zero_iff_norm.mp y.2, mul_one] using
      abs_real_inner_le_norm w (y : EucSpace d)
  simpa only [probReal_univ, mul_one] using
    norm_integral_le_of_norm_le_const (μ := (ν : Measure (SSphere d))) (ae_of_all _ hbound)

/-- Exact specialization of the original full field to rank-one values.
Source: arXiv:2411.04551v3, §3.2, `eq: cauchy.pb.reloaded`. -/
theorem fullVF_meanAxisParams (w : EucSpace d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) :
    fullVF d (fun _ => meanAxisParams w κ) ν t x =
      (κ * meanAxisMoment w ν) • proj d x w := by
  simp only [fullVF, meanAxisParams, zero_apply, inner_zero_left, exp_zero, integral_const,
    probReal_univ, one_smul, inv_one, smul_apply, ContinuousLinearMap.smulRight_apply,
    coe_innerSL_apply, add_zero]
  simp only [barycenter, meanAxisMoment, proj, real_inner_smul_right, smul_sub, smul_smul]

/-- The rank-one original field has a uniform spherical speed bound.
Auxiliary for arXiv:2411.04551v3, §3.2, the time rescaling of Step 1. -/
theorem norm_fullVF_meanAxisParams_le (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : SSphere d) :
    ‖fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) ν t x‖ ≤ |κ| := by
  rw [fullVF_meanAxisParams, norm_smul, Real.norm_eq_abs, abs_mul]
  have hm : |meanAxisMoment (w : EucSpace d) ν| ≤ 1 := by
    simpa only [mem_sphere_zero_iff_norm.mp w.2] using abs_meanAxisMoment_le (w : EucSpace d) ν
  have hp : ‖proj d (x : EucSpace d) (w : EucSpace d)‖ ≤ 1 := by
    simpa only [mem_sphere_zero_iff_norm.mp w.2] using norm_proj_le (mem_sphere_zero_iff_norm.mp x.2) (w : EucSpace d)
  calc
    _ ≤ (|κ| * 1) * 1 := mul_le_mul (mul_le_mul_of_nonneg_left hm (abs_nonneg κ)) hp
      (norm_nonneg _) (by positivity)
    _ = _ := by ring

/-- The original weak equation gives weak continuity along any actual
rank-one mean-attention law. Source: arXiv:2411.04551v3, §1 and §3.2. -/
theorem continuous_meanAxis_cauchyPB_test (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ)
    (f : EucSpace d → ℝ) (hf : Continuous f) :
    Continuous (fun t => ∫ x : SSphere d, f x ∂(μ t : Measure (SSphere d))) :=
  continuous_cauchyPB_bounded_test _ |κ| (abs_nonneg κ)
    (fun ν t x => norm_fullVF_meanAxisParams_le w κ ν t x) μ hμ f hf

/-- The actual mean coefficient is continuous in time, without assuming
law transport. Auxiliary for arXiv:2411.04551v3, §3.2, Step 1. -/
theorem continuous_meanAxisMoment_of_cauchyPB (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) :
    Continuous (fun t => meanAxisMoment (w : EucSpace d) (μ t)) := by
  simp_rw [meanAxisMoment_eq_integral]
  exact continuous_meanAxis_cauchyPB_test w κ μ hμ _ (continuous_const.inner continuous_id)

/-- The original field along a given weak law is jointly continuous
at every ambient query. Auxiliary for arXiv:2411.04551v3, §3.2. -/
theorem continuous_fullVF_meanAxis_joint (w : SSphere d) (κ : ℝ) (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (fun _ => meanAxisParams (w : EucSpace d) κ) μ) :
    Continuous (fun q : ℝ × EucSpace d =>
      fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) (μ q.1) q.1 q.2) := by
  simp_rw [fullVF_meanAxisParams, proj]
  have hm : Continuous (fun q : ℝ × EucSpace d => meanAxisMoment (w : EucSpace d) (μ q.1)) :=
    (continuous_meanAxisMoment_of_cauchyPB w κ μ hμ).comp continuous_fst
  have hp : Continuous (fun q : ℝ × EucSpace d => (w : EucSpace d) - inner ℝ q.2 (w : EucSpace d) • q.2) :=
    continuous_const.sub ((continuous_snd.inner (𝕜 := ℝ) continuous_const).smul continuous_snd)
  exact (continuous_const.mul hm).smul hp

/-- The current original rank-one field has a uniform spatial Lipschitz
constant on the sphere. Auxiliary for arXiv:2411.04551v3, §3.2. -/
theorem lipschitzOnWith_fullVF_meanAxis (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    LipschitzOnWith ⟨2 * |κ|, by positivity⟩
      (fullVF d (fun _ => meanAxisParams (w : EucSpace d) κ) ν t) (Sphere d) := by
  apply lipschitzOnWith_iff_norm_sub_le.mpr
  intro x hx y hy
  rw [fullVF_meanAxisParams, fullVF_meanAxisParams, ← smul_sub, norm_smul, Real.norm_eq_abs, abs_mul]
  change |κ| * |meanAxisMoment (w : EucSpace d) ν| * ‖proj d x w - proj d y w‖ ≤ (2 * |κ|) * ‖x - y‖
  have hm : |meanAxisMoment (w : EucSpace d) ν| ≤ 1 := by
    simpa only [mem_sphere_zero_iff_norm.mp w.2] using abs_meanAxisMoment_le (w : EucSpace d) ν
  have hp := norm_proj_sub_proj_le (mem_sphere_zero_iff_norm.mp hx) (mem_sphere_zero_iff_norm.mp hy)
    (mem_sphere_zero_iff_norm.mp w.2).le (le_refl ‖x - y‖) (show ‖(w : EucSpace d) - w‖ ≤ (0 : ℝ) by simp)
  calc
    _ ≤ (|κ| * 1) * (0 + 2 * ‖x - y‖) :=
      mul_le_mul (mul_le_mul_of_nonneg_left hm (abs_nonneg κ)) hp (norm_nonneg _) (by positivity)
    _ = _ := by ring

/-- A unit axis and its genuine stationary Dirac weak law jointly
satisfy the original equation, with a nonzero value matrix and a
nonconstant linear test. No existence assumption is used. -/
example : ∃ (w : SSphere 2) (μ : ℝ → Perspective.ProbSphere 2) (f : EucSpace 2 → ℝ),
    cauchyPB 2 (fun _ => meanAxisParams (w : EucSpace 2) 1) μ ∧ Continuous f := by
  let p : SSphere 2 := basePoint 1
  let μ : ℝ → Perspective.ProbSphere 2 := fun _ => Perspective.diracProb 2 p
  have hh : fullVF 2 (fun _ => meanAxisParams (p : EucSpace 2) 1) (Perspective.diracProb 2 p) 0 p = 0 := by
    rw [fullVF_meanAxisParams]
    have hp := proj_smul_self (mem_sphere_zero_iff_norm.mp p.2) (1 : ℝ)
    simpa only [one_smul, smul_zero] using congrArg (fun v => (1 * meanAxisMoment (p : EucSpace 2) (Perspective.diracProb 2 p)) • v) hp
  refine ⟨p, μ, (fun x => inner ℝ (p : EucSpace 2) x), ?_, continuous_const.inner continuous_id⟩
  intro φ _ t
  have hflux : (fun s => ∫ y : SSphere 2, inner ℝ (gradient φ (y : EucSpace 2))
      (fullVF 2 (fun _ => meanAxisParams (p : EucSpace 2) 1) (μ s) s y) ∂(μ s : Measure (SSphere 2))) = fun _ => 0 := by
    funext s
    change (∫ y : SSphere 2, inner ℝ (gradient φ (y : EucSpace 2))
      (fullVF 2 (fun _ => meanAxisParams (p : EucSpace 2) 1) (Perspective.diracProb 2 p) 0 y) ∂Measure.dirac p) = 0
    rw [integral_dirac, hh, inner_zero_right]
  rw [hflux]
  exact ⟨intervalIntegrable_const, by simp [μ]⟩

end Transformer.Interpolation
