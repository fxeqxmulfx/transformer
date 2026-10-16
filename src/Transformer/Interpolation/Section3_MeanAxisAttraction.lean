/-
# Uniform attraction of original rank-one mean-attention laws

ArXiv:2411.04551v3, §3.2, Step 1 of `lem: induction.barycenter`,
`eq: cauchy.pb.reloaded` and `eq: supports.in.balls`.
A positive initial axial mean makes the actual intrinsic clock tend to
positive infinity. Uniform projection contraction then attracts every
compact initial set missing the repelling pole to the attracting pole.

Negating the unit axis leaves the original rank-one value matrix and
weak law unchanged. A negative initial axial mean is therefore attracted
to the opposite pole. For every nonzero initial axial mean, the entire
original support eventually lies in the union of the two prescribed pole
balls, whenever the initial support misses both poles. The source's
positive quadrant and transverse axes supply precisely this exclusion.
-/

import Transformer.Interpolation.Section3_MeanAxisStability
import Transformer.Interpolation.Section3_ProjectionContraction

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Opposite axes give exactly the same original rank-one controls.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisParams_neg_axis (w : EucSpace d) (κ : ℝ) :
    meanAxisParams (-w) κ = meanAxisParams w κ := by
  unfold meanAxisParams
  congr 1
  ext x
  simp only [smul_apply, ContinuousLinearMap.smulRight_apply, coe_innerSL_apply,
    inner_neg_left, neg_smul, smul_neg, neg_neg]

/-- Opposite axes give the same actual original nonlinear weak law.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisLaw_antipode_axis (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d) (t : ℝ) :
    meanAxisLaw (antipode d w) κ ν t = meanAxisLaw w κ ν t := by
  have hμ := cauchyPB_meanAxisLaw (antipode d w) κ ν
  change cauchyPB d (fun _ => meanAxisParams (-(w : EucSpace d)) κ) _ at hμ
  rw [meanAxisParams_neg_axis] at hμ
  have he := cauchyPB_meanAxis_eq w κ _ hμ
  simpa only [meanAxisLaw_zero] using congrFun he t

/-- A positive initial mean and positive control drive the actual clock
to infinity. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem tendsto_meanAxisClock_atTop (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 < κ) (h0 : 0 < meanAxisMoment (w : EucSpace d) ν) :
    Tendsto (meanAxisClock w κ ν) atTop atTop := by
  apply tendsto_atTop_mono' atTop _ ((tendsto_const_mul_atTop_of_pos (mul_pos hκ h0)).2 tendsto_id)
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with t ht
  exact meanAxisClock_lower_bound w κ ν hκ.le h0 t ht

/-- Actual nonlinear characteristics contract uniformly on compact sets
missing the repeller. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem eventually_meanAxisFlow_ball (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 < κ) (h0 : 0 < meanAxisMoment (w : EucSpace d) ν)
    (K : Set (SSphere d)) (hK : IsCompact K) (hmiss : antipode d w ∉ K)
    (ε : ℝ) (hε : 0 < ε) :
    ∀ᶠ t : ℝ in atTop, ∀ x ∈ K, dist (meanAxisFlow w κ ν t x) w < ε := by
  obtain ⟨c, hc, hcx⟩ := exists_height_floor_of_compact w K hK hmiss
  have hq := tendsto_meanAxisClock_atTop w κ ν hκ h0
  have htend : Tendsto (fun t : ℝ => 8 / (1 + c) * exp (-meanAxisClock w κ ν t) ^ 2) atTop (nhds 0) := by
    simpa only [Function.comp_def, zero_pow (by decide : 2 ≠ 0), mul_zero] using
      ((tendsto_exp_neg_atTop_nhds_zero.comp hq).pow 2).const_mul (8 / (1 + c))
  filter_upwards [htend.eventually (gt_mem_nhds (sq_pos_of_pos hε))] with t ht
  intro x hx
  have hb := (projectionFlow_distance_sq_le (w : EucSpace d) (x : EucSpace d)
    (mem_sphere_zero_iff_norm.mp w.2) (mem_sphere_zero_iff_norm.mp x.2)
    c hc (hcx x hx) (meanAxisClock w κ ν t)).trans_lt ht
  change dist (projectionFlow (w : EucSpace d) (meanAxisClock w κ ν t) (x : EucSpace d)) (w : EucSpace d) < ε
  rw [dist_eq_norm]
  nlinarith [norm_nonneg (projectionFlow (w : EucSpace d) (meanAxisClock w κ ν t) (x : EucSpace d) - (w : EucSpace d))]

/-- Every original support is eventually inside the attracting pole ball.
Source: arXiv:2411.04551v3, §3.2, Step 1, `eq: supports.in.balls`. -/
theorem eventually_meanAxisLaw_ball (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 < κ) (h0 : 0 < meanAxisMoment (w : EucSpace d) ν)
    (hmiss : antipode d w ∉ (ν : Measure (SSphere d)).support) (ε : ℝ) (hε : 0 < ε) :
    ∀ᶠ t : ℝ in atTop, (meanAxisLaw w κ ν t : Measure (SSphere d)).support ⊆ Metric.ball w ε := by
  have he := eventually_meanAxisFlow_ball w κ ν hκ h0 _ (Measure.isClosed_support (μ := (ν : Measure (SSphere d)))).isCompact hmiss ε hε
  filter_upwards [he] with t ht
  have hs : (meanAxisLaw w κ ν t : Measure (SSphere d)).support ⊆
      meanAxisFlow w κ ν t '' (ν : Measure (SSphere d)).support := by
    rw [coe_meanAxisLaw]
    exact support_map_subset_image _ _ ((continuous_meanAxisFlow w κ ν).comp (continuous_const.prodMk continuous_id))
  intro y hy
  obtain ⟨x, hx, rfl⟩ := hs hy
  exact ht x hx

/-- The sign of the actual initial mean selects one attracting pole for
the entire law. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem eventually_meanAxisLaw_one_pole (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 < κ) (h0 : meanAxisMoment (w : EucSpace d) ν ≠ 0)
    (hw : w ∉ (ν : Measure (SSphere d)).support)
    (hwa : antipode d w ∉ (ν : Measure (SSphere d)).support) (ε : ℝ) (hε : 0 < ε) :
    ∃ z : SSphere d, (z = w ∨ z = antipode d w) ∧
      ∀ᶠ t : ℝ in atTop, (meanAxisLaw w κ ν t : Measure (SSphere d)).support ⊆ Metric.ball z ε := by
  rcases lt_or_gt_of_ne h0 with hn | hp
  · refine ⟨antipode d w, Or.inr rfl, ?_⟩
    have hpos : 0 < meanAxisMoment (antipode d w : EucSpace d) ν := by
      change 0 < inner ℝ (-(w : EucSpace d)) (barycenter d ν)
      rw [inner_neg_left]
      exact neg_pos.mpr hn
    have ha : antipode d (antipode d w) = w := Subtype.ext (neg_neg _)
    have he := eventually_meanAxisLaw_ball (antipode d w) κ ν hκ hpos (by rwa [ha]) ε hε
    simpa only [meanAxisLaw_antipode_axis] using he
  · exact ⟨w, Or.inl rfl, eventually_meanAxisLaw_ball w κ ν hκ hp hwa ε hε⟩

/-- The whole original support eventually lies in the prescribed union
of pole balls. Source: arXiv:2411.04551v3, §3.2, `eq: supports.in.balls`. -/
theorem eventually_meanAxisLaw_two_poles (w : SSphere d) (κ : ℝ) (ν : Perspective.ProbSphere d)
    (hκ : 0 < κ) (h0 : meanAxisMoment (w : EucSpace d) ν ≠ 0)
    (hw : w ∉ (ν : Measure (SSphere d)).support)
    (hwa : antipode d w ∉ (ν : Measure (SSphere d)).support) (ε : ℝ) (hε : 0 < ε) :
    ∀ᶠ t : ℝ in atTop, (meanAxisLaw w κ ν t : Measure (SSphere d)).support ⊆
      Metric.ball w ε ∪ Metric.ball (antipode d w) ε := by
  obtain ⟨z, hz, he⟩ := eventually_meanAxisLaw_one_pole w κ ν hκ h0 hw hwa ε hε
  filter_upwards [he] with t ht
  intro x hx
  rcases hz with rfl | rfl
  · exact Or.inl (ht hx)
  · exact Or.inr (ht hx)

/-- An actual off-axis atom on the circle has a positive axial mean,
compact support missing both poles, and positive control and accuracy.
Its characteristic is nonstationary, so attraction has genuine content. -/
example : ∃ (w : SSphere 2) (ν : Perspective.ProbSphere 2) (K : Set (SSphere 2)),
    (0 : ℝ) < 1 ∧ 0 < meanAxisMoment (w : EucSpace 2) ν ∧ IsCompact K ∧
    antipode 2 w ∉ K ∧ w ∉ (ν : Measure (SSphere 2)).support ∧
      antipode 2 w ∉ (ν : Measure (SSphere 2)).support := by
  let w : SSphere 2 := basePoint 1
  let z : SSphere 2 := ⟨EuclideanSpace.single 0 (3 / 5 : ℝ) + EuclideanSpace.single 1 (4 / 5 : ℝ), by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]
    norm_num⟩
  let ν := Perspective.diracProb 2 z
  have hmiss (y : SSphere 2) (hy : (y : EucSpace 2) 1 = 0) :
      y ∉ (ν : Measure (SSphere 2)).support := by
    intro hh
    have he : y = z := eq_of_mem_support_dirac hh
    have hc := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 1) he
    norm_num [z, hy] at hc
  have hw : (w : EucSpace 2) 1 = 0 := by simp [w, basePoint]
  have hwa : (antipode 2 w : EucSpace 2) 1 = 0 := by simp [antipode, hw]
  refine ⟨w, ν, (ν : Measure (SSphere 2)).support, zero_lt_one, ?_,
    Measure.isClosed_support.isCompact, hmiss _ hwa, hmiss _ hw, hmiss _ hwa⟩
  rw [meanAxisMoment_eq_integral]
  change (∫ x : SSphere 2, inner ℝ (w : EucSpace 2) (x : EucSpace 2) ∂Measure.dirac z) > 0
  rw [integral_dirac]
  norm_num [w, z, basePoint, inner_add_right, EuclideanSpace.inner_single_left]

end Transformer.Interpolation
