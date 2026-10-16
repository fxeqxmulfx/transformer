/-
# Uniform original attention gathering on fixed compact support regions

ArXiv:2411.04551v3, §3.2, Step 1 of selective isolation,
and §5, the bounded gathering cost in atomic-target interpolation.
The projection contraction estimate is uniform on an actual compact
set missing the repelling pole, for all sufficiently large clocks.
A fixed positive lower bound on the original axial mean then gives
one physical gathering time for every law supported in that set.

Applying this to both orientations of an axis gives a single time
for every such law whose absolute initial axial mean stays away from
zero. The actual nonlinear clock selects one of the two attracting
poles. The choice of time is independent of the particular law and
any later concentration tolerance used by the selected-law contraction.
-/

import Transformer.Interpolation.Section3_MeanAxisAttraction
import Transformer.Interpolation.Section3_PositiveBarycenters
import Transformer.Interpolation.Section1_RidgeParameterBounds

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- All sufficiently large original projection clocks carry an
entire fixed compact set into its attracting pole ball. Source:
arXiv:2411.04551v3, §3.1 and §3.2, uniform projection contraction. -/
theorem eventually_projection_ball_on_compact (w : SSphere d) (K : Set (SSphere d))
    (hK : IsCompact K) (hmiss : antipode d w ∉ K) (ε : ℝ) (hε : 0 < ε) :
    ∀ᶠ t : ℝ in atTop, ∀ x ∈ K, dist (sphereProjectionFlow w t x) w < ε := by
  obtain ⟨c, hc, hcx⟩ := exists_height_floor_of_compact w K hK hmiss
  have htend : Tendsto (fun t : ℝ => 8 / (1 + c) * exp (-t) ^ 2) atTop (nhds 0) := by
    simpa only [zero_pow (by decide : 2 ≠ 0), mul_zero] using
      (tendsto_exp_neg_atTop_nhds_zero.pow 2).const_mul (8 / (1 + c))
  filter_upwards [htend.eventually (gt_mem_nhds (sq_pos_of_pos hε))] with t ht
  intro x hx
  have hb := (projectionFlow_distance_sq_le (w : EucSpace d) (x : EucSpace d)
    (mem_sphere_zero_iff_norm.mp w.2) (mem_sphere_zero_iff_norm.mp x.2) c hc (hcx x hx) t).trans_lt ht
  change dist (projectionFlow (w : EucSpace d) t (x : EucSpace d)) (w : EucSpace d) < ε
  rw [dist_eq_norm]
  nlinarith [norm_nonneg (projectionFlow (w : EucSpace d) t (x : EucSpace d) - (w : EucSpace d))]

/-- One positive original attention time works for every law in a
fixed compact support region with the same positive axial-mean floor.
Source auxiliary: arXiv:2411.04551v3, §3.2 and §5, uniform isolation cost. -/
theorem exists_uniform_positive_meanAxis_time (w : SSphere d) (K : Set (SSphere d))
    (hK : IsCompact K) (hmiss : antipode d w ∉ K) (c ε : ℝ) (hc : 0 < c) (hε : 0 < ε) :
    ∃ T : ℝ, 0 < T ∧ ∀ ν : Perspective.ProbSphere d,
      (ν : Measure (SSphere d)).support ⊆ K → c ≤ meanAxisMoment (w : EucSpace d) ν →
      ∀ t : ℝ, T ≤ t → (meanAxisLaw w 1 ν t : Measure (SSphere d)).support ⊆ Metric.ball w ε := by
  obtain ⟨Q, hQ⟩ := eventually_atTop.mp (eventually_projection_ball_on_compact w K hK hmiss ε hε)
  let T := (max Q 0 + 1) / c
  have hT : 0 < T := by dsimp only [T]; positivity
  have hclock : c * T = max Q 0 + 1 := by dsimp only [T]; field_simp
  refine ⟨T, hT, ?_⟩
  intro ν hν hmean t ht
  have ht0 : 0 ≤ t := hT.le.trans ht
  have hmean0 : 0 < meanAxisMoment (w : EucSpace d) ν := hc.trans_le hmean
  have hlo := meanAxisClock_lower_bound w 1 ν zero_le_one hmean0 t ht0
  have hct : c * T ≤ c * t := mul_le_mul_of_nonneg_left ht hc.le
  have hmt : c * t ≤ meanAxisMoment (w : EucSpace d) ν * t :=
    mul_le_mul_of_nonneg_right hmean ht0
  have hq : Q ≤ meanAxisClock w 1 ν t := by
    simp only [one_mul] at hlo
    have hm := le_max_left Q 0
    linarith
  intro y hy
  rw [meanAxisLaw, coe_projectionLaw] at hy
  have hcont : Continuous (sphereProjectionFlow w (meanAxisClock w 1 ν t)) :=
    (continuous_sphereProjectionFlow w).comp (continuous_const.prodMk continuous_id)
  obtain ⟨x, hx, rfl⟩ := support_map_subset_image _ _ hcont hy
  exact hQ _ hq x (hν hx)

/-- One positive gathering time works for all laws in a fixed compact
region with a common nonzero absolute mean floor. Source auxiliary:
arXiv:2411.04551v3, §3.2, `eq: supports.in.balls`, and §5. -/
theorem exists_uniform_meanAxis_two_pole_time (w : SSphere d) (K : Set (SSphere d))
    (hK : IsCompact K) (hw : w ∉ K) (hwa : antipode d w ∉ K)
    (c ε : ℝ) (hc : 0 < c) (hε : 0 < ε) :
    ∃ T : ℝ, 0 < T ∧ ∀ ν : Perspective.ProbSphere d,
      (ν : Measure (SSphere d)).support ⊆ K → c ≤ |meanAxisMoment (w : EucSpace d) ν| →
      ∀ t : ℝ, T ≤ t → (meanAxisLaw w 1 ν t : Measure (SSphere d)).support ⊆
        Metric.ball w ε ∪ Metric.ball (antipode d w) ε := by
  obtain ⟨Tpos, hTpos, hpos⟩ := exists_uniform_positive_meanAxis_time w K hK hwa c ε hc hε
  obtain ⟨Tneg, _, hneg⟩ := exists_uniform_positive_meanAxis_time (antipode d w) K hK
    (by simpa only [antipode_antipode] using hw) c ε hc hε
  refine ⟨max Tpos Tneg, hTpos.trans_le (le_max_left _ _), ?_⟩
  intro ν hν hm t ht y hy
  by_cases hmean : 0 ≤ meanAxisMoment (w : EucSpace d) ν
  · have hfloor : c ≤ meanAxisMoment (w : EucSpace d) ν := by
      simpa only [abs_of_nonneg hmean] using hm
    exact Or.inl (hpos ν hν hfloor t ((le_max_left _ _).trans ht) hy)
  · have hn : meanAxisMoment (w : EucSpace d) ν < 0 := lt_of_not_ge hmean
    have hfloor : c ≤ meanAxisMoment (antipode d w : EucSpace d) ν := by
      change c ≤ inner ℝ (-(w : EucSpace d)) (barycenter d ν)
      rw [inner_neg_left]
      have hm' : c ≤ -meanAxisMoment (w : EucSpace d) ν := by
        simpa only [abs_of_neg hn] using hm
      simpa only [meanAxisMoment] using hm'
    have hs := hneg ν hν hfloor t ((le_max_right _ _).trans ht)
    rw [meanAxisLaw_antipode_axis] at hs
    exact Or.inr (hs hy)

/-- Uniform original gathering can be performed at every positive
physical horizon with one constant attention piece and a common
data-dependent clock divided by that horizon. Source auxiliary:
arXiv:2411.04551v3, §3.2 and §5, bounded gathering before contraction. -/
theorem exists_uniform_meanAxis_gathering_controls (w : SSphere d) (K : Set (SSphere d))
    (hK : IsCompact K) (hw : w ∉ K) (hwa : antipode d w ∉ K)
    (c ε : ℝ) (hc : 0 < c) (hε : 0 < ε) :
    ∃ τ : ℝ, 0 < τ ∧ ∀ H : ℝ, 0 < H → ∃ θ : TimeParams d,
      PiecewiseConstant d θ H 1 ∧ (∀ s, (θ s).B = 0) ∧
      (∀ s, ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖ ≤ τ / H) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ ν : Perspective.ProbSphere d,
        (ν : Measure (SSphere d)).support ⊆ K → c ≤ |meanAxisMoment (w : EucSpace d) ν| →
        ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d θ μ →
          (μ H : Measure (SSphere d)).support ⊆ Metric.ball w ε ∪ Metric.ball (antipode d w) ε := by
  obtain ⟨τ, hτ, hbound⟩ := exists_uniform_meanAxis_two_pole_time w K hK hw hwa c ε hc hε
  refine ⟨τ, hτ, ?_⟩
  intro H hH
  let κ := τ / H
  have hκ : 0 ≤ κ := div_nonneg hτ.le hH.le
  let θ : TimeParams d := fun _ => meanAxisParams (w : EucSpace d) κ
  refine ⟨θ, piecewiseConstant_constant _ H hH.le, (fun _ => rfl), ?_, ?_, ?_⟩
  · intro s
    simpa only [abs_of_nonneg hκ] using norm_meanAxisParams_sum_le w κ
  · exact fun ρ => exists_cauchyPB_meanAxis w κ ρ
  · intro ν hν hm μ h0 hμ
    have he := congrFun (cauchyPB_meanAxis_eq w κ μ hμ) H
    rw [meanAxisLaw_scale, h0] at he
    have hclock : κ * H = τ := div_mul_cancel₀ τ hH.ne'
    rw [hclock] at he
    rw [he]
    exact hbound ν hν hm τ le_rfl

/-- Two genuine off-axis circle atoms, one with positive and one
with negative axial mean, inhabit the same compact support class and
absolute mean floor. Both poles are excluded by a nonzero second coordinate. -/
example : ∃ (w : SSphere 2) (K : Set (SSphere 2)) (ν : Fin 2 → Perspective.ProbSphere 2),
    IsCompact K ∧ w ∉ K ∧ antipode 2 w ∉ K ∧
      (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧
      (∀ k, (ν k : Measure (SSphere 2)).support ⊆ K) ∧
      (∀ k, (1 / 2 : ℝ) ≤ |meanAxisMoment (w : EucSpace 2) (ν k)|) ∧
      (1 / 2 : ℝ) ≤ meanAxisMoment (w : EucSpace 2) (ν 0) := by
  let w := basePoint 1
  let v (k : Fin 2) : SSphere 2 := circlePt (if k = 0 then 3 / 5 else -3 / 5) (4 / 5)
    (by fin_cases k <;> norm_num)
  let K := Set.range v
  let ν (k : Fin 2) := Perspective.diracProb 2 (v k)
  have hmiss (y : SSphere 2) (hy : (y : EucSpace 2) 1 = 0) : y ∉ K := by
    rintro ⟨k, hk⟩
    have hh : (y : EucSpace 2) 1 = 4 / 5 := by rw [← hk]; rfl
    rw [hy] at hh
    norm_num at hh
  refine ⟨w, K, ν, (Set.finite_range v).isCompact, ?_, ?_, by norm_num, one_pos, ?_, ?_, ?_⟩
  · apply hmiss
    simp [w, basePoint]
  · apply hmiss
    simp [w, basePoint, antipode]
  · intro k x hx
    rw [eq_of_mem_support_dirac hx]
    exact ⟨k, rfl⟩
  · intro k
    fin_cases k <;>
      norm_num [ν, meanAxisMoment, barycenter, Perspective.diracProb, w, v, circlePt,
        basePoint, EuclideanSpace.inner_single_left]
  · norm_num [ν, meanAxisMoment, barycenter, Perspective.diracProb, w, v, circlePt,
      basePoint, EuclideanSpace.inner_single_left]

end Transformer.Interpolation
