module

public import Transformer.Interpolation.Section3_CapCenter
public import Transformer.Interpolation.Section3_TwoRidgeLaw
public import Transformer.Interpolation.Section3_ProjectionTransport
public import Transformer.Interpolation.Section3_TwoRidgeScaling

/-!
# Actual gated center laws for the barycenter induction

ArXiv:2411.04551v3, §3.2, Step 2 of `lem: induction.barycenter`,
`eq: inclusion`. A single ridge pointing at the target pole contracts
all initial laws whose supports have a common positive activation margin.
The source's original two-neuron controls contain this ridge exactly when
the second output is zero; equality follows from actual particle uniqueness.

The uniform exponential estimate chooses one positive contraction time
before every initial law satisfying the same height floor. Inactive
supports remain exactly fixed. Small balls about poles orthogonal to the
target lie in that inactive region. Output scaling retains the actual
phase clock used by the three-phase control construction.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A zero second output gives the actual single-ridge particle flow.
Source: arXiv:2411.04551v3, §1 and §3.2, the perceptron phase. -/
theorem twoRidgeFlow_zero_second (p u : EucSpace d) (b : ℝ) (v : EucSpace d) (c : ℝ) (x : SSphere d) :
    twoRidgeFlow p u b 0 v c x = ridgeFlow p u b x := by
  apply ridgeTrajectory_unique p u b _ _ ?_ (hasDerivAt_ridgeFlow p u b x)
    ((twoRidgeFlow_zero p u b 0 v c x).trans (ridgeFlow_zero p u b x).symm)
  intro t
  simpa only [twoRidgeField, proj, inner_zero_right, zero_smul, smul_zero, sub_zero, add_zero] using
    hasDerivAt_twoRidgeFlow p u b 0 v c x t

/-- The original two-neuron probability law is the actual single-ridge
weak law when its second output vanishes. Source: arXiv:2411.04551v3, §3.2. -/
theorem twoRidgeLaw_zero_second (p u : EucSpace d) (b : ℝ) (v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) : twoRidgeLaw p u b 0 v c ν t = ridgeLaw p u b ν t := by
  apply Subtype.ext
  change (twoRidgeLaw p u b 0 v c ν t : Measure (SSphere d)) = (ridgeLaw p u b ν t : Measure (SSphere d))
  simp only [coe_twoRidgeLaw, coe_ridgeLaw, twoRidgeFlow_zero_second]

/-- An original probability supported where the ridge is inactive stays
exactly fixed. Source: arXiv:2411.04551v3, §3.2, Step 2. -/
theorem ridgeLaw_inactive_support (p u : EucSpace d) (b : ℝ) (ν : Perspective.ProbSphere d)
    (hν : ∀ x ∈ (ν : Measure (SSphere d)).support, b - inner ℝ (x : EucSpace d) u ≤ 0)
    (t : ℝ) : ridgeLaw p u b ν t = ν := by
  apply Subtype.ext
  change (ridgeLaw p u b ν t : Measure (SSphere d)) = (ν : Measure (SSphere d))
  rw [coe_ridgeLaw]
  have hs : ∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)), x ∈ (ν : Measure (SSphere d)).support :=
    Measure.support_mem_ae
  have he : (fun x => ridgeFlow p u b x t) =ᵐ[(ν : Measure (SSphere d))] id :=
    hs.mono fun x hx => ridgeFlow_inactive p u b x (hν x hx) t
  rw [Measure.map_congr he, Measure.map_id]

/-- A common positive initial height margin gives uniform exponential
particle attraction. Source: arXiv:2411.04551v3, §3.2, `eq: inclusion`. -/
theorem dist_ridge_center_le_of_height (p x : SSphere d) (a c : ℝ) (hc : 0 ≤ c) (hca : c < a)
    (hx : a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d)) (t : ℝ) (ht : 0 ≤ t) :
    dist (ridgeFlow (p : EucSpace d) (-(p : EucSpace d)) (-c) x t) p ≤
      2 * exp (-((a - c) / 2 * t)) := by
  have hactive : 0 < inner ℝ (x : EucSpace d) (p : EucSpace d) - c := by linarith
  have hh := norm_ridge_cap_center_sub_le p x c hactive t ht
  have hr : (a - c) / 2 ≤ (inner ℝ (x : EucSpace d) (p : EucSpace d) - c) *
      (1 + inner ℝ (x : EucSpace d) (p : EucSpace d)) / 2 := by
    nlinarith [mul_nonneg (by linarith : 0 ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) - c)
      (by linarith : 0 ≤ inner ℝ (x : EucSpace d) (p : EucSpace d))]
  have he : exp (-((inner ℝ (x : EucSpace d) (p : EucSpace d) - c) *
      (1 + inner ℝ (x : EucSpace d) (p : EucSpace d)) / 2 * t)) ≤ exp (-((a - c) / 2 * t)) :=
    exp_le_exp.mpr (by nlinarith [mul_le_mul_of_nonneg_right hr ht])
  have hn : ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ 2 := by
    have hb := norm_sub_le (x : EucSpace d) (p : EucSpace d)
    simpa only [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp p.2, one_add_one_eq_two] using hb
  exact hh.trans (mul_le_mul hn he (exp_pos _).le (by norm_num))

/-- The entire original support obeys the uniform center-radius bound.
Source: arXiv:2411.04551v3, §3.2, `eq: inclusion`. -/
theorem support_ridgeLaw_center_radius (p : SSphere d) (ν : Perspective.ProbSphere d) (a c : ℝ)
    (hc : 0 ≤ c) (hca : c < a)
    (hν : ∀ x ∈ (ν : Measure (SSphere d)).support, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d))
    (t : ℝ) (ht : 0 ≤ t) :
    (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ν t : Measure (SSphere d)).support ⊆
      Metric.closedBall p (2 * exp (-((a - c) / 2 * t))) := by
  rw [coe_ridgeLaw]
  have hs := support_map_subset_image (ν : Measure (SSphere d)) (fun x => ridgeFlow (p : EucSpace d) (-(p : EucSpace d)) (-c) x t)
    ((continuous_ridgeFlow _ _ _).comp (continuous_const.prodMk continuous_id))
  intro y hy
  obtain ⟨x, hx, rfl⟩ := hs hy
  exact dist_ridge_center_le_of_height p x a c hc hca (hν x hx) t ht

/-- One positive time contracts every original law with the same
initial height floor. Source: arXiv:2411.04551v3, §3.2, Step 2. -/
theorem exists_time_ridgeLaw_center_radius (p : SSphere d) (a c δ : ℝ)
    (hc : 0 ≤ c) (hca : c < a) (hδ : 0 < δ) :
    ∃ t : ℝ, 0 < t ∧ ∀ ν : Perspective.ProbSphere d,
      (∀ x ∈ (ν : Measure (SSphere d)).support, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d)) →
      (ridgeLaw (p : EucSpace d) (-(p : EucSpace d)) (-c) ν t : Measure (SSphere d)).support ⊆
        Metric.closedBall p δ := by
  have hr : 0 < (a - c) / 2 := div_pos (sub_pos.mpr hca) (by norm_num)
  have hl := (tendsto_const_mul_atTop_of_pos hr).2 (tendsto_id : Tendsto (id : ℝ → ℝ) atTop atTop)
  have he : Tendsto (fun t : ℝ => 2 * exp (-((a - c) / 2 * t))) atTop (nhds 0) := by
    simpa only [Function.comp_def, id_eq, mul_zero] using (tendsto_exp_neg_atTop_nhds_zero.comp hl).const_mul 2
  obtain ⟨t, ht, ht0⟩ := ((he.eventually (gt_mem_nhds hδ)).and (eventually_gt_atTop (0 : ℝ))).exists
  refine ⟨t, ht0, fun ν hν x hx => ?_⟩
  exact ((support_ridgeLaw_center_radius p ν a c hc hca hν t ht0.le) hx).trans ht.le

/-- Scaling the actual output retains the exact single-ridge weak-law
clock. Source: arXiv:2411.04551v3, §3.2, prescribed-time phases. -/
theorem twoRidgeLaw_scaled_ridge (κ : ℝ) (p u : EucSpace d) (b : ℝ) (v : EucSpace d) (c : ℝ)
    (ν : Perspective.ProbSphere d) (t : ℝ) :
    twoRidgeLaw (κ • p) u b 0 v c ν t = ridgeLaw p u b ν (κ * t) := by
  have he := twoRidgeLaw_smul_output κ p u b 0 v c ν t
  simpa only [smul_zero, twoRidgeLaw_zero_second] using he

/-- Original laws near the two orthogonal poles are fixed by the gated
perceptron. Source: arXiv:2411.04551v3, §3.2, Step 2. -/
theorem ridgeLaw_inactive_near_orthogonal_poles (p w : SSphere d) (q : EucSpace d) (c : ℝ)
    (hc : 0 < c) (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (ν : Perspective.ProbSphere d)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.ball w (c / 2) ∪ Metric.ball (antipode d w) (c / 2))
    (t : ℝ) : ridgeLaw q (-(p : EucSpace d)) (-c) ν t = ν := by
  apply ridgeLaw_inactive_support
  intro y hy
  have hpolar (z : SSphere d) (hz : inner ℝ (z : EucSpace d) (p : EucSpace d) = 0)
      (hball : dist y z < c / 2) : inner ℝ (y : EucSpace d) (p : EucSpace d) < c := by
    have hh := abs_real_inner_le_norm ((y : EucSpace d) - z) (p : EucSpace d)
    rw [inner_sub_left, hz, sub_zero, mem_sphere_zero_iff_norm.mp p.2, mul_one] at hh
    have hb := (le_abs_self (inner ℝ (y : EucSpace d) (p : EucSpace d))).trans hh
    change ‖(y : EucSpace d) - (z : EucSpace d)‖ < c / 2 at hball
    linarith
  have hi : inner ℝ (y : EucSpace d) (p : EucSpace d) < c := by
    rcases hν hy with hw | hw
    · exact hpolar w hwp hw
    · apply hpolar (antipode d w) ?_ hw
      change inner ℝ (-(w : EucSpace d)) (p : EucSpace d) = 0
      rw [inner_neg_left, hwp, neg_zero]
  rw [inner_neg_right]
  linarith

/-- An active off-axis circle atom and a transverse inactive atom
jointly realize the positive margin, observation time, inactivity and
orthogonal-pole support hypotheses for the same genuine gated field. -/
example : ∃ (p w x : SSphere 2) (ν₀ ν₁ : Perspective.ProbSphere 2) (a c δ t : ℝ),
    0 < c ∧ c < a ∧ 0 < δ ∧ 0 ≤ t ∧
    a ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2) ∧
    inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 ∧
    (∀ y ∈ (ν₀ : Measure (SSphere 2)).support, a ≤ inner ℝ (y : EucSpace 2) (p : EucSpace 2)) ∧
    (∀ y ∈ (ν₁ : Measure (SSphere 2)).support, -c - inner ℝ (y : EucSpace 2) (-(p : EucSpace 2)) ≤ 0) ∧
    (ν₁ : Measure (SSphere 2)).support ⊆ Metric.ball w (c / 2) ∪ Metric.ball (antipode 2 w) (c / 2) := by
  let p : SSphere 2 := basePoint 1
  let w : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  let x : SSphere 2 := ⟨EuclideanSpace.single 0 (3 / 5 : ℝ) + EuclideanSpace.single 1 (4 / 5 : ℝ), by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]
    norm_num⟩
  have hwp : inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 := by
    simp [w, p, basePoint, EuclideanSpace.inner_single_left]
  have hxp : (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2) := by
    norm_num [x, p, basePoint, inner_add_left, EuclideanSpace.inner_single_right]
  refine ⟨p, w, x, Perspective.diracProb 2 x, Perspective.diracProb 2 w, 1 / 2, 1 / 4, 1 / 10, 1,
    by norm_num, by norm_num, by norm_num, zero_le_one, hxp, hwp, ?_, ?_, ?_⟩
  · intro y hy
    rw [eq_of_mem_support_dirac hy]
    exact hxp
  · intro y hy
    rw [eq_of_mem_support_dirac hy, inner_neg_right, hwp]
    norm_num
  · intro y hy
    rw [eq_of_mem_support_dirac hy]
    exact Or.inl (by change dist w w < _; rw [dist_self]; norm_num)

end Transformer.Interpolation
