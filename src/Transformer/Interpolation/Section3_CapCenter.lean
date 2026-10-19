module

public import Transformer.Interpolation.Section3_RidgeCaps
public import Transformer.Interpolation.Settling

/-!
# Center drift inside an arbitrary proper spherical cap

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
`eq: activation.ball` and `eq: dissipation.sphere`.

With the inward activation sign corrected, one neuron pointing toward the
cap center attracts every point in its active cap. The rate depends on the
initial point, as points arbitrarily close to the activation boundary move
arbitrarily slowly. The cap may be larger than a hemisphere.

This center-drift estimate supplies entry into a smaller cap for the
two-neuron construction. It does not assert the paper's unproved
single-neuron attraction to an arbitrary target in the original cap.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The projected ridge velocity has an exact radial dissipation identity.
Auxiliary for arXiv:2411.04551v3, Appendix, `eq: dissipation.sphere`;
the target and the particle are both actual unit vectors. -/
theorem inner_ridge_velocity_sub_output (p x : SSphere d) (u : EucSpace d) (b : ℝ) :
    inner ℝ (max (b - inner ℝ (x : EucSpace d) u) 0 • proj d x p) ((x : EucSpace d) - (p : EucSpace d)) =
      -(max (b - inner ℝ (x : EucSpace d) u) 0 * (1 + inner ℝ (x : EucSpace d) (p : EucSpace d)) / 2) *
        ‖(x : EucSpace d) - (p : EucSpace d)‖ ^ 2 := by
  simp only [proj, real_inner_smul_left, inner_sub_left, inner_sub_right,
    real_inner_self_eq_norm_sq, norm_sub_sq_real, mem_sphere_zero_iff_norm.mp x.2,
    mem_sphere_zero_iff_norm.mp p.2, real_inner_comm (x : EucSpace d) (p : EucSpace d)]
  ring

/-- An active particle approaches the cap center exponentially, with the
explicit rate determined by its initial activation and angular distance.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`.
The original autonomous ridge flow is used throughout. -/
theorem norm_ridge_cap_center_sub_le (z x₀ : SSphere d) (c : ℝ)
    (hactive : 0 < inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c) (t : ℝ) (ht : 0 ≤ t) :
    ‖(ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀ t : EucSpace d) - (z : EucSpace d)‖ ≤
      ‖(x₀ : EucSpace d) - (z : EucSpace d)‖ *
        exp (-(((inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c) *
          (1 + inner ℝ (x₀ : EucSpace d) (z : EucSpace d)) / 2) * t)) := by
  let X := ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀
  let v (s : ℝ) := max (-c - inner ℝ (X s : EucSpace d) (-(z : EucSpace d))) 0 • proj d (X s) z
  let rate := (inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c) *
    (1 + inner ℝ (x₀ : EucSpace d) (z : EucSpace d)) / 2
  have hκ : 0 ≤ 1 + inner ℝ (x₀ : EucSpace d) (z : EucSpace d) := by
    have h := abs_real_inner_le_norm (x₀ : EucSpace d) (z : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp x₀.2, mem_sphere_zero_iff_norm.mp z.2, mul_one] at h
    have hb := abs_le.mp h
    linarith
  have hcontr (s : ℝ) (hs : 0 ≤ s) : inner ℝ (v s) ((X s : EucSpace d) - (z : EucSpace d)) ≤
      -(rate * ‖(X s : EucSpace d) - (z : EucSpace d)‖ ^ 2) := by
    have hm := monotone_inner_ridgeFlow_output z (-(z : EucSpace d)) (-c) x₀ hs
    change inner ℝ (ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀ 0 : EucSpace d) (z : EucSpace d) ≤
      inner ℝ (X s : EucSpace d) (z : EucSpace d) at hm
    rw [ridgeFlow_zero] at hm
    have hα : 0 ≤ inner ℝ (X s : EucSpace d) (z : EucSpace d) - c := by linarith
    have hprod := mul_le_mul
      (show inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c ≤ inner ℝ (X s : EucSpace d) (z : EucSpace d) - c by linarith)
      (show 1 + inner ℝ (x₀ : EucSpace d) (z : EucSpace d) ≤ 1 + inner ℝ (X s : EucSpace d) (z : EucSpace d) by linarith)
      hκ hα
    have ha : max (-c - inner ℝ (X s : EucSpace d) (-(z : EucSpace d))) 0 =
        inner ℝ (X s : EucSpace d) (z : EucSpace d) - c := by
      rw [inner_neg_right]
      have he : -c - -inner ℝ (X s : EucSpace d) (z : EucSpace d) = inner ℝ (X s : EucSpace d) (z : EucSpace d) - c := by ring
      rw [he, max_eq_left hα]
    dsimp only [v]
    rw [inner_ridge_velocity_sub_output z (X s) (-(z : EucSpace d)) (-c), ha]
    have hb := mul_le_mul_of_nonneg_right hprod (sq_nonneg ‖(X s : EucSpace d) - (z : EucSpace d)‖)
    dsimp only [rate]
    nlinarith
  have hh := norm_sub_le_of_contraction d (fun s => (X s : EucSpace d)) v (z : EucSpace d) rate
    (fun s _ => hasDerivAt_ridgeFlow _ _ _ _ s) hcontr t ht
  simpa only [X, rate, ridgeFlow_zero] using hh

/-- Every active point in a proper spherical cap converges to its center.
The threshold may equal `-1`, giving the sphere with its antipode removed.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem tendsto_ridge_cap_center (z x₀ : SSphere d) (c : ℝ) (hc : -1 ≤ c)
    (hactive : 0 < inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c) :
    Tendsto (fun t => ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀ t) atTop (nhds z) := by
  let rate := (inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c) *
    (1 + inner ℝ (x₀ : EucSpace d) (z : EucSpace d)) / 2
  have hκ : 0 < 1 + inner ℝ (x₀ : EucSpace d) (z : EucSpace d) := by linarith
  have hr : 0 < rate := div_pos (mul_pos hactive hκ) (by norm_num)
  have hlin : Tendsto (fun t : ℝ => rate * t) atTop atTop := (tendsto_const_mul_atTop_of_pos hr).2 tendsto_id
  have he : Tendsto (fun t : ℝ => ‖(x₀ : EucSpace d) - (z : EucSpace d)‖ * exp (-(rate * t))) atTop (nhds 0) := by
    simpa only [Function.comp_def, mul_zero] using (tendsto_exp_neg_atTop_nhds_zero.comp hlin).const_mul ‖(x₀ : EucSpace d) - (z : EucSpace d)‖
  apply Metric.tendsto_nhds.mpr
  intro ε hε
  filter_upwards [he.eventually (gt_mem_nhds hε), eventually_ge_atTop (0 : ℝ)] with t ht ht0
  change ‖(ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀ t : EucSpace d) - (z : EucSpace d)‖ < ε
  exact (norm_ridge_cap_center_sub_le z x₀ c hactive t ht0).trans_lt ht

/-- The actual center drift eventually enters any cap containing the center.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`;
the entered cap can use a different axis from the original activation. -/
theorem exists_time_ridge_cap_center_enters (z x₀ w : SSphere d) (c h : ℝ) (hc : -1 ≤ c)
    (hactive : 0 < inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c)
    (hz : h < inner ℝ (z : EucSpace d) (w : EucSpace d)) :
    ∃ t : ℝ, 0 < t ∧ h < inner ℝ
      (ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀ t : EucSpace d) (w : EucSpace d) := by
  have hcφ : Continuous (fun x : SSphere d => inner ℝ (x : EucSpace d) (w : EucSpace d)) :=
    continuous_subtype_val.inner continuous_const
  have hl := (hcφ.tendsto z).comp
    (tendsto_ridge_cap_center z x₀ c hc hactive)
  obtain ⟨t, ht, hgt⟩ := ((hl.eventually (lt_mem_nhds hz)).and (eventually_gt_atTop (0 : ℝ))).exists
  exact ⟨t, hgt, ht⟩

/-- A positive-time endpoint can be chosen inside every prescribed
neighborhood of the center. Auxiliary for arXiv:2411.04551v3,
Appendix, the large-time concentration step in `lem: two.balls`. -/
theorem exists_time_ridge_cap_center_close (z x₀ : SSphere d) (c ε : ℝ)
    (hc : -1 ≤ c)
    (hactive : 0 < inner ℝ (x₀ : EucSpace d) (z : EucSpace d) - c)
    (hε : 0 < ε) :
    ∃ t : ℝ, 0 < t ∧
      dist (ridgeFlow (z : EucSpace d) (-(z : EucSpace d)) (-c) x₀ t) z < ε := by
  have he := (Metric.tendsto_nhds.mp (tendsto_ridge_cap_center z x₀ c hc hactive)) ε hε
  obtain ⟨t, ht, ht0⟩ := (he.and (eventually_gt_atTop (0 : ℝ))).exists
  exact ⟨t, ht0, ht⟩

/-- A transverse unit particle is active in a cap larger than a hemisphere.
The time, threshold, entrance cap and accuracy hypotheses hold jointly,
and the initial projected center velocity is nonzero. -/
example : ∃ (z x₀ w : SSphere 2) (c h t ε : ℝ),
    (-1 : ℝ) ≤ c ∧
    0 < inner ℝ (x₀ : EucSpace 2) (z : EucSpace 2) - c ∧
    h < inner ℝ (z : EucSpace 2) (w : EucSpace 2) ∧
    0 ≤ t ∧ 0 < ε ∧ proj 2 x₀ z ≠ 0 := by
  let z : SSphere 2 := basePoint 1
  let x₀ : SSphere 2 := ⟨EuclideanSpace.single 1 1, by simp⟩
  have hi : inner ℝ (x₀ : EucSpace 2) (z : EucSpace 2) = 0 := by
    norm_num [x₀, z, basePoint, EuclideanSpace.inner_single_left]
  have hz : inner ℝ (z : EucSpace 2) (z : EucSpace 2) = 1 := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp z.2]
    norm_num
  refine ⟨z, x₀, z, -1 / 2, 0, 1, 1, by norm_num, ?_, ?_,
    by norm_num, by norm_num, ?_⟩
  · rw [hi]
    norm_num
  · rw [hz]
    norm_num
  · have hp : proj 2 x₀ z = (z : EucSpace 2) := by simp [proj, hi]
    rw [hp]
    intro he
    have hn := mem_sphere_zero_iff_norm.mp z.2
    rw [he, norm_zero] at hn
    norm_num at hn

end Transformer.Interpolation
