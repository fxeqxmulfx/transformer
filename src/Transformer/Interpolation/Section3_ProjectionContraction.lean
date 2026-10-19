module

public import Transformer.Interpolation.Section3_ProjectionDynamics

/-!
# Uniform contraction away from the repelling pole

ArXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`.
The actual flow of `Proj_x w` drives every compact set missing `-w`
into any prescribed ball about `w` in finite time. An explicit squared
distance estimate gives the uniformity over the original compact set.
-/

@[expose] public section

open Real Filter Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The exact decay identity for the original constant projection field.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionFlow_distance_sq_mul (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (t : ℝ) :
    ‖projectionFlow w t x - w‖ ^ 2 * projectionDenom w t x =
      2 * exp (-t) * (1 - inner (𝕜 := ℝ) w x) := by
  have hD0 := (projectionDenom_pos w x hw hx t).ne'
  rw [norm_sub_sq_real, norm_projectionFlow w x hw hx t, hw,
    real_inner_comm w (projectionFlow w t x), inner_projectionFlow w x hw t]
  field_simp
  simp only [projectionDenom, cosh_eq, sinh_eq]
  ring

/-- A uniform lower height gives a uniform growing denominator.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionDenom_lower (w x : EucSpace d) (hw : ‖w‖ = 1) (hx : ‖x‖ = 1)
    (c : ℝ) (hc : c ≤ inner (𝕜 := ℝ) w x) (t : ℝ) :
    exp t * (1 + c) / 2 ≤ projectionDenom w t x := by
  have hh : inner (𝕜 := ℝ) w x ≤ 1 := by
    have h := abs_real_inner_le_norm w x
    rw [hw, hx, mul_one] at h
    exact (le_abs_self _).trans h
  simp only [projectionDenom, cosh_eq, sinh_eq]
  nlinarith [mul_nonneg (exp_pos t).le (sub_nonneg.mpr hc),
    mul_nonneg (exp_pos (-t)).le (sub_nonneg.mpr hh)]

/-- Actual squared distances decay uniformly for heights bounded above the
repelling value `-1`. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`. No regularity of a measure is used here. -/
theorem projectionFlow_distance_sq_le (w x : EucSpace d)
    (hw : ‖w‖ = 1) (hx : ‖x‖ = 1) (c : ℝ) (hc : -1 < c)
    (hcx : c ≤ inner (𝕜 := ℝ) w x) (t : ℝ) :
    ‖projectionFlow w t x - w‖ ^ 2 ≤ 8 / (1 + c) * exp (-t) ^ 2 := by
  have hh : -1 ≤ inner (𝕜 := ℝ) w x := by
    have h := abs_real_inner_le_norm w x
    rw [hw, hx, mul_one] at h
    exact (abs_le.mp h).1
  have hD := projectionDenom_lower w x hw hx c hcx t
  have hid := projectionFlow_distance_sq_mul w x hw hx t
  have hprod : exp t * exp (-t) = 1 := by rw [← exp_add]; simp
  have hmul : ‖projectionFlow w t x - w‖ ^ 2 * exp t * (1 + c) ≤ 8 * exp (-t) := by
    nlinarith [mul_le_mul_of_nonneg_left hD
      (sq_nonneg ‖projectionFlow w t x - w‖), exp_pos (-t)]
  have hfinal : ‖projectionFlow w t x - w‖ ^ 2 * (1 + c) ≤ 8 * exp (-t) ^ 2 := by
    calc
      _ = (‖projectionFlow w t x - w‖ ^ 2 * exp t * (1 + c)) * exp (-t) := by
        calc
          _ = ‖projectionFlow w t x - w‖ ^ 2 * (1 + c) * (exp t * exp (-t)) := by
            rw [hprod, mul_one]
          _ = _ := by ring
      _ ≤ 8 * exp (-t) * exp (-t) :=
        mul_le_mul_of_nonneg_right hmul (exp_pos (-t)).le
      _ = _ := by ring
  have hc0 : 0 < 1 + c := by linarith
  calc
    _ ≤ (8 * exp (-t) ^ 2) / (1 + c) := (le_div_iff₀ hc0).mpr hfinal
    _ = _ := by ring

/-- A compact set missing the repelling pole has a strict uniform lower
height. Source: arXiv:2411.04551v3, §3.1, the first phase of
`lem: first.quadrant`. -/
theorem exists_height_floor_of_compact (w : SSphere d) (K : Set (SSphere d))
    (hK : IsCompact K) (hmiss : antipode d w ∉ K) :
    ∃ c : ℝ, -1 < c ∧ ∀ x ∈ K, c ≤ inner (𝕜 := ℝ) (w : EucSpace d) (x : EucSpace d) := by
  by_cases hne : K.Nonempty
  swap
  · exact ⟨0, by norm_num, fun x hx => (hne ⟨x, hx⟩).elim⟩
  have hc : Continuous (fun x : SSphere d =>
      inner (𝕜 := ℝ) (w : EucSpace d) (x : EucSpace d)) :=
    continuous_const.inner continuous_subtype_val
  obtain ⟨x, hx, hmin⟩ := hK.exists_isMinOn hne hc.continuousOn
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  have hxn : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  refine ⟨inner (𝕜 := ℝ) (w : EucSpace d) (x : EucSpace d), ?_, fun y hy => hmin hy⟩
  have hh := abs_real_inner_le_norm (w : EucSpace d) (x : EucSpace d)
  rw [hw, hxn, mul_one] at hh
  by_contra hstrict
  have heq : inner (𝕜 := ℝ) (w : EucSpace d) (x : EucSpace d) = -1 := by
    linarith [(abs_le.mp hh).1]
  have hz : (x : EucSpace d) + (w : EucSpace d) = 0 := by
    apply norm_eq_zero.mp
    have hsq := norm_add_sq_real (x : EucSpace d) (w : EucSpace d)
    rw [hxn, hw, real_inner_comm (w : EucSpace d) (x : EucSpace d), heq] at hsq
    nlinarith [norm_nonneg ((x : EucSpace d) + (w : EucSpace d))]
  have hxa : x = antipode d w := Subtype.ext (add_eq_zero_iff_eq_neg.mp hz)
  exact hmiss (hxa ▸ hx)

/-- The actual flow sends the entire original compact set into an arbitrary
prescribed ball about its attractor at one positive time.
Source: arXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`. -/
theorem exists_time_sphereProjectionFlow_ball (w : SSphere d) (K : Set (SSphere d))
    (hK : IsCompact K) (hmiss : antipode d w ∉ K) (ε : ℝ) (hε : 0 < ε) :
    ∃ t : ℝ, 0 < t ∧ ∀ x ∈ K, dist (sphereProjectionFlow w t x) w < ε := by
  obtain ⟨c, hc, hcx⟩ := exists_height_floor_of_compact w K hK hmiss
  have htend : Tendsto (fun t : ℝ => 8 / (1 + c) * exp (-t) ^ 2) atTop (nhds 0) := by
    simpa only [zero_pow (by decide : 2 ≠ 0), mul_zero] using
      (tendsto_exp_neg_atTop_nhds_zero.pow 2).const_mul (8 / (1 + c))
  obtain ⟨t, ht, ht0⟩ := ((htend.eventually (gt_mem_nhds (sq_pos_of_pos hε))).and
    (eventually_gt_atTop (0 : ℝ))).exists
  refine ⟨t, ht0, fun x hx => ?_⟩
  have hb := (projectionFlow_distance_sq_le (w : EucSpace d) (x : EucSpace d)
    (mem_sphere_zero_iff_norm.mp w.2) (mem_sphere_zero_iff_norm.mp x.2)
    c hc (hcx x hx) t).trans_lt ht
  change dist (projectionFlow (w : EucSpace d) t (x : EucSpace d)) (w : EucSpace d) < ε
  rw [dist_eq_norm]
  nlinarith [norm_nonneg (projectionFlow (w : EucSpace d) t (x : EucSpace d) - (w : EucSpace d))]

/-- Every initial point other than the repelling pole converges to the
attracting pole along the actual original ODE. Source: arXiv:2411.04551v3,
§3.1, `lem: first.quadrant`. -/
theorem tendsto_sphereProjectionFlow (w x : SSphere d) (hx : x ≠ antipode d w) :
    Tendsto (fun t => sphereProjectionFlow w t x) atTop (nhds w) := by
  obtain ⟨c, hc, hcx⟩ := exists_height_floor_of_compact w {x} isCompact_singleton
    (by simpa only [Set.mem_singleton_iff] using hx.symm)
  have htend : Tendsto (fun t : ℝ => 8 / (1 + c) * exp (-t) ^ 2) atTop (nhds 0) := by
    simpa only [zero_pow (by decide : 2 ≠ 0), mul_zero] using
      (tendsto_exp_neg_atTop_nhds_zero.pow 2).const_mul (8 / (1 + c))
  apply Metric.tendsto_nhds.mpr
  intro ε hε
  filter_upwards [htend.eventually (gt_mem_nhds (sq_pos_of_pos hε))] with t ht
  have hb := (projectionFlow_distance_sq_le (w : EucSpace d) (x : EucSpace d)
    (mem_sphere_zero_iff_norm.mp w.2) (mem_sphere_zero_iff_norm.mp x.2)
    c hc (hcx x (Set.mem_singleton x)) t).trans_lt ht
  change dist (projectionFlow (w : EucSpace d) t (x : EucSpace d)) (w : EucSpace d) < ε
  rw [dist_eq_norm]
  nlinarith [norm_nonneg (projectionFlow (w : EucSpace d) t (x : EucSpace d) - (w : EucSpace d))]

/-- All compact-contraction inputs hold for a nonempty actual spherical set. -/
example : IsCompact ({basePoint 1} : Set (SSphere 2)) ∧
    antipode 2 (basePoint 1) ∉ ({basePoint 1} : Set (SSphere 2)) ∧ (0 : ℝ) < 1 := by
  refine ⟨isCompact_singleton, ?_, one_pos⟩
  simpa only [Set.mem_singleton_iff] using antipode_ne 2 (basePoint 1)

end Transformer.Interpolation
