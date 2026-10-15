/-
# Scalar uniqueness at a repelling limiting angle

ArXiv:2411.04990v2, Appendix A and §5, repelling-limit
avoidance for the smooth frozen specialization.
We keep the actual scalar and parameter dependence explicit.
A positive limiting own-angle derivative stays nonnegative
on the segment between two angles approaching that limit,
when their common parameter trajectory also converges.

The actual scalar field is then monotone between those two
angles. Their squared difference has nonnegative derivative
at late times. That squared difference also tends to zero,
so it must vanish from a finite time onward. This is the
scalar ingredient for uniqueness of a follower approaching a
fixed repelling limit with a fixed original prefix.
It requires continuity of the actual field derivative, not
the false general printed C1 energy convergence statement.
-/

import Transformer.Causal.SectionA_ScalarConvergence

open Real Set Filter

namespace Transformer.Causal

/-- An eventually nonnegative actual scalar curve with eventually nonnegative derivative and limit zero vanishes at every sufficiently late time.
Source: arXiv:2411.04990v2, Appendix A and §5, the scalar separation argument for the smooth frozen repeller. -/
theorem frozenScalar_eq_zero_of_nondecreasing_limit (r r' : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (r' t) t) (T : ℝ) (hr : ∀ t, T ≤ t → 0 ≤ r t)
    (hdr : ∀ t, T ≤ t → 0 ≤ r' t) (hlim : Tendsto r atTop (nhds 0)) :
    ∀ s, T ≤ s → r s = 0 := by
  have hm : MonotoneOn r (Ici T) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici T)
      (fun x hx => (hd x).continuousAt.continuousWithinAt)
      (fun x hx => (hd x).differentiableAt.differentiableWithinAt)
    intro x hx
    rw [interior_Ici, mem_Ioi] at hx
    rw [(hd x).deriv]
    exact hdr x hx.le
  intro s hs
  have he : ∀ᶠ t in atTop, r s ≤ r t :=
    (eventually_ge_atTop s).mono (fun t ht => hm hs (hs.trans ht) ht)
  exact le_antisymm (ge_of_tendsto hlim he) (hr s hs)

/-- Joint continuity at a positive derivative gives eventual nonnegativity throughout each segment between two actual angles approaching its limiting angle.
Source: arXiv:2411.04990v2, Appendix A and §5, the local smoothness input for original frozen repelling-limit avoidance. -/
theorem frozenScalar_eventually_nonneg_on_segments {E : Type*} [TopologicalSpace E]
    (F : E × ℝ → ℝ) (pstar : E) (l : ℝ) (hF : ContinuousAt F (pstar, l)) (hpos : 0 < F (pstar, l))
    (p : ℝ → E) (hp : Tendsto p atTop (nhds pstar)) (x y : ℝ → ℝ)
    (hx : Tendsto x atTop (nhds l)) (hy : Tendsto y atTop (nhds l)) :
    ∀ᶠ t in atTop, ∀ u ∈ Icc (min (x t) (y t)) (max (x t) (y t)), 0 ≤ F (p t, u) := by
  have hgood : {q | 0 < F q} ∈ nhds (pstar, l) := hF.eventually (eventually_gt_nhds hpos)
  obtain ⟨U, hU, V, hV, hUV⟩ := mem_nhds_prod_iff.mp hgood
  obtain ⟨δ, hδ, hball⟩ := Metric.mem_nhds_iff.mp hV
  have hb := Metric.ball_mem_nhds l hδ
  filter_upwards [hp.eventually hU, hx.eventually hb, hy.eventually hb] with t ht hxt hyt u hu
  rw [Real.dist_eq, abs_lt] at hxt hyt
  have hulo : l - δ < u := by
    have h := lt_min (show l - δ < x t by linarith [hxt.1]) (show l - δ < y t by linarith [hyt.1])
    exact h.trans_le hu.1
  have huhi : u < l + δ := by
    have h := max_lt (show x t < l + δ by linarith [hxt.2]) (show y t < l + δ by linarith [hyt.2])
    exact hu.2.trans_lt h
  have hub : u ∈ Metric.ball l δ := by rw [Metric.mem_ball, Real.dist_eq, abs_lt]; constructor <;> linarith
  have hpair : (p t, u) ∈ U ×ˢ V := ⟨ht, hball hub⟩
  exact (hUV hpair).le

/-- The actual positive limiting derivative has a positive uniform lower bound on all sufficiently late segments between the two converging angles.
Source: arXiv:2411.04990v2, Appendix A and §5, the uniform smooth repulsion estimate for the original frozen specialization. -/
theorem frozenScalar_eventually_positive_on_segments {E : Type*} [TopologicalSpace E]
    (F : E × ℝ → ℝ) (pstar : E) (l : ℝ) (hF : ContinuousAt F (pstar, l)) (hpos : 0 < F (pstar, l))
    (p : ℝ → E) (hp : Tendsto p atTop (nhds pstar)) (x y : ℝ → ℝ)
    (hx : Tendsto x atTop (nhds l)) (hy : Tendsto y atTop (nhds l)) :
    ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, ∀ u ∈ Icc (min (x t) (y t)) (max (x t) (y t)), c ≤ F (p t, u) := by
  let c := F (pstar, l) / 2
  have hc : 0 < c := by dsimp [c]; positivity
  have h := frozenScalar_eventually_nonneg_on_segments (fun q => F q - c) pstar l
    (hF.sub continuousAt_const) (by dsimp [c]; linarith) p hp x y hx hy
  refine ⟨c, hc, h.mono ?_⟩
  intro t ht u hu
  have htu := ht u hu
  linarith

/-- A scalar field whose actual derivative is nonnegative between two angles has a nonnegative signed separation product there.
Source: arXiv:2411.04990v2, Appendix A and §5, the scalar monotonicity estimate for original frozen repelling-limit avoidance. -/
theorem frozenScalar_field_separation_nonneg (f d : ℝ → ℝ) (hd : ∀ u, HasDerivAt f (d u) u)
    (x y : ℝ) (hder : ∀ u ∈ Icc (min x y) (max x y), 0 ≤ d u) :
    0 ≤ (x - y) * (f x - f y) := by
  have hm : MonotoneOn f (Icc (min x y) (max x y)) := by
    apply monotoneOn_of_deriv_nonneg (convex_Icc _ _)
      (fun u hu => (hd u).continuousAt.continuousWithinAt)
      (fun u hu => (hd u).differentiableAt.differentiableWithinAt)
    intro u hu
    rw [interior_Icc] at hu
    rw [(hd u).deriv]
    exact hder u ⟨hu.1.le, hu.2.le⟩
  have hx : x ∈ Icc (min x y) (max x y) := ⟨min_le_left _ _, le_max_left _ _⟩
  have hy : y ∈ Icc (min x y) (max x y) := ⟨min_le_right _ _, le_max_right _ _⟩
  by_cases hxy : x ≤ y
  · exact mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr hxy) (sub_nonpos.mpr (hm hx hy hxy))
  · have hyx : y ≤ x := (lt_of_not_ge hxy).le
    exact mul_nonneg (sub_nonneg.mpr hyx) (sub_nonneg.mpr (hm hy hx hyx))

/-- Two actual scalar solutions with the same converging parameter trajectory cannot approach the same strictly repelling limit while remaining distinct at arbitrarily late times.
Source: arXiv:2411.04990v2, Appendix A and §5, the smooth scalar repulsion argument for the original frozen specialization. -/
theorem frozenScalar_eventually_eq_of_repelling_limit {E : Type*} [TopologicalSpace E]
    (f d : E → ℝ → ℝ) (hdf : ∀ p u, HasDerivAt (f p) (d p u) u)
    (pstar : E) (l : ℝ) (hD : ContinuousAt (fun q : E × ℝ => d q.1 q.2) (pstar, l))
    (hpos : 0 < d pstar l) (p : ℝ → E) (hp : Tendsto p atTop (nhds pstar)) (x y : ℝ → ℝ)
    (hx : ∀ t, HasDerivAt x (f (p t) (x t)) t) (hy : ∀ t, HasDerivAt y (f (p t) (y t)) t)
    (hxlim : Tendsto x atTop (nhds l)) (hylim : Tendsto y atTop (nhds l)) :
    ∃ T : ℝ, ∀ t, T ≤ t → x t = y t := by
  obtain ⟨c, hc, hseg⟩ := frozenScalar_eventually_positive_on_segments
    (fun q : E × ℝ => d q.1 q.2) pstar l hD hpos p hp x y hxlim hylim
  obtain ⟨T, hT⟩ := eventually_atTop.mp hseg
  let r (t : ℝ) := (x t - y t) ^ 2
  let r' (t : ℝ) := 2 * (x t - y t) * (f (p t) (x t) - f (p t) (y t))
  have hdr (t : ℝ) : HasDerivAt r (r' t) t := by
    simpa! only [r, r', Nat.cast_ofNat, Nat.reduceSub, pow_one] using ((hx t).fun_sub (hy t)).fun_pow 2
  have hlim : Tendsto r atTop (nhds 0) := by simpa only [r, sub_self, zero_pow (by decide : (2 : ℕ) ≠ 0)] using (hxlim.sub hylim).pow 2
  have hnonneg : ∀ t, T ≤ t → 0 ≤ r' t := by
    intro t ht
    have h := frozenScalar_field_separation_nonneg (f (p t)) (d (p t)) (hdf (p t)) (x t) (y t)
      (fun u hu => hc.le.trans (hT t ht u hu))
    dsimp [r']
    nlinarith only [h]
  have hz := frozenScalar_eq_zero_of_nondecreasing_limit r r' hdr T (fun t _ => sq_nonneg _) hnonneg hlim
  exact ⟨T, fun t ht => sub_eq_zero.mp (sq_eq_zero_iff.mp (hz t ht))⟩

/-- An actual zero separation jointly satisfies nonnegative derivative and vanishing-limit premises.
Source: arXiv:2411.04990v2, Appendix A and §5, coincident scalar trajectories in the original repeller comparison. -/
example : (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 t) ∧
    (∀ t : ℝ, (0 : ℝ) ≤ t → (0 : ℝ) ≤ 0) ∧ Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds 0) :=
  ⟨fun t => hasDerivAt_const t 0, fun _ _ => le_rfl, tendsto_const_nhds⟩

/-- The genuinely parameter-dependent smooth field `exp(p+u)-1` realizes every scalar repeller premise at its actual zero.
Source: arXiv:2411.04990v2, Appendix A and §5, the auxiliary smooth scalar repulsion argument. -/
example : ∃ T : ℝ, ∀ t : ℝ, T ≤ t → (fun _ : ℝ => (0 : ℝ)) t = (fun _ : ℝ => (0 : ℝ)) t := by
  apply frozenScalar_eventually_eq_of_repelling_limit
    (fun p u : ℝ => Real.exp (p + u) - 1) (fun p u : ℝ => Real.exp (p + u)) _ 0 0
    (by fun_prop) (by norm_num) (fun _ => 0) tendsto_const_nhds (fun _ => 0) (fun _ => 0)
    _ _ tendsto_const_nhds tendsto_const_nhds
  · intro p u
    simpa only [id_eq, zero_add, mul_one] using (((hasDerivAt_const u p).fun_add (hasDerivAt_id u)).exp).sub_const 1
  · intro t
    simpa only [zero_add, Real.exp_zero, sub_self] using hasDerivAt_const t (0 : ℝ)
  · intro t
    simpa only [zero_add, Real.exp_zero, sub_self] using hasDerivAt_const t (0 : ℝ)

end Transformer.Causal
