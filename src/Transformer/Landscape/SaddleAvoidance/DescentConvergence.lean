module

public import Transformer.Landscape.SaddleAvoidance.DescentTrap
public import Transformer.Landscape.SaddleAvoidance.FlowConvergence

/-!
# Saddle avoidance — descent curves converge near a Łojasiewicz point

A curve that descends `f` (`IsDescentCurve`) and comes back at arbitrarily late
times arbitrarily close to `z`, up to translations leaving `f` invariant,
converges when `f` satisfies the Łojasiewicz inequality at `z` with exponent
`a < 1` (`IsDescentCurve.exists_tendsto_of_isLojasiewiczAt`), and its limit is
a critical point where `∇f` is continuous
(`IsDescentCurve.gradient_eq_zero_of_tendsto`). These are
`exists_tendsto_gradientFlow_of_isLojasiewiczAt` and
`gradient_eq_zero_of_tendsto_gradientFlow` for descent curves. `f` decreases
to `f z` along the curve. If it reaches `f z`, the curve stops
(`IsDescentCurve.eq_of_comp_eq`). Otherwise a translate of the curve
(`IsDescentCurve.add_const`), started close to `z` with `f` close to `f z`, is
trapped (`IsDescentCurve.mem_ball`), and the length bound
`IsDescentCurve.mul_norm_sub_le` makes it Cauchy
(`Landscape.Lojasiewicz.converges_of_displacement_bound`).

The flows `ẋ = -G(x) ∇f(x)` in a metric `G` with `c₁ ≤ G ≤ c₂` descend `f`
(`isDescentCurve_of_metric`), so they converge as well: among them the
trajectories of `metricGradientFlow` that `Landscape.SaddleAvoidance.MetricAvoidance`
keeps away from the strict saddles.

Source: arXiv:2312.10794v5, §4, after `thm: beta.interval`, and Appendix A:
Łojasiewicz's theorem makes every trajectory of a gradient flow of an analytic
function, in an analytic metric, converge.
-/

@[expose] public section

open Filter Metric Set
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {x : ℝ → E} {a c₁ c₂ t₀ : ℝ}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **A descent curve converges near a point where the Łojasiewicz inequality
holds.**  Let `f` be differentiable and satisfy the Łojasiewicz inequality at
`z` with exponent `a < 1`, and let `x` descend `f` from time `0` on at rates
`c₁ > 0` and `c₂ ≥ 0`. If at arbitrarily late times `x` comes arbitrarily
close to `z` up to translations in a set `Λ` leaving `f` invariant, `x`
converges. -/
theorem IsDescentCurve.exists_tendsto_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hx : IsDescentCurve f c₁ c₂ 0 x) (hc₁ : 0 < c₁) (hc₂ : 0 ≤ c₂) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) (ha1 : a < 1) {z : E} (hz : IsLojasiewiczAt f z a)
    (hcl : ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ, ‖x t + v - z‖ < ε) :
    ∃ w, Tendsto x atTop (𝓝 w) := by
  obtain ⟨m, hm, ε, hε, hev⟩ := hz
  obtain ⟨r, hr, hloj⟩ := Metric.eventually_nhds_iff_ball.1 hev
  have hanti := hx.antitoneOn hf hc₁.le
  have hcl' : ∀ δ > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ, ‖x (max 0 t) + v - z‖ < δ := fun δ hδ =>
    ((hcl δ hδ).and_eventually (eventually_ge_atTop 0)).mono fun t ht => by
      rw [max_eq_right ht.2]
      exact ht.1
  obtain ⟨hfz, hlim⟩ := le_and_tendsto_of_frequently (u := fun t => f (x (max 0 t)))
    (fun s t hst => hanti (le_max_left 0 s) (le_max_left 0 t) (max_le_max_left 0 hst))
    (frequently_abs_sub_lt hf.continuous.continuousAt hΛ hcl')
  have hfz' : ∀ t, 0 ≤ t → f z ≤ f (x t) := fun t ht => by
    simpa [max_eq_right ht] using hfz t
  by_cases hA : ∃ t₁, 0 ≤ t₁ ∧ f (x t₁) ≤ f z
  · obtain ⟨t₁, ht₁, hle⟩ := hA
    have hstop := fun t (ht : t₁ ≤ t) => (hx.mono ht₁).eq_of_comp_eq hf hc₁
      (fun u hu => le_antisymm (hanti ht₁ (ht₁.trans hu) hu)
        (hle.trans (hfz' u (ht₁.trans hu)))) ht
    exact ⟨x t₁, tendsto_const_nhds.congr'
      ((eventually_ge_atTop t₁).mono fun t ht => (hstop t ht).symm)⟩
  push Not at hA
  have hκ : 0 < c₁ * m * (1 - a) := mul_pos (mul_pos hc₁ hm) (sub_pos.2 ha1)
  have hlim' : Tendsto (fun t => f (x t)) atTop (𝓝 (f z)) :=
    hlim.congr' ((eventually_ge_atTop 0).mono fun t ht => by simp only [max_eq_right ht])
  have hφ : Tendsto (fun t => c₂ * (f (x t) - f z) ^ (1 - a)) atTop (𝓝 0) := by
    have h := ((hlim'.sub_const (f z)).rpow_const (Or.inr (sub_nonneg.2 ha1.le))).const_mul c₂
    rwa [sub_self, Real.zero_rpow (sub_pos.2 ha1).ne', mul_zero] at h
  obtain ⟨t₀, ⟨v, hv, ht₀z⟩, ⟨hφ₀, hε₀⟩, ht₀⟩ := ((hcl (r / 2) (half_pos hr)).and_eventually
    (((hφ.eventually (gt_mem_nhds (half_pos (mul_pos hκ hr)))).and
      (hlim'.eventually (gt_mem_nhds (lt_add_of_pos_right (f z) hε)))).and
      (eventually_ge_atTop 0))).exists
  have hfy : ∀ t, f (x t + v) = f (x t) := fun t => hΛ v hv (x t)
  have hy := (hx.mono ht₀).add_const (hΛ v hv)
  have hgap : ∀ t, t₀ ≤ t → f z < f (x t + v) := fun t ht => by
    rw [hfy]
    exact hA t (ht₀.trans ht)
  have htrap := fun t (ht : t₀ ≤ t) => hy.mem_ball hf hc₁ hc₂ hm ha1 hloj hgap
    (by rwa [hfy]) (by
      rw [hfy]
      have := mul_lt_mul_of_pos_left ht₀z hκ
      linarith) ht
  have hlen : ∀ s t, t₀ ≤ s → s ≤ t →
      ‖x t - x s‖ ≤ c₂ * (f (x s) - f z) ^ (1 - a) / (c₁ * m * (1 - a)) := fun s t hs hst => by
    have h := (hy.mono hs).mul_norm_sub_le hf hst hc₁ hc₂ hm ha1
      (fun u hu => hgap u (hs.trans hu.1)) fun u hu => by
        have hl := hloj _ (htrap u (hs.trans hu.1)) (hgap u (hs.trans hu.1)) (by
          rw [hfy]
          exact (hanti ht₀ (ht₀.trans (hs.trans hu.1)) (hs.trans hu.1)).trans_lt hε₀)
        rwa [abs_of_pos (sub_pos.2 (hgap u (hs.trans hu.1)))] at hl
    rw [hfy, hfy, add_sub_add_right_eq_sub] at h
    rw [le_div_iff₀ hκ, mul_comm]
    have := mul_nonneg hc₂
      (Real.rpow_nonneg (sub_nonneg.2 (hfz' t (ht₀.trans (hs.trans hst)))) (1 - a))
    linarith
  refine Landscape.Lojasiewicz.converges_of_displacement_bound x
    (fun t => c₂ * (f (x t) - f z) ^ (1 - a) / (c₁ * m * (1 - a))) t₀ ?_ hlen
  simpa only [zero_div] using hφ.div_const (c₁ * m * (1 - a))

/-- **The limit of a descent curve is a critical point**: if `x` descends `f`
at a rate `c₁ > 0` and converges to `w`, where `∇f` is continuous, then
`∇f(w) = 0`. Otherwise `‖∇f‖ > ‖∇f(w)‖ / 2` near `w`, so `f` drops by at
least `c₁ ‖∇f(w)‖² / 4` over every late unit of time, while it converges. -/
theorem IsDescentCurve.gradient_eq_zero_of_tendsto (hf : Differentiable ℝ f)
    (hx : IsDescentCurve f c₁ c₂ t₀ x) (hc₁ : 0 < c₁) {w : E}
    (hc : ContinuousAt (gradient f) w) (hw : Tendsto x atTop (𝓝 w)) : gradient f w = 0 := by
  by_contra hne
  set κ := c₁ * (‖gradient f w‖ / 2) ^ 2 with hκ
  have hκ0 : 0 < κ := mul_pos hc₁ (pow_pos (half_pos (norm_pos_iff.2 hne)) 2)
  obtain ⟨T, hT⟩ := eventually_atTop.1 (((hc.norm.tendsto.comp hw).eventually
    (lt_mem_nhds (half_lt_self (norm_pos_iff.2 hne)))).and (eventually_ge_atTop t₀))
  have hanti : AntitoneOn (fun s => f (x s) + κ * s) (Ici T) := by
    refine antitoneOn_of_hasDerivWithinAt_nonpos
      (f' := fun s => ⟪gradient f (x s), deriv x s⟫ + κ * 1) (convex_Ici T)
      (fun s hs => ((hx.hasDerivAt_comp (hT s hs).2 (hf _)).add
        ((hasDerivAt_id s).const_mul κ)).continuousAt.continuousWithinAt)
      (fun s hs => ((hx.hasDerivAt_comp (hT s (interior_subset hs)).2 (hf _)).add
        ((hasDerivAt_id s).const_mul κ)).hasDerivWithinAt) fun s hs => ?_
    obtain ⟨hg, hs₀⟩ := hT s (interior_subset hs)
    have h₁ := (hx s hs₀).2.1
    have hg' : (‖gradient f w‖ / 2) ^ 2 ≤ ‖gradient f (x s)‖ ^ 2 :=
      pow_le_pow_left₀ (by positivity) hg.le 2
    nlinarith [mul_le_mul_of_nonneg_left hg' hc₁.le]
  have hstep : ∀ t, T ≤ t → f (x (t + 1)) - f (x t) ≤ -κ := fun t ht => by
    have h := hanti ht (ht.trans (by linarith : t ≤ t + 1)) (by linarith : t ≤ t + 1)
    simp only at h
    linarith
  have hlim : Tendsto (fun t => f (x (t + 1)) - f (x t)) atTop (𝓝 (f w - f w)) :=
    ((hf.continuous.tendsto w).comp (hw.comp (tendsto_atTop_add_const_right _ 1 tendsto_id))).sub
      ((hf.continuous.tendsto w).comp hw)
  rw [sub_self] at hlim
  linarith [le_of_tendsto hlim (eventually_atTop.2 ⟨T, hstep⟩)]

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line, with `a = 1/2`, `Λ = {0}` and the critical point `0`: the gradient flow
`e^{-t}` descends `f` at the rates `1` and `1` and converges to `0`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let x := fun t : ℝ => Real.exp (-t)
    Differentiable ℝ f ∧ IsDescentCurve f 1 1 0 x ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
      (∀ v ∈ ({0} : Set ℝ), ∀ y, f (y + v) = f y) ∧ (1 / 2 : ℝ) < 1 ∧
      IsLojasiewiczAt f 0 (1 / 2) ∧
      (∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ ({0} : Set ℝ), ‖x t + v - 0‖ < ε) ∧
      ContinuousAt (gradient f) 0 ∧ Tendsto x atTop (𝓝 0) := by
  intro f x
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hlim : Tendsto x atTop (𝓝 0) := Real.tendsto_exp_neg_atTop_nhds_zero
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero,
    isDescentCurve_of_gradientFlow fun t _ => ?_, one_pos, zero_le_one, by simp, by norm_num,
    isLojasiewiczAt_half_mul_norm_sq, fun ε hε => ?_, by rw [hg]; exact continuousAt_id, hlim⟩
  · rw [hg]
    simpa [x] using (hasDerivAt_neg t).exp
  · exact (Metric.tendsto_nhds.1 hlim ε hε).frequently.mono fun t ht =>
      ⟨0, Set.mem_singleton 0, by rwa [add_zero, ← dist_eq_norm]⟩

end Transformer.Landscape.SaddleAvoidance
