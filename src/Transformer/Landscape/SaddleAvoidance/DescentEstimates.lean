/-
# Saddle avoidance — the estimates behind the rates of descent curves

Let `x` descend `f` from time `0` on at the rates `c₁ > 0` and `c₂ ≥ 0`
(`IsDescentCurve`) and converge to `w`, where the Łojasiewicz inequality holds
with an exponent `a < 1`.  Then `f (x t) ≥ f w`, and either `x` reaches `w` and
stays there, or from some time on `r = f ∘ x - f w` is positive, its derivative
`⟪∇f(x), ẋ⟫` is at most `-c r ^ (2a)`, and `‖x t - w‖ ≤ K r ^ (1 - a)`
(`IsDescentCurve.exists_rate_estimates`).  On every `[0, T]`, `f (x t) - f w`
and `‖x t - w‖` are bounded (`IsDescentCurve.exists_bounds`).

These are the estimates of the proof of `prop:rates` of arXiv:1602.04915 for
flows (for gradient descent: `exists_rate_estimates_of_isLojasiewiczAt`);
`Landscape.SaddleAvoidance.FlowRates` turns them into rates.
-/

import Transformer.Landscape.SaddleAvoidance.DescentTrap

open Filter Metric Set
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {x : ℝ → E} {a c₁ c₂ : ℝ}

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- Along a descent curve from time `0` on, `f (x t) - f w` and `‖x t - w‖` are
bounded on every `[0, T]`. -/
theorem IsDescentCurve.exists_bounds (hf : Differentiable ℝ f)
    (hx : IsDescentCurve f c₁ c₂ 0 x) (hc₁ : 0 ≤ c₁) (w : E) :
    (∀ T, ∃ M, ∀ t ∈ Icc 0 T, f (x t) - f w ≤ M) ∧ ∀ T, ∃ M, ∀ t ∈ Icc 0 T, ‖x t - w‖ ≤ M :=
  ⟨fun _ => ⟨f (x 0) - f w, fun _ ht =>
      sub_le_sub_right (hx.antitoneOn hf hc₁ self_mem_Ici ht.1 ht.1) _⟩,
    fun _ => isCompact_Icc.exists_bound_of_continuousOn fun t ht =>
      ((hx t ht.1).1.continuousAt.sub continuousAt_const).continuousWithinAt⟩

/-- **The estimates behind the rates, for descent curves.**  Let `x` descend
`f` from time `0` on at the rates `c₁ > 0` and `c₂ ≥ 0` and converge to `w`,
where the Łojasiewicz inequality holds with `a < 1`.  Then `f (x t) ≥ f w`, and
either `x` reaches `w` and stays there, or from some time `T` on
`r = f ∘ x - f w` is positive, its derivative `⟪∇f(x), ẋ⟫` is at most
`-c r ^ (2a)`, and `‖x t - w‖ ≤ K r ^ (1 - a)`: the length of the rest of the
curve is bounded by the drop of `r ^ (1 - a)` (`IsDescentCurve.mul_norm_sub_le`). -/
theorem IsDescentCurve.exists_rate_estimates (hf : Differentiable ℝ f)
    (hx : IsDescentCurve f c₁ c₂ 0 x) (hc₁ : 0 < c₁) (hc₂ : 0 ≤ c₂) (ha1 : a < 1) {w : E}
    (hw : IsLojasiewiczAt f w a) (hlim : Tendsto x atTop (𝓝 w)) :
    (∀ t, 0 ≤ t → f w ≤ f (x t)) ∧ ((∀ᶠ t in atTop, x t = w) ∨
      ∃ c > 0, ∃ K ≥ 0, ∃ T ≥ 0, ∀ t, T ≤ t → 0 < f (x t) - f w ∧
        ⟪gradient f (x t), deriv x t⟫ ≤ -(c * (f (x t) - f w) ^ (2 * a)) ∧
          ‖x t - w‖ ≤ K * (f (x t) - f w) ^ (1 - a)) := by
  have hanti := hx.antitoneOn hf hc₁.le
  have hflim : Tendsto (fun t => f (x t)) atTop (𝓝 (f w)) := (hf.continuous.tendsto w).comp hlim
  have hfw : ∀ t, 0 ≤ t → f w ≤ f (x t) := fun t ht =>
    le_of_tendsto hflim (eventually_atTop.2 ⟨t, fun s hs => hanti ht (ht.trans hs) hs⟩)
  refine ⟨hfw, ?_⟩
  by_cases hreach : ∃ t₁, 0 ≤ t₁ ∧ f (x t₁) = f w
  · obtain ⟨t₁, ht₁, heq⟩ := hreach
    have hconst : ∀ t, t₁ ≤ t → x t = x t₁ := fun t ht =>
      (hx.mono ht₁).eq_of_comp_eq hf hc₁ (fun s hs => le_antisymm
        (hanti ht₁ (ht₁.trans hs) hs) (heq.trans_le (hfw s (ht₁.trans hs)))) ht
    have hxw : x t₁ = w := tendsto_nhds_unique (tendsto_const_nhds.congr'
      (eventually_atTop.2 ⟨t₁, fun t ht => (hconst t ht).symm⟩)) hlim
    exact Or.inl (eventually_atTop.2 ⟨t₁, fun t ht => (hconst t ht).trans hxw⟩)
  have hpos : ∀ t, 0 ≤ t → f w < f (x t) := fun t ht =>
    (hfw t ht).lt_of_ne fun h => hreach ⟨t, ht, h.symm⟩
  obtain ⟨m, hm, ε, hε, hev⟩ := hw
  obtain ⟨ρ, hρ, hloj⟩ := Metric.eventually_nhds_iff_ball.1 hev
  obtain ⟨T₁, hT₁⟩ := eventually_atTop.1 ((hlim.eventually (ball_mem_nhds w hρ)).and
    (hflim.eventually (gt_mem_nhds (lt_add_of_pos_right (f w) hε))))
  set T := max T₁ 0 with hT
  have hT0 : 0 ≤ T := le_max_right _ _
  have hl : ∀ t, T ≤ t → m * (f (x t) - f w) ^ a ≤ ‖gradient f (x t)‖ := fun t ht => by
    have h := hT₁ t ((le_max_left _ _).trans ht)
    have h' := hloj _ h.1 (hpos t (hT0.trans ht)) h.2
    rwa [abs_of_pos (sub_pos.2 (hpos t (hT0.trans ht)))] at h'
  have hκ : 0 < c₁ * m * (1 - a) := mul_pos (mul_pos hc₁ hm) (sub_pos.2 ha1)
  refine Or.inr ⟨c₁ * m ^ 2, by positivity, c₂ / (c₁ * m * (1 - a)), div_nonneg hc₂ hκ.le, T,
    hT0, fun t ht => ⟨sub_pos.2 (hpos t (hT0.trans ht)), ?_, ?_⟩⟩
  · have hr := sub_pos.2 (hpos t (hT0.trans ht))
    have h2 := pow_le_pow_left₀ (mul_nonneg hm.le (Real.rpow_nonneg hr.le a)) (hl t ht) 2
    have hsq : (f (x t) - f w) ^ (2 * a) = ((f (x t) - f w) ^ a) ^ 2 := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul hr.le]
      norm_num [mul_comm]
    rw [hsq]
    nlinarith [(hx t (hT0.trans ht)).2.1, mul_le_mul_of_nonneg_left h2 hc₁.le]
  · have hlen : ∀ s, t ≤ s → c₁ * m * (1 - a) * ‖x s - x t‖ ≤
        c₂ * (f (x t) - f w) ^ (1 - a) := fun s hs => by
      have h := (hx.mono (hT0.trans ht)).mul_norm_sub_le hf hs hc₁ hc₂ hm ha1
        (fun u hu => hpos u (hT0.trans (ht.trans hu.1))) (fun u hu => hl u (ht.trans hu.1))
      nlinarith [mul_nonneg hc₂
        (Real.rpow_nonneg (sub_nonneg.2 (hfw s (hT0.trans (ht.trans hs)))) (1 - a))]
    have hlim' : Tendsto (fun s => c₁ * m * (1 - a) * ‖x s - x t‖) atTop
        (𝓝 (c₁ * m * (1 - a) * ‖w - x t‖)) := ((hlim.sub_const (x t)).norm.const_mul _)
    have h := le_of_tendsto hlim' (eventually_atTop.2 ⟨t, hlen⟩)
    rw [norm_sub_rev, div_mul_eq_mul_div, le_div_iff₀ hκ]
    linarith

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line, with `a = 1/2` and the gradient flow `e^{-t}`, which descends `f` at the
rates `1` and `1` and converges to the critical point `0`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let x := fun t : ℝ => Real.exp (-t)
    Differentiable ℝ f ∧ IsDescentCurve f 1 1 0 x ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
      (1 / 2 : ℝ) < 1 ∧ IsLojasiewiczAt f 0 (1 / 2) ∧ Tendsto x atTop (𝓝 0) := by
  intro f x
  have hg : gradient f = id := gradient_half_mul_norm_sq
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero,
    isDescentCurve_of_gradientFlow fun t _ => ?_, one_pos, zero_le_one, by norm_num,
    isLojasiewiczAt_half_mul_norm_sq, Real.tendsto_exp_neg_atTop_nhds_zero⟩
  rw [hg]
  simpa [x] using (hasDerivAt_neg t).exp

end Transformer.Landscape.SaddleAvoidance
