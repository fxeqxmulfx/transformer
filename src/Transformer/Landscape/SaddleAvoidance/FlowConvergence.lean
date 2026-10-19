module

public import Transformer.Landscape.SaddleAvoidance.FlowTrap
public import Transformer.Landscape.Lojasiewicz.ModulatedLength

/-!
# Saddle avoidance — gradient flows converge near a Łojasiewicz point

A trajectory of the gradient flow `ẋ = -∇f(x)` that comes back at arbitrarily
late times arbitrarily close to `z`, up to translations leaving `f`
invariant, converges when `f` satisfies the Łojasiewicz inequality at `z`
with exponent `a < 1` (`exists_tendsto_gradientFlow_of_isLojasiewiczAt`).
Without translations its limit is `z`
(`tendsto_gradientFlow_of_isLojasiewiczAt`). This is the flow analogue of
`exists_tendsto_of_isLojasiewiczAt`, with the same proof. `f` decreases to
`f z` along the trajectory. If it reaches `f z`, the gradient vanishes from
then on and the trajectory stops (`eq_of_comp_gradientFlow_eq`). Otherwise
a translate of the trajectory (`hasDerivAt_gradientFlow_add`), started close
to `z` with `f` close to `f z`, is trapped (`mem_ball_of_gradientFlow`), and
the length bound `mul_norm_sub_le_of_gradientFlow` makes it Cauchy.

`Landscape.Lojasiewicz.lojasiewicz_modulated_of_local_gradient_inequality` proves
that a bounded modulated flow of an analytic function on `ℝᴺ` converges when
the inequality holds at every critical point. Its last step,
`Landscape.Lojasiewicz.converges_of_displacement_bound`, turns the length bound
into convergence here too.
-/

@[expose] public section

open Filter Metric Set
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {x : ℝ → E} {a : ℝ}

/-- The translate of a gradient-flow trajectory by a vector leaving `f`
invariant is a trajectory. -/
theorem hasDerivAt_gradientFlow_add {v : E} (hv : ∀ y, f (y + v) = f y) {t : ℝ}
    (hx : HasDerivAt x (-gradient f (x t)) t) :
    HasDerivAt (fun s => x s + v) (-gradient f (x t + v)) t := by
  rw [gradient_add_eq hv]
  exact hx.add_const v

/-- **A gradient flow along which `f` stays constant does not move**: its
speed `‖∇f‖` is the square root of the rate at which `f` decreases. -/
theorem eq_of_comp_gradientFlow_eq (hf : Differentiable ℝ f) {t₀ : ℝ}
    (hx : ∀ t, t₀ ≤ t → HasDerivAt x (-gradient f (x t)) t)
    (hc : ∀ t, t₀ ≤ t → f (x t) = f (x t₀)) {t : ℝ} (ht : t₀ ≤ t) : x t = x t₀ := by
  have hvel : ∀ u ∈ Ici t₀, HasDerivWithinAt x 0 (Ici t₀) u := fun u hu => by
    have hd := (hasDerivAt_comp_gradientFlow (hf _) (hx u hu)).hasDerivWithinAt (s := Ici t₀)
    have hconst : HasDerivWithinAt (fun s => f (x s)) 0 (Ici t₀) u :=
      (hasDerivWithinAt_const u (Ici t₀) (f (x t₀))).congr_of_mem hc hu
    have hsq := (uniqueDiffOn_Ici t₀ u hu).eq_deriv _ hd hconst
    have hg : gradient f (x u) = 0 := by
      rwa [neg_eq_zero, pow_eq_zero_iff two_ne_zero, norm_eq_zero] at hsq
    simpa [hg] using (hx u hu).hasDerivWithinAt (s := Ici t₀)
  have h := (convex_Ici t₀).norm_image_sub_le_of_norm_hasDerivWithin_le (C := 0) hvel
    (fun _ _ => by simp) self_mem_Ici ht
  rwa [zero_mul, norm_le_zero_iff, sub_eq_zero] at h

/-- **A gradient flow converges near a point where the Łojasiewicz
inequality holds.**  Let `f` be differentiable and satisfy the Łojasiewicz
inequality at `z` with exponent `a < 1`, and let `x` solve `ẋ = -∇f(x)` from
time `0` on. If at arbitrarily late times `x` comes arbitrarily close to
`z` up to translations in a set `Λ` leaving `f` invariant, `x` converges. -/
theorem exists_tendsto_gradientFlow_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) (ha1 : a < 1) {z : E} (hz : IsLojasiewiczAt f z a)
    (hcl : ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ, ‖x t + v - z‖ < ε) :
    ∃ w, Tendsto x atTop (𝓝 w) := by
  obtain ⟨m, hm, ε, hε, hev⟩ := hz
  obtain ⟨r, hr, hloj⟩ := Metric.eventually_nhds_iff_ball.1 hev
  have hanti := antitoneOn_comp_gradientFlow hf hx
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
    have hstop := fun t (ht : t₁ ≤ t) => eq_of_comp_gradientFlow_eq hf
      (fun u hu => hx u (ht₁.trans hu)) (fun u hu => le_antisymm
        (hanti ht₁ (ht₁.trans hu) hu) (hle.trans (hfz' u (ht₁.trans hu)))) ht
    exact ⟨x t₁, tendsto_const_nhds.congr'
      ((eventually_ge_atTop t₁).mono fun t ht => (hstop t ht).symm)⟩
  push Not at hA
  have hκ : 0 < m * (1 - a) := mul_pos hm (sub_pos.2 ha1)
  have hlim' : Tendsto (fun t => f (x t)) atTop (𝓝 (f z)) :=
    hlim.congr' ((eventually_ge_atTop 0).mono fun t ht => by simp only [max_eq_right ht])
  have hφ : Tendsto (fun t => (f (x t) - f z) ^ (1 - a)) atTop (𝓝 0) := by
    have h := (hlim'.sub_const (f z)).rpow_const (Or.inr (sub_nonneg.2 ha1.le))
    rwa [sub_self, Real.zero_rpow (sub_pos.2 ha1).ne'] at h
  obtain ⟨t₀, ⟨v, hv, ht₀z⟩, ⟨hφ₀, hε₀⟩, ht₀⟩ := ((hcl (r / 2) (half_pos hr)).and_eventually
    (((hφ.eventually (gt_mem_nhds (half_pos (mul_pos hκ hr)))).and
      (hlim'.eventually (gt_mem_nhds (lt_add_of_pos_right (f z) hε)))).and
      (eventually_ge_atTop 0))).exists
  have hfy : ∀ t, f (x t + v) = f (x t) := fun t => hΛ v hv (x t)
  have hy : ∀ t, t₀ ≤ t → HasDerivAt (fun s => x s + v) (-gradient f (x t + v)) t :=
    fun t ht => hasDerivAt_gradientFlow_add (hΛ v hv) (hx t (ht₀.trans ht))
  have hgap : ∀ t, t₀ ≤ t → f z < f (x t + v) := fun t ht => by
    rw [hfy]
    exact hA t (ht₀.trans ht)
  have htrap := fun t (ht : t₀ ≤ t) => mem_ball_of_gradientFlow hf hy hm ha1 hloj hgap
    (by rwa [hfy]) (by
      rw [hfy]
      have := mul_lt_mul_of_pos_left ht₀z hκ
      linarith) ht
  have hlen : ∀ s t, t₀ ≤ s → s ≤ t → ‖x t - x s‖ ≤ (f (x s) - f z) ^ (1 - a) / (m * (1 - a)) :=
    fun s t hs hst => by
      have h := mul_norm_sub_le_of_gradientFlow hf hst (fun u hu => hy u (hs.trans hu.1)) hm ha1
        (fun u hu => hgap u (hs.trans hu.1)) fun u hu => by
          have hl := hloj _ (htrap u (hs.trans hu.1)) (hgap u (hs.trans hu.1))
            (by
              rw [hfy]
              exact (hanti ht₀ (ht₀.trans (hs.trans hu.1)) (hs.trans hu.1)).trans_lt hε₀)
          rwa [abs_of_pos (sub_pos.2 (hgap u (hs.trans hu.1)))] at hl
      rw [hfy, hfy, add_sub_add_right_eq_sub] at h
      rw [le_div_iff₀ hκ, mul_comm]
      have := Real.rpow_nonneg (sub_nonneg.2 (hfz' t (ht₀.trans (hs.trans hst)))) (1 - a)
      linarith
  refine Landscape.Lojasiewicz.converges_of_displacement_bound x
    (fun t => (f (x t) - f z) ^ (1 - a) / (m * (1 - a))) t₀ ?_ hlen
  simpa only [zero_div] using hφ.div_const (m * (1 - a))

/-- **A gradient flow converges to a limit point where the Łojasiewicz
inequality holds**, under the hypotheses of
`exists_tendsto_gradientFlow_of_isLojasiewiczAt`. -/
theorem tendsto_gradientFlow_of_isLojasiewiczAt (hf : Differentiable ℝ f)
    (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t) (ha1 : a < 1) {z : E}
    (hz : IsLojasiewiczAt f z a) (hcl : MapClusterPt z atTop x) :
    Tendsto x atTop (𝓝 z) := by
  have hfreq : ∀ ε > 0, ∃ᶠ t in atTop, x t ∈ ball z ε := fun ε hε =>
    mapClusterPt_iff_frequently.1 hcl _ (ball_mem_nhds z hε)
  obtain ⟨w, hw⟩ := exists_tendsto_gradientFlow_of_isLojasiewiczAt hf hx (Λ := {0}) (by simp)
    ha1 hz fun ε hε => (hfreq ε hε).mono fun t ht =>
      ⟨0, Set.mem_singleton 0, by rw [add_zero, ← dist_eq_norm]; exact mem_ball.1 ht⟩
  have hwz : w = z := eq_of_forall_dist_le fun ε hε => by
    obtain ⟨t, ht, htw⟩ := ((hfreq (ε / 2) (half_pos hε)).and_eventually
      (Metric.tendsto_nhds.1 hw _ (half_pos hε))).exists
    rw [mem_ball] at ht
    linarith [dist_triangle_left w z (x t)]
  rwa [hwz] at hw

/-- The hypotheses of the theorems above hold for `f(x) = ‖x‖² / 2` on the
line, with `a = 1/2`, `Λ = {0}` and the critical point `0`: for the gradient
flow `e^{-t}`, which converges to `0`, and for the constant trajectory `0`,
along which `f` is constant. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    let x := fun t : ℝ => Real.exp (-t)
    Differentiable ℝ f ∧ (∀ t, HasDerivAt x (-gradient f (x t)) t) ∧
      (∀ v ∈ ({0} : Set ℝ), ∀ y, f (y + v) = f y) ∧ (1 / 2 : ℝ) < 1 ∧
      IsLojasiewiczAt f 0 (1 / 2) ∧ (∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ ({0} : Set ℝ),
        ‖x t + v - 0‖ < ε) ∧ MapClusterPt 0 atTop x ∧
      (∀ t, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (-gradient f 0) t) ∧
      ∀ t : ℝ, 0 ≤ t → f 0 = f 0 := by
  intro f x
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hlim : Tendsto x atTop (𝓝 0) := Real.tendsto_exp_neg_atTop_nhds_zero
  refine ⟨(contDiff_const.mul (contDiff_norm_sq ℝ)).differentiable one_ne_zero, fun t => ?_,
    by simp, by norm_num, isLojasiewiczAt_half_mul_norm_sq, fun ε hε => ?_, hlim.mapClusterPt,
    fun t => ?_, fun _ _ => rfl⟩
  · rw [hg]
    simpa [x] using (hasDerivAt_neg t).exp
  · exact (Metric.tendsto_nhds.1 hlim ε hε).frequently.mono fun t ht =>
      ⟨0, Set.mem_singleton 0, by rwa [add_zero, ← dist_eq_norm]⟩
  · rw [hg, id, neg_zero]
    exact hasDerivAt_const t 0

end Transformer.Landscape.SaddleAvoidance
