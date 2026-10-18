/-
# Local uniformization: operations on monomial charts

One chart from normal crossings; restriction of charts to a neighbourhood;
invariance under equality of germs; composition with an analytic map; and the
local-to-global principle: charts for `f ∘ Φ` at every point of the compact
domains of finitely many analytic maps `Φ` give charts for `f`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Uniformization.NormalCrossing

open Filter Set
open scoped Topology

namespace Transformer.Uniformization

open Transformer.AnalyticPreparation (Base)

variable {n : ℕ}

/-- Normal crossings at a point gives one chart: the identity on a small
closed ball. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_of_normalCrossingAt {f : Base n → ℝ} {w : Base n}
    (h : NormalCrossingAt f w) : HasMonomialCharts f w := by
  obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff_ball.1 h.eventually
  refine ⟨{(id, Metric.closedBall w (r / 2))}, finite_singleton _, ?_, ?_⟩
  · rintro c rfl
    exact ⟨isCompact_closedBall _ _, fun x _ => analyticAt_id,
      fun x hx => hball x (Metric.closedBall_subset_ball (half_lt_self hr) hx)⟩
  · simpa only [mem_singleton_iff, iUnion_iUnion_eq_left, image_id] using
      Metric.closedBall_mem_nhds w (half_pos hr)

/-- Charts can be chosen with images inside any neighbourhood of the point.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem HasMonomialCharts.shrink {f : Base n → ℝ} {z : Base n} (h : HasMonomialCharts f z)
    {V : Set (Base n)} (hV : V ∈ 𝓝 z) :
    ∃ S : Set ((Base n → Base n) × Set (Base n)), S.Finite ∧
      (∀ c ∈ S, IsMonomialChart f c ∧ c.1 '' c.2 ⊆ V) ∧ (⋃ c ∈ S, c.1 '' c.2) ∈ 𝓝 z := by
  obtain ⟨S, hS, hc, hcover⟩ := h
  obtain ⟨r, hr, hrV⟩ := Metric.mem_nhds_iff.1 hV
  let B := Metric.closedBall z (r / 2)
  refine ⟨(fun c => (c.1, c.2 ∩ c.1 ⁻¹' B)) '' S, hS.image _, ?_, ?_⟩
  · rintro _ ⟨c, hcS, rfl⟩
    obtain ⟨hK, hA, hN⟩ := hc c hcS
    refine ⟨⟨hK.of_isClosed_subset (hA.continuousOn.preimage_isClosed_of_isClosed hK.isClosed
      Metric.isClosed_closedBall) inter_subset_left, hA.mono inter_subset_left,
      fun w hw => hN w hw.1⟩, ?_⟩
    rintro _ ⟨x, hx, rfl⟩
    exact hrV (Metric.closedBall_subset_ball (half_lt_self hr) hx.2)
  · rw [biUnion_image]
    refine mem_of_superset (inter_mem hcover (Metric.closedBall_mem_nhds z (half_pos hr))) ?_
    rintro y ⟨hy, hyB⟩
    obtain ⟨c, hcS, x, hx, rfl⟩ := mem_iUnion₂.1 hy
    exact mem_iUnion₂.2 ⟨c, hcS, x, ⟨hx, hyB⟩, rfl⟩

/-- Monomial charts depend only on the germ of the function. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem HasMonomialCharts.congr {f g : Base n → ℝ} {z : Base n} (h : HasMonomialCharts f z)
    (hfg : f =ᶠ[𝓝 z] g) : HasMonomialCharts g z := by
  obtain ⟨V, hVfg, hVo, hzV⟩ := eventually_nhds_iff.1 hfg
  obtain ⟨S, hS, hc, hcover⟩ := h.shrink (hVo.mem_nhds hzV)
  refine ⟨S, hS, fun c hcS => ?_, hcover⟩
  obtain ⟨⟨hK, hA, hN⟩, hsub⟩ := hc c hcS
  refine ⟨hK, hA, fun w hw => (hN w hw).congr ?_⟩
  filter_upwards [(hA w hw).continuousAt.preimage_mem_nhds
    (hVo.mem_nhds (hsub ⟨w, hw, rfl⟩))] with x hx
  exact hVfg _ hx

/-- Charts for `f ∘ Φ` at `w`, with `Φ` analytic near `w`, compose to charts
for `f` whose images cover the image of a neighbourhood of `w`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem HasMonomialCharts.exists_comp {f : Base n → ℝ} {Φ : Base n → Base n} {w : Base n}
    (h : HasMonomialCharts (f ∘ Φ) w) (hΦ : ∀ᶠ x in 𝓝 w, AnalyticAt ℝ Φ x) :
    ∃ S : Set ((Base n → Base n) × Set (Base n)), S.Finite ∧
      (∀ c ∈ S, IsMonomialChart f c) ∧ ∃ N ∈ 𝓝 w, Φ '' N ⊆ ⋃ c ∈ S, c.1 '' c.2 := by
  obtain ⟨S, hS, hc, hcover⟩ := h.shrink hΦ
  refine ⟨(fun c => (Φ ∘ c.1, c.2)) '' S, hS.image _, ?_, _, hcover, ?_⟩
  · rintro _ ⟨c, hcS, rfl⟩
    obtain ⟨⟨hK, hA, hN⟩, hsub⟩ := hc c hcS
    refine ⟨hK, fun x hx => ?_, hN⟩
    have h1 : AnalyticAt ℝ Φ (c.1 x) := hsub ⟨x, hx, rfl⟩
    exact h1.comp (hA x hx)
  · rw [biUnion_image]
    rintro _ ⟨y, hy, rfl⟩
    obtain ⟨c, hcS, x, hx, rfl⟩ := mem_iUnion₂.1 hy
    exact mem_iUnion₂.2 ⟨c, hcS, x, hx, rfl⟩

/-- Local-to-global: charts for `f ∘ c.1` at every point of the compact
domain `c.2` give finitely many charts for `f` covering `c.1 '' c.2`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_monomialCharts_of_compact {f : Base n → ℝ}
    (c : (Base n → Base n) × Set (Base n)) (hK : IsCompact c.2)
    (hA : AnalyticOnNhd ℝ c.1 c.2) (hH : ∀ w ∈ c.2, HasMonomialCharts (f ∘ c.1) w) :
    ∃ T : Set ((Base n → Base n) × Set (Base n)), T.Finite ∧
      (∀ d ∈ T, IsMonomialChart f d) ∧ c.1 '' c.2 ⊆ ⋃ d ∈ T, d.1 '' d.2 := by
  have hloc : ∀ w ∈ c.2, ∃ T : Set ((Base n → Base n) × Set (Base n)), T.Finite ∧
      (∀ d ∈ T, IsMonomialChart f d) ∧ ∃ N ∈ 𝓝 w, c.1 '' N ⊆ ⋃ d ∈ T, d.1 '' d.2 :=
    fun w hw => (hH w hw).exists_comp (hA w hw).eventually_analyticAt
  choose! T hT hTc N hN hNT using hloc
  obtain ⟨t, ht⟩ := hK.elim_nhds_subcover' (fun w _ => N w) (fun w hw => hN w hw)
  refine ⟨⋃ x ∈ (t : Set c.2), T x, t.finite_toSet.biUnion fun x _ => hT x x.2, ?_, ?_⟩
  · intro d hd
    obtain ⟨x, -, hd⟩ := mem_iUnion₂.1 hd
    exact hTc x x.2 d hd
  · rintro _ ⟨y, hy, rfl⟩
    obtain ⟨x, hxt, hyx⟩ := mem_iUnion₂.1 (ht hy)
    obtain ⟨d, hdT, hd⟩ := mem_iUnion₂.1 (hNT x x.2 ⟨y, hyx, rfl⟩)
    exact mem_iUnion₂.2 ⟨d, mem_iUnion₂.2 ⟨x, hxt, hdT⟩, hd⟩

/-- Local-to-global: finitely many analytic maps with compact domains whose
images cover a neighbourhood of `z`, with charts for `f ∘ c.1` at every point
of every domain, give charts for `f` at `z`. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_of_cover {f : Base n → ℝ} {z : Base n}
    (S : Set ((Base n → Base n) × Set (Base n))) (hS : S.Finite)
    (hc : ∀ c ∈ S, IsCompact c.2 ∧ AnalyticOnNhd ℝ c.1 c.2 ∧
      ∀ w ∈ c.2, HasMonomialCharts (f ∘ c.1) w)
    (hcover : (⋃ c ∈ S, c.1 '' c.2) ∈ 𝓝 z) : HasMonomialCharts f z := by
  have key := fun c (hcS : c ∈ S) =>
    exists_monomialCharts_of_compact c (hc c hcS).1 (hc c hcS).2.1 (hc c hcS).2.2
  choose! T hT hTc hTcover using key
  refine ⟨⋃ c ∈ S, T c, hS.biUnion hT, ?_, mem_of_superset hcover ?_⟩
  · intro d hd
    obtain ⟨c, hcS, hd⟩ := mem_iUnion₂.1 hd
    exact hTc c hcS d hd
  · intro y hy
    obtain ⟨c, hcS, hyc⟩ := mem_iUnion₂.1 hy
    obtain ⟨d, hdT, hd⟩ := mem_iUnion₂.1 (hTcover c hcS hyc)
    exact mem_iUnion₂.2 ⟨d, mem_iUnion₂.2 ⟨c, hcS, hdT⟩, hd⟩

/-- Charts for `f ∘ T` at `w` give charts for `f` at `T w` when `T` is
analytic near `w` and has a right inverse continuous at `T w` sending `T w`
to `w`. Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem HasMonomialCharts.comp {f : Base n → ℝ} {T T' : Base n → Base n} {w : Base n}
    (h : HasMonomialCharts (f ∘ T) w) (hT : ∀ᶠ x in 𝓝 w, AnalyticAt ℝ T x)
    (hT' : ContinuousAt T' (T w)) (hT'w : T' (T w) = w)
    (hTT' : ∀ᶠ y in 𝓝 (T w), T (T' y) = y) : HasMonomialCharts f (T w) := by
  obtain ⟨S, hS, hc, N, hN, hNS⟩ := h.exists_comp hT
  refine ⟨S, hS, hc, mem_of_superset ?_ hNS⟩
  have hN' : N ∈ 𝓝 (T' (T w)) := by rw [hT'w]; exact hN
  filter_upwards [hT'.preimage_mem_nhds hN', hTT'] with y hy hyy
  exact ⟨T' y, hy, hyy⟩

/-- Charts at `0` for `x ↦ f (x + z)` give charts for `f` at `z`. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem hasMonomialCharts_of_add {f : Base n → ℝ} {z : Base n}
    (h : HasMonomialCharts (fun x => f (x + z)) 0) : HasMonomialCharts f z := by
  have h' := HasMonomialCharts.comp (T := fun x => x + z) (T' := fun y => y - z) h
    (Eventually.of_forall fun x => analyticAt_id.fun_add analyticAt_const)
    (continuous_sub_right z).continuousAt (by simp)
    (Eventually.of_forall fun y => sub_add_cancel y z)
  simpa only [zero_add] using h'

end Transformer.Uniformization
