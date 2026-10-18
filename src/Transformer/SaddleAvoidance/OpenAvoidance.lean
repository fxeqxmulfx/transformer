/-
# Saddle avoidance — flows `ẋ = -G(x) ∇f(x)` on an open set

Let `f` be `C²` and `G` be `C¹` on an open set `U` of a finite-dimensional real
inner product space, with `G` symmetric positive definite on `U`.  For almost every
start, no solution of `ẋ = -G(x) ∇f(x)` on `[0, ∞)` that stays in a compact `K ⊆ U`
converges to a strict saddle of `f` (`ae_not_tendsto_metricGradient_of_isCompact`),
and no solution on `[0, ∞)` that stays in `U` converges to a strict saddle in `U`
(`ae_not_tendsto_metricGradient_of_isOpen`).

For the first, a cutoff `χ` equal to `1` near `K` with compact support in `U`
(`exists_cutoff`) gives `F = χ f`, of class `C²` on the whole space, and the field
`G' = χ G + (1 - χ) I`, of class `C¹` and symmetric positive definite everywhere.
`G' ∇F` is `C¹` with compact support, hence bounded and Lipschitz, so
`ae_not_tendsto_metricGradientFlow_of_isStrictSaddle` applies to `F` and `G'`.  Near
`K` they are `f` and `G`, so by forward uniqueness (`ODE_solution_unique`) a solution
in `K` is a trajectory of the flow of `-G' ∇F` for `t ≥ 0`; its limit lies in `K`,
where `F` and `f` have the same strict saddles (`IsStrictSaddle.congr_of_eventuallyEq`).
For the second, a solution in `U` converging to `z ∈ U` stays in the compact set made
of its arc up to some time and a closed ball around `z` inside `U`, so in one of the
countably many compact sets of `exists_isCompact_cover`.

This is item M6.7 of `landscape_plan.md`: the cutoff of
`Perspective.ae_not_tendsto_of_cutoff`, for a general metric.  The plan takes an open
`R` with compact closure in `U` and limits in `R`, the case `K = closure R` of the
first theorem; the second is its form for a countable cover of `U`.  The source is
`l:nosaddleconv` in Appendix A of arXiv:2312.10794v5, there for gradient ascent of
`𝖤_β` on the whole space.
-/

import Transformer.SaddleAvoidance.Cutoff
import Transformer.SaddleAvoidance.MetricAvoidance

open Filter MeasureTheory Metric Set
open scoped Topology

namespace Transformer.SaddleAvoidance

open GlobalFlow

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {U K : Set E} {f : E → ℝ} {G : E → E →L[ℝ] E}

/-- **Almost no solution in a compact set converges to a strict saddle.**  For `f` of
class `C²` and `G` of class `C¹` on an open `U`, `G` symmetric positive definite on
`U`, a compact `K ⊆ U` and `ν` absolutely continuous with respect to a Haar measure:
for `ν`-almost every `y`, no solution of `ẋ = -G(x) ∇f(x)` on `[0, ∞)` from `y` that
stays in `K` converges to a strict saddle of `f`. -/
theorem ae_not_tendsto_metricGradient_of_isCompact (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hU : IsOpen U) (hK : IsCompact K) (hKU : K ⊆ U)
    (hf : ∀ x ∈ U, ContDiffAt ℝ 2 f x) (hG : ∀ x ∈ U, ContDiffAt ℝ 1 G x)
    (hpos : ∀ x ∈ U, (G x : E →ₗ[ℝ] E).IsSymmetric ∧ ∀ v, v ≠ 0 → 0 < ⟪G x v, v⟫) :
    ∀ᵐ y ∂ν, ∀ Y : ℝ → E, Y 0 = y → (∀ t, 0 ≤ t → Y t ∈ K) →
      (∀ t, 0 ≤ t → HasDerivWithinAt Y (-G (Y t) (gradient f (Y t))) (Ici 0) t) →
      ∀ z, IsStrictSaddle f z → ¬Tendsto Y atTop (𝓝 z) := by
  obtain ⟨χ, hχ, hχ01, hχc, hχU, V, hV, hKV, hχV⟩ := exists_cutoff hK hU hKU 2
  have hχ1 : ContDiff ℝ 1 χ := hχ.of_le (by norm_num)
  set F : E → ℝ := fun x => χ x • f x
  set G' : E → E →L[ℝ] E := fun x => χ x • G x + (1 - χ x) • ContinuousLinearMap.id ℝ E
  have hF : ContDiff ℝ 2 F := contDiff_smul_of_tsupport_subset hχ hχU hf
  have hG' : ContDiff ℝ 1 G' := (contDiff_smul_of_tsupport_subset hχ1 hχU hG).add
    ((contDiff_const.sub hχ1).smul contDiff_const)
  have hF0 : ∀ x, x ∉ tsupport χ → gradient F x = 0 := fun x hx => by
    have h : F =ᶠ[𝓝 x] fun _ => 0 := by
      filter_upwards [notMem_tsupport_iff_eventuallyEq.1 hx] with y hy
      simp [F, show χ y = 0 from hy]
    rw [h.gradient_eq, gradient_fun_const]
  have hVc : ContDiff ℝ 1 fun x => G' x (gradient F x) := hG'.clm_apply (contDiff_gradient hF)
  have hVs : HasCompactSupport fun x => G' x (gradient F x) :=
    hχc.mono' fun x hx => by_contra fun h => hx (by simp only [hF0 x h, map_zero])
  obtain ⟨L, hL⟩ := hVc.lipschitzWith_of_hasCompactSupport hVs one_ne_zero
  obtain ⟨M, hM⟩ := hVs.exists_bound_of_continuous hVc.continuous
  have happ : ∀ x u, G' x u = χ x • G x u + (1 - χ x) • u := fun x u => by simp [G']
  have hpos' : ∀ x, (G' x : E →ₗ[ℝ] E).IsSymmetric ∧ ∀ v, v ≠ 0 → 0 < ⟪G' x v, v⟫ := by
    intro x
    by_cases hx : χ x = 0
    · refine ⟨fun u w => ?_, fun v hv => ?_⟩
      · change ⟪G' x u, w⟫ = ⟪u, G' x w⟫
        simp [happ, hx]
      · simpa [happ, hx] using real_inner_self_pos.2 hv
    obtain ⟨hsymm, hposx⟩ := hpos x (hχU (subset_tsupport χ hx))
    refine ⟨fun u w => ?_, fun v hv => ?_⟩
    · change ⟪G' x u, w⟫ = ⟪u, G' x w⟫
      have h := hsymm u w
      simp only [ContinuousLinearMap.coe_coe] at h
      rw [happ, happ, inner_add_left, inner_add_right, real_inner_smul_left,
        real_inner_smul_left, real_inner_smul_right, real_inner_smul_right, h]
    · rw [happ, inner_add_left, real_inner_smul_left, real_inner_smul_left]
      have h1 := mul_pos (lt_of_le_of_ne (hχ01 x).1 (Ne.symm hx)) (hposx v hv)
      have h2 := mul_nonneg (sub_nonneg.2 (hχ01 x).2) (real_inner_self_pos.2 hv).le
      linarith
  have hae := ae_not_tendsto_metricGradientFlow_of_isStrictSaddle μ hν hF hG' hL hM
    fun x _ => hpos' x
  filter_upwards [hae] with y hy
  intro Y hY0 hYK hY z hz hlim
  have hFV : ∀ x ∈ V, F =ᶠ[𝓝 x] f := fun x hx => by
    filter_upwards [hV.mem_nhds hx] with x' hx'
    simp [F, hχV x' hx']
  have hYF : ∀ t, 0 ≤ t → HasDerivWithinAt Y (-G' (Y t) (gradient F (Y t))) (Ici 0) t := by
    intro t ht
    have hYV := hKV (hYK t ht)
    rw [(hFV _ hYV).gradient_eq]
    convert hY t ht using 2
    simp [G', hχV _ hYV]
  have hflow : ∀ t, 0 ≤ t → Y t = metricGradientFlow G' F t y := fun t ht =>
    ODE_solution_unique (v := fun _ x => -G' x (gradient F x)) (fun _ => hL.neg)
      (fun s hs => (hYF s hs.1).continuousWithinAt.mono Icc_subset_Ici_self)
      (fun s hs => (hYF s hs.1).mono (Ici_subset_Ici.2 hs.1))
      (fun s _ => (hasDerivAt_metricGradientFlow hL hM s y).continuousAt.continuousWithinAt)
      (fun s _ => (hasDerivAt_metricGradientFlow hL hM s y).hasDerivWithinAt)
      (by rw [hY0]; exact ((isFlowField_neg_metricGradient hL hM).flow_self 0 y).symm)
      ⟨ht, le_rfl⟩
  have hzK : z ∈ K := hK.isClosed.mem_of_tendsto hlim (eventually_atTop.2 ⟨0, hYK⟩)
  refine hy z (hz.congr_of_eventuallyEq (hFV z (hKV hzK))) (hlim.congr' ?_)
  filter_upwards [eventually_ge_atTop 0] with t ht
  exact hflow t ht

/-- **Almost no solution in an open set converges to a strict saddle in it.**  For `f`
of class `C²` and `G` of class `C¹` on an open `U`, `G` symmetric positive definite on
`U`, and `ν` absolutely continuous with respect to a Haar measure: for `ν`-almost
every `y`, no solution of `ẋ = -G(x) ∇f(x)` on `[0, ∞)` from `y` that stays in `U`
converges to a strict saddle of `f` in `U`. -/
theorem ae_not_tendsto_metricGradient_of_isOpen (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hU : IsOpen U)
    (hf : ∀ x ∈ U, ContDiffAt ℝ 2 f x) (hG : ∀ x ∈ U, ContDiffAt ℝ 1 G x)
    (hpos : ∀ x ∈ U, (G x : E →ₗ[ℝ] E).IsSymmetric ∧ ∀ v, v ≠ 0 → 0 < ⟪G x v, v⟫) :
    ∀ᵐ y ∂ν, ∀ Y : ℝ → E, Y 0 = y → (∀ t, 0 ≤ t → Y t ∈ U) →
      (∀ t, 0 ≤ t → HasDerivWithinAt Y (-G (Y t) (gradient f (Y t))) (Ici 0) t) →
      ∀ z ∈ U, IsStrictSaddle f z → ¬Tendsto Y atTop (𝓝 z) := by
  obtain ⟨K, hK, hKU, hcover⟩ := exists_isCompact_cover hU
  filter_upwards [ae_all_iff.2 fun k => ae_not_tendsto_metricGradient_of_isCompact μ hν hU
    (hK k) (hKU k) hf hG hpos] with y hy
  intro Y hY0 hYU hY z hzU hz hlim
  obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.1 hU z hzU
  obtain ⟨T, hT⟩ := eventually_atTop.1 (hlim.eventually (closedBall_mem_nhds z (half_pos hε)))
  have hcont : ContinuousOn Y (Icc 0 (max T 0)) := fun t ht =>
    (hY t ht.1).continuousWithinAt.mono Icc_subset_Ici_self
  obtain ⟨k, hk⟩ := hcover (Y '' Icc 0 (max T 0) ∪ closedBall z (ε / 2))
    ((isCompact_Icc.image_of_continuousOn hcont).union (isCompact_closedBall z _))
    (union_subset (image_subset_iff.2 fun t ht => hYU t ht.1)
      ((closedBall_subset_ball (half_lt_self hε)).trans hball))
  refine hy k Y hY0 (fun t ht => hk ?_) hY z hz hlim
  rcases le_total t (max T 0) with htT | htT
  · exact Or.inl ⟨t, ⟨ht, htT⟩, rfl⟩
  · exact Or.inr (hT t ((le_max_left T 0).trans htT))

/-- The hypotheses of `ae_not_tendsto_metricGradient_of_isCompact` hold for `f = log`
and `G = 1` on `U = (0, ∞)`, where `log` is smooth though not on the whole line, with
`K = [1, 2]` and Lebesgue measure. -/
example := ae_not_tendsto_metricGradient_of_isCompact (volume : Measure ℝ)
  Measure.AbsolutelyContinuous.rfl (U := Ioi 0) isOpen_Ioi (isCompact_Icc (a := 1) (b := 2))
  (fun _ hx => lt_of_lt_of_le one_pos hx.1) (f := Real.log)
  (fun _ hx => Real.contDiffAt_log.2 (ne_of_gt hx)) (G := fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ _ => contDiffAt_const)
  fun _ _ => ⟨LinearMap.IsSymmetric.id, fun _ hv => by simpa using real_inner_self_pos.2 hv⟩

/-- The hypotheses of `ae_not_tendsto_metricGradient_of_isOpen` hold for `f = log` and
`G = 1` on `U = (0, ∞)`, with Lebesgue measure. -/
example := ae_not_tendsto_metricGradient_of_isOpen (volume : Measure ℝ)
  Measure.AbsolutelyContinuous.rfl (U := Ioi 0) isOpen_Ioi (f := Real.log)
  (fun _ hx => Real.contDiffAt_log.2 (ne_of_gt hx)) (G := fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ _ => contDiffAt_const)
  fun _ _ => ⟨LinearMap.IsSymmetric.id, fun _ hv => by simpa using real_inner_self_pos.2 hv⟩

end Transformer.SaddleAvoidance
