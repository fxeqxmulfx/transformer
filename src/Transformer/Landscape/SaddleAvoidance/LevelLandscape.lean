module

public import Transformer.Landscape.SaddleAvoidance.Landscape
public import Transformer.Landscape.SaddleAvoidance.SecondOrderCritical

/-!
# Saddle avoidance — the landscape below a level

Landscape theorems usually describe only the critical points below a level
`c`: every critical point `z` with `f z < c` is a global minimizer or a
strict saddle. Gradient descent with `0 < η` and `η L < 1` does not raise
`f` (`antitone_iterate`), so an orbit starting below `c` stays below it, and
so does every limit point of the orbit up to translations leaving `f`
invariant (`le_of_frequently_norm_add_sub_lt`). The landscape hypothesis
`hcrit` and the Łojasiewicz inequality `hloj` are therefore needed only
below `c`. Almost every start below `c` whose orbit has a limit point up to
translations converges to a critical point, no higher than the start, that
is not a strict saddle (`ae_exists_tendsto_not_isStrictSaddle_of_lt`); hence
to a global minimizer (`ae_exists_tendsto_isMinOn_of_lt`). When the
translations span `E`, every orbit has such a limit point
(`ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top`). `FlowLevelLandscape`
has the same for gradient flows, and `AnalyticLevelLandscape` the forms
without `hloj` for analytic `f`.

Source: item M6.3 of `landscape_plan.md`, the theorems of `Landscape` with
their hypotheses asked only below `c`.
-/

@[expose] public section

open Filter Set MeasureTheory
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L : ℝ≥0} {η : ℝ}

/-- **Below a level, gradient descent almost surely converges to a critical
point that is not a strict saddle**, if `f` satisfies the Łojasiewicz
inequality with an exponent `0 ≤ a < 1` at each critical point below `c`,
and almost every orbit starting below `c` has a limit point up to
translations in a set `Λ` leaving `f` invariant. The limit is no higher than
the start. This is `ae_exists_tendsto_not_isStrictSaddle` with `hloj` and
`hcl` asked only below `c`. -/
theorem ae_exists_tendsto_not_isStrictSaddle_of_lt (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, f x < c → ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ,
      ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) ∧
      gradient f w = 0 ∧ f w ≤ f x ∧ ¬IsStrictSaddle f w := by
  have hd : Differentiable ℝ f := hf.differentiable (by norm_num)
  filter_upwards [hcl, ae_not_tendsto_of_isStrictSaddle μ hν hf hL hη hηL] with x hcl hx hxc
  obtain ⟨z, hz⟩ := hcl hxc
  have hb : ∀ᶠ k in atTop, f ((gradientStep f η)^[k] x) ≤ f x :=
    Eventually.of_forall fun k => antitone_iterate hd hL hη.le hηL.le x (Nat.zero_le k)
  obtain ⟨a, ha0, ha1, hza⟩ : ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
    by_cases h0 : gradient f z = 0
    · exact hloj z h0
        ((le_of_frequently_norm_add_sub_lt hΛ hd.continuous.continuousAt hz hb).trans_lt hxc)
    · exact ⟨0, le_rfl, one_pos,
        isLojasiewiczAt_of_gradient_ne_zero hL.continuous.continuousAt h0 le_rfl⟩
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt hd hL hη hηL hΛ ha0 ha1 hza hz
  exact ⟨w, hw, gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw,
    le_of_tendsto ((hd.continuous.tendsto w).comp hw) hb, fun hs => hx w hs hw⟩

/-- **Below a level, gradient descent almost surely converges to a global
minimizer**, if every critical point below `c` is a global minimizer or a
strict saddle and satisfies the Łojasiewicz inequality with an exponent
`0 ≤ a < 1`, and almost every orbit starting below `c` has a limit point up
to translations in a set `Λ` leaving `f` invariant. This is
`ae_exists_tendsto_isMinOn` for the starts below `c`, with `hcrit`, `hloj`
and `hcl` asked only below `c`. -/
theorem ae_exists_tendsto_isMinOn_of_lt (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E}
    (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f)) (hη : 0 < η)
    (hηL : η * L < 1) {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, f x < c → ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ,
      ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  filter_upwards [ae_exists_tendsto_not_isStrictSaddle_of_lt μ hν hf hL hη hηL hΛ hloj hcl]
    with x hx hxc
  obtain ⟨w, hw, hw0, hwx, hws⟩ := hx hxc
  exact ⟨w, (hcrit w hw0 (hwx.trans_lt hxc)).resolve_right hws, hw⟩

/-- **Below a level, gradient descent on a periodic function almost surely
converges to a global minimizer**, if every critical point below `c` is a
global minimizer or a strict saddle and satisfies the Łojasiewicz
inequality. The translations `Λ` leaving `f` invariant span `E`, so every
orbit has a limit point up to their integer combinations. -/
theorem ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E} (hspan : Submodule.span ℝ Λ = ⊤)
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
  refine ae_exists_tendsto_isMinOn_of_lt μ hν hf hL hη hηL (Λ := (Submodule.span ℤ Λ : Set E))
    (fun v hv => periodic_of_mem_span hΛ hv) hcrit hloj (Eventually.of_forall fun x _ => ?_)
  obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : E) R)
    (Λ := (Submodule.span ℤ Λ : Set E)) (u := fun k => (gradientStep f η)^[k] x)
    fun k => hR _
  exact ⟨z, hz⟩

/-- The hypotheses of the theorems above hold for
`f(x) = 2 - cos x - cos² x` on the line with Lebesgue measure, `L = 3`,
`η = 1/4`, the period `2π` or its integer multiples, and `c = 2`, while the
hypothesis `hcrit` of `ae_exists_tendsto_isMinOn` fails. The gradient is
`sin x + sin 2x = sin x (1 + 2 cos x)`. Below `2`, where `cos x > 0`, the
critical points are the global minimizers, where `cos x = 1` and the Hessian
`cos x + 2 cos 2x` is `3`. At `π`, where `f = 2`, lies a local minimizer
that is neither a global one nor a strict saddle. -/
example : let f := fun x : ℝ => 2 - Real.cos x - Real.cos x ^ 2
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧ ContDiff ℝ 2 f ∧
      LipschitzWith 3 (gradient f) ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) * ((3 : ℝ≥0) : ℝ) < 1 ∧
      Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → f z < 2 → IsMinOn f univ z ∨ IsStrictSaddle f z) ∧
      (∀ z, gradient f z = 0 → f z < 2 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) ∧
      (∀ᵐ x ∂(volume : Measure ℝ), f x < 2 → ∃ z, ∀ ε > 0, ∃ᶠ k in atTop,
        ∃ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ),
          ‖(gradientStep f (1 / 4))^[k] x + v - z‖ < ε) ∧
      ¬∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z := by
  intro f
  have hf : ContDiff ℝ 2 f := ((analyticOnNhd_const.fun_sub Real.analyticOnNhd_cos).fun_sub
    (Real.analyticOnNhd_cos.fun_pow 2)).contDiff
  have hd : Differentiable ℝ f := hf.differentiable (by norm_num)
  have hg : gradient f = fun x => Real.sin x + Real.sin (2 * x) := funext fun x => by
    rw [gradient_eq_deriv']
    exact (((Real.hasDerivAt_cos x).const_sub 2).fun_sub
      ((Real.hasDerivAt_cos x).fun_pow 2)).deriv.trans (by rw [Real.sin_two_mul]; norm_num; ring)
  have hH : ∀ z, HasFDerivAt (gradient f)
      (ContinuousLinearMap.toSpanSingleton ℝ (Real.cos z + 2 * Real.cos (2 * z))) z :=
    fun z => by
      rw [hg]
      exact (((Real.hasDerivAt_sin z).fun_add ((hasDerivAt_id' z).const_mul 2).sin).congr_deriv
        (by ring)).hasFDerivAt
  have hspan : Submodule.span ℝ {2 * Real.pi} = ⊤ :=
    Ideal.span_singleton_eq_top.2 (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne')
  have hΛ : ∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y := fun v hv y => by
    simp [mem_singleton_iff.1 hv, f]
  have hkey : ∀ z, gradient f z = 0 → f z < 2 → Real.cos z = 1 := fun z hz hc => by
    simp only [hg, Real.sin_two_mul] at hz
    have hcos : 0 < Real.cos z := by
      by_contra h
      push Not at h
      simp only [f] at hc
      nlinarith [Real.neg_one_le_cos z]
    have hs : Real.sin z = 0 := by
      rcases mul_eq_zero.1 (show Real.sin z * (1 + 2 * Real.cos z) = 0 by linear_combination hz)
        with h | h
      · exact h
      · linarith
    nlinarith [Real.sin_sq_add_cos_sq z, Real.cos_le_one z]
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl, hf, ?_, by norm_num, by norm_num,
    hspan, hΛ, fun v hv => periodic_of_mem_span hΛ hv, fun z hz hc => Or.inl ?_,
    fun z hz hc => ⟨1 / 2, by norm_num, by norm_num, ?_⟩, Eventually.of_forall fun x _ => ?_, ?_⟩
  · rw [hg]
    refine (Real.lipschitzWith_sin.add (Real.lipschitzWith_sin.comp
      (lipschitzWith_smul (2 : ℝ)))).weaken ?_
    rw [← NNReal.coe_le_coe]
    norm_num
  · refine isMinOn_univ_iff.2 fun y => ?_
    simp only [f, hkey z hz hc]
    nlinarith [Real.cos_le_one y, Real.neg_one_le_cos y]
  · refine isLojasiewiczAt_of_injective hd (hH z) hz fun a b hab => ?_
    rw [Real.cos_two_mul, hkey z hz hc] at hab
    simp only [ContinuousLinearMap.toSpanSingleton_apply, smul_eq_mul] at hab
    linarith
  · obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
    obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : ℝ) R)
      (Λ := (Submodule.span ℤ {2 * Real.pi} : Set ℝ))
      (u := fun k => (gradientStep f (1 / 4))^[k] x) fun k => hR _
    exact ⟨z, hz⟩
  · intro h
    have hmin : IsLocalMin f Real.pi :=
      Filter.eventually_of_mem (Icc_mem_nhds (by linarith [Real.pi_pos] : Real.pi / 2 < Real.pi)
        (by linarith [Real.pi_pos] : Real.pi < Real.pi + Real.pi / 2)) fun x hx => by
          have := Real.cos_nonpos_of_pi_div_two_le_of_le hx.1 hx.2
          simp only [f, Real.cos_pi]
          nlinarith [Real.neg_one_le_cos x]
    rcases h Real.pi (by simp [hg]) with h | h
    · have := isMinOn_univ_iff.1 h 0
      norm_num [f] at this
    · obtain ⟨v, hv⟩ := h.exists_inner_neg
      exact (inner_nonneg_of_isLocalMin hd (hH Real.pi).differentiableAt.hasFDerivAt hmin
        v).not_gt hv

end Transformer.Landscape.SaddleAvoidance
