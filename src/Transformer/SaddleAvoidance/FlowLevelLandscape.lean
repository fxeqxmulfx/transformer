/-
# Saddle avoidance — gradient flows below a level

The gradient flow does not raise `f` (`antitoneOn_comp_gradientFlow`), so
the theorems of `LevelLandscape` hold for it.  If `f` satisfies the
Łojasiewicz inequality at each critical point below `c`, almost every
trajectory starting below `c` with a limit point up to translations leaving
`f` invariant converges to a critical point, no higher than the start, that
is not a strict saddle
(`ae_exists_tendsto_gradientFlow_not_isStrictSaddle_of_lt`); to a global
minimizer if every critical point below `c` is a global minimizer or a
strict saddle (`ae_exists_tendsto_gradientFlow_isMinOn_of_lt`).  When the
translations span `E`, neither the limit point nor bounds on `∇f` are
needed (`ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_span_eq_top`).

Source: item M6.3 of `landscape_plan.md`, the theorems of `FlowLandscape`
with their hypotheses asked only below `c`.
-/

import Transformer.SaddleAvoidance.FlowLandscape
import Transformer.SaddleAvoidance.LevelLandscape

open Filter Set MeasureTheory
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L : ℝ≥0} {M : ℝ}

/-- **Below a level, a gradient flow almost surely converges to a critical
point that is not a strict saddle**, if `f` satisfies the Łojasiewicz
inequality with an exponent `0 ≤ a < 1` at each critical point below `c`,
and almost every trajectory starting below `c` has a limit point up to
translations in a set `Λ` leaving `f` invariant.  The limit is no higher than
the start.  This is `ae_exists_tendsto_gradientFlow_not_isStrictSaddle` with
`hloj` and `hcl` asked only below `c`. -/
theorem ae_exists_tendsto_gradientFlow_not_isStrictSaddle_of_lt (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f)
    (hL : LipschitzWith L (gradient f)) (hM : ∀ x, ‖gradient f x‖ ≤ M) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, f x < c → ∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ,
      ‖gradientFlow f t x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) ∧
      gradient f w = 0 ∧ f w ≤ f x ∧ ¬IsStrictSaddle f w := by
  have hdf : Differentiable ℝ f := hf.differentiable (by norm_num)
  filter_upwards [hcl, ae_not_tendsto_gradientFlow_of_isStrictSaddle μ hν hf hL hM]
    with x hcl hx hxc
  obtain ⟨z, hz⟩ := hcl hxc
  have hd : ∀ t, 0 ≤ t → HasDerivAt (fun t => gradientFlow f t x)
      (-gradient f (gradientFlow f t x)) t := fun t _ => hasDerivAt_gradientFlow hL hM t x
  have hb : ∀ᶠ t in atTop, f (gradientFlow f t x) ≤ f x := by
    filter_upwards [eventually_ge_atTop 0] with t ht
    have h : f (gradientFlow f t x) ≤ f (gradientFlow f 0 x) :=
      antitoneOn_comp_gradientFlow hdf hd self_mem_Ici ht ht
    have h0 : gradientFlow f 0 x = x := (isFlowField_neg_gradient hL hM).flow_self 0 x
    rwa [h0] at h
  obtain ⟨a, -, ha1, hza⟩ : ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
    by_cases h0 : gradient f z = 0
    · exact hloj z h0
        ((le_of_frequently_norm_add_sub_lt hΛ hdf.continuous.continuousAt hz hb).trans_lt hxc)
    · exact ⟨0, le_rfl, one_pos,
        isLojasiewiczAt_of_gradient_ne_zero hL.continuous.continuousAt h0 le_rfl⟩
  obtain ⟨w, hw⟩ := exists_tendsto_gradientFlow_of_isLojasiewiczAt hdf hd hΛ ha1 hza hz
  exact ⟨w, hw, gradient_eq_zero_of_tendsto_gradientFlow hd hL.continuous.continuousAt hw,
    le_of_tendsto ((hdf.continuous.tendsto w).comp hw) hb, fun hs => hx w hs hw⟩

/-- **Below a level, a gradient flow almost surely converges to a global
minimizer**, if every critical point below `c` is a global minimizer or a
strict saddle and satisfies the Łojasiewicz inequality with an exponent
`0 ≤ a < 1`, and almost every trajectory starting below `c` has a limit
point up to translations in a set `Λ` leaving `f` invariant.  This is
`ae_exists_tendsto_gradientFlow_isMinOn` for the starts below `c`, with
`hcrit`, `hloj` and `hcl` asked only below `c`. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn_of_lt (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, f x < c → ∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ,
      ‖gradientFlow f t x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) := by
  filter_upwards [ae_exists_tendsto_gradientFlow_not_isStrictSaddle_of_lt μ hν hf hL hM hΛ hloj
    hcl] with x hx hxc
  obtain ⟨w, hw, hw0, hwx, hws⟩ := hx hxc
  exact ⟨w, (hcrit w hw0 (hwx.trans_lt hxc)).resolve_right hws, hw⟩

/-- **Below a level, the gradient flow of a periodic function almost surely
converges to a global minimizer**, if every critical point below `c` is a
global minimizer or a strict saddle and satisfies the Łojasiewicz
inequality.  The translations `Λ` leaving `f` invariant span `E`, so the
gradient is bounded and Lipschitz, and every trajectory has a limit point
up to their integer combinations. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_span_eq_top (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) {Λ : Set E}
    (hspan : Submodule.span ℝ Λ = ⊤) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) := by
  obtain ⟨L, hM, -, hL⟩ := exists_lipschitzWith_gradient_of_periodic hf hspan hΛ
  exact ae_exists_tendsto_gradientFlow_isMinOn_of_lt μ hν hf hL hM
    (fun v hv => periodic_of_mem_span hΛ hv) hcrit hloj (Eventually.of_forall fun x _ =>
      exists_frequently_norm_add_sub_lt_of_span_eq_top hspan _)

/-- The hypotheses of the theorems above hold for `f(x) = 2 - cos x - cos² x`,
the function of the example in `LevelLandscape`, on the line with Lebesgue
measure, `L = 3`, `M = 2`, the period `2π` or its integer multiples, and
`c = 2`.  The gradient is `sin x + sin 2x`.  Below `2` the critical points
are the global minimizers, where `cos x = 1` and the Hessian
`cos x + 2 cos 2x` is `3`. -/
example : let f := fun x : ℝ => 2 - Real.cos x - Real.cos x ^ 2
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧ ContDiff ℝ 2 f ∧
      LipschitzWith 3 (gradient f) ∧ (∀ x, ‖gradient f x‖ ≤ 2) ∧
      Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → f z < 2 → IsMinOn f univ z ∨ IsStrictSaddle f z) ∧
      (∀ z, gradient f z = 0 → f z < 2 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) ∧
      ∀ᵐ x ∂(volume : Measure ℝ), f x < 2 → ∃ z, ∀ ε > 0, ∃ᶠ t in atTop,
        ∃ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ‖gradientFlow f t x + v - z‖ < ε := by
  intro f
  have hf : ContDiff ℝ 2 f := ((analyticOnNhd_const.fun_sub Real.analyticOnNhd_cos).fun_sub
    (Real.analyticOnNhd_cos.fun_pow 2)).contDiff
  have hg : gradient f = fun x => Real.sin x + Real.sin (2 * x) := funext fun x => by
    rw [gradient_eq_deriv']
    exact (((Real.hasDerivAt_cos x).const_sub 2).fun_sub
      ((Real.hasDerivAt_cos x).fun_pow 2)).deriv.trans (by rw [Real.sin_two_mul]; norm_num; ring)
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
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl, hf, ?_, fun x => ?_, hspan, hΛ,
    fun v hv => periodic_of_mem_span hΛ hv, fun z hz hc => Or.inl ?_,
    fun z hz hc => ⟨1 / 2, by norm_num, by norm_num, ?_⟩,
    Eventually.of_forall fun x _ => exists_frequently_norm_add_sub_lt_of_span_eq_top hspan _⟩
  · rw [hg]
    refine (Real.lipschitzWith_sin.add (Real.lipschitzWith_sin.comp
      (lipschitzWith_smul (2 : ℝ)))).weaken ?_
    rw [← NNReal.coe_le_coe]
    norm_num
  · simp only [hg, Real.norm_eq_abs]
    linarith [abs_add_le (Real.sin x) (Real.sin (2 * x)), Real.abs_sin_le_one x,
      Real.abs_sin_le_one (2 * x)]
  · refine isMinOn_univ_iff.2 fun y => ?_
    simp only [f, hkey z hz hc]
    nlinarith [Real.cos_le_one y, Real.neg_one_le_cos y]
  · refine isLojasiewiczAt_of_injective (hf.differentiable (by norm_num)) ?_ hz
      (A := ContinuousLinearMap.toSpanSingleton ℝ (Real.cos z + 2 * Real.cos (2 * z)))
      fun a b hab => ?_
    · rw [hg]
      exact (((Real.hasDerivAt_sin z).fun_add ((hasDerivAt_id' z).const_mul 2).sin).congr_deriv
        (by ring)).hasFDerivAt
    · rw [Real.cos_two_mul, hkey z hz hc] at hab
      simp only [ContinuousLinearMap.toSpanSingleton_apply, smul_eq_mul] at hab
      linarith

end Transformer.SaddleAvoidance
