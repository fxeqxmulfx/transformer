/-
# Saddle avoidance — the scheme for analytic energies

The scheme theorems ask `hf : ContDiff ℝ 2 f` and the Łojasiewicz inequality
at every critical point (`hloj`).  An analytic `f` is `C²`
(`AnalyticOnNhd.contDiff`) and satisfies the inequality at every point
(`exists_isLojasiewiczAt_of_analyticOnNhd`, Łojasiewicz's theorem), so
`hf : AnalyticOnNhd ℝ f univ` replaces both.  For gradient descent:
`ae_exists_tendsto_isMinOn_of_analyticOnNhd`,
`ae_exists_tendsto_isMinOn_of_span_eq_top_of_analyticOnNhd` and
`ae_exists_tendsto_isLocalMin_of_analyticOnNhd`; for gradient flows:
`ae_exists_tendsto_gradientFlow_isMinOn_of_analyticOnNhd` and
`ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top_of_analyticOnNhd`.
A model then brings its analyticity, a Lipschitz constant, and its
landscape `hcrit`.
-/

import Transformer.SaddleAvoidance.Analytic

open Filter Set MeasureTheory
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L : ℝ≥0} {η M : ℝ}

/-- **Gradient descent on an analytic function almost surely converges to a
global minimizer** if every critical point is a global minimizer or a strict
saddle and almost every orbit has a limit point up to translations in a set `Λ`
leaving `f` invariant: `ae_exists_tendsto_isMinOn` without `hloj`. -/
theorem ae_exists_tendsto_isMinOn_of_analyticOnNhd (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hcl : ∀ᵐ x ∂ν, ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ,
      ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f univ w ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) :=
  ae_exists_tendsto_isMinOn μ hν hf.contDiff hL hη hηL hΛ hcrit
    (fun z _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z) hcl

/-- **Gradient descent on an analytic function invariant under translations
spanning `E` almost surely converges to a global minimizer** if every critical
point is a global minimizer or a strict saddle:
`ae_exists_tendsto_isMinOn_of_span_eq_top` without `hloj`. -/
theorem ae_exists_tendsto_isMinOn_of_span_eq_top_of_analyticOnNhd (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hspan : Submodule.span ℝ Λ = ⊤) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f univ w ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) :=
  ae_exists_tendsto_isMinOn_of_span_eq_top μ hν hf.contDiff hL hη hηL hspan hΛ hcrit
    (fun z _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z)

/-- **Gradient descent on an analytic function almost surely converges to a
local minimizer** if every critical point is a local minimizer or a strict
saddle and almost every orbit is bounded: `ae_exists_tendsto_isLocalMin`, the
first part of `prop:rates` of arXiv:1602.04915, without `hloj`. -/
theorem ae_exists_tendsto_isLocalMin_of_analyticOnNhd (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1)
    (hcrit : ∀ z, gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z)
    (hbdd : ∀ᵐ x ∂ν, Bornology.IsBounded (range fun k => (gradientStep f η)^[k] x)) :
    ∀ᵐ x ∂ν, ∃ w, IsLocalMin f w ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) :=
  ae_exists_tendsto_isLocalMin μ hν hf.contDiff hL hη hηL hcrit
    (fun z _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z) hbdd

/-- **The gradient flow of an analytic function almost surely converges to a
global minimizer** if `∇f` is bounded and Lipschitz, every critical point is a
global minimizer or a strict saddle, and almost every trajectory has a limit
point up to translations in a set `Λ` leaving `f` invariant:
`ae_exists_tendsto_gradientFlow_isMinOn` without `hloj`. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn_of_analyticOnNhd (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hM : ∀ x, ‖gradient f x‖ ≤ M) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hcl : ∀ᵐ x ∂ν, ∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ, ‖gradientFlow f t x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f univ w ∧ Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) :=
  ae_exists_tendsto_gradientFlow_isMinOn μ hν hf.contDiff hL hM hΛ hcrit
    (fun z _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z) hcl

/-- **The gradient flow of an analytic function invariant under translations
spanning `E` almost surely converges to a global minimizer** if every critical
point is a global minimizer or a strict saddle:
`ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top` without `hloj`. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top_of_analyticOnNhd
    (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ)
    (hf : AnalyticOnNhd ℝ f univ) {Λ : Set E} (hspan : Submodule.span ℝ Λ = ⊤)
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f univ w ∧ Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) :=
  ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top μ hν hf.contDiff hspan hΛ hcrit
    (fun z _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z)

/-- The hypotheses of the theorems above on global minimizers hold for
`f(x) = 1 - cos x` on the line with Lebesgue measure, `L = M = 1`, `η = 1/2`,
and the period `2π` or its integer multiples.  The even multiples of `π` are
global minimizers, the odd ones strict saddles, where the Hessian `cos x = -1`
is negative. -/
example : let f := fun x : ℝ => 1 - Real.cos x
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧
      AnalyticOnNhd ℝ f univ ∧ LipschitzWith 1 (gradient f) ∧ (∀ x, ‖gradient f x‖ ≤ 1) ∧
      (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧ Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z) ∧
      (∀ᵐ x ∂(volume : Measure ℝ), ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈
        (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ‖(gradientStep f (1 / 2))^[k] x + v - z‖ < ε) ∧
      ∀ᵐ x ∂(volume : Measure ℝ), ∃ z, ∀ ε > 0, ∃ᶠ t in atTop,
        ∃ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ‖gradientFlow f t x + v - z‖ < ε := by
  intro f
  have hg : gradient f = Real.sin := funext fun x => by
    rw [gradient_eq_deriv', ((Real.hasDerivAt_cos x).const_sub 1).deriv, neg_neg]
  have hH : ∀ z, fderiv ℝ (gradient f) z = ContinuousLinearMap.toSpanSingleton ℝ (Real.cos z) :=
    fun z => by rw [hg, (Real.hasDerivAt_sin z).hasFDerivAt.fderiv]
  have hspan : Submodule.span ℝ {2 * Real.pi} = ⊤ :=
    Ideal.span_singleton_eq_top.2 (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne')
  have hΛ : ∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y := fun v hv y => by
    simp [mem_singleton_iff.1 hv, f]
  obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl,
    analyticOnNhd_const.sub Real.analyticOnNhd_cos, by rw [hg]; exact Real.lipschitzWith_sin,
    fun x => by rw [hg, Real.norm_eq_abs]; exact Real.abs_sin_le_one x, by norm_num, by norm_num,
    hspan, hΛ, fun v hv => periodic_of_mem_span hΛ hv, fun z hz => ?_,
    Eventually.of_forall fun x => ?_,
    Eventually.of_forall fun x => exists_frequently_norm_add_sub_lt_of_span_eq_top hspan _⟩
  · rw [hg] at hz
    rcases Real.sin_eq_zero_iff_cos_eq.1 hz with h | h
    · refine Or.inl (isMinOn_univ_iff.2 fun y => ?_)
      simp only [f, h, sub_self, sub_nonneg]
      exact Real.cos_le_one y
    · refine Or.inr ⟨by rw [hg]; exact hz, -1, by norm_num,
        Module.End.hasEigenvalue_of_hasEigenvector ⟨Module.End.mem_eigenspace_iff.2 ?_,
          one_ne_zero⟩⟩
      simp [hH, h]
  · obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : ℝ) R)
      (Λ := (Submodule.span ℤ {2 * Real.pi} : Set ℝ))
      (u := fun k => (gradientStep f (1 / 2))^[k] x) fun k => hR _
    exact ⟨z, hz⟩

/-- The hypotheses of `ae_exists_tendsto_isLocalMin_of_analyticOnNhd` hold for
`f(x) = ‖x‖² / 2` on the line with Lebesgue measure, `L = 1` and `η = 1/2`: the
only critical point `0` is a minimizer, and gradient descent halves every
start, so every orbit is bounded. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧
      AnalyticOnNhd ℝ f univ ∧ LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧
      (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧
      (∀ z, gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z) ∧
      ∀ᵐ x ∂(volume : Measure ℝ),
        Bornology.IsBounded (range fun k => (gradientStep f (1 / 2))^[k] x) := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have horb : ∀ k x, (gradientStep f (1 / 2))^[k] x = (1 / 2 : ℝ) ^ k * x := fun k x => by
    simp only [f, iterate_gradientStep_half_mul_norm_sq, smul_eq_mul]
    norm_num
  have hf : f = fun x => 1 / 2 * x ^ 2 := funext fun x => by simp [f]
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl,
    hf ▸ analyticOnNhd_const.mul (analyticOnNhd_id.fun_pow 2),
    by rw [hg]; exact LipschitzWith.id, by norm_num, by norm_num, fun z hz => Or.inl ?_,
    Eventually.of_forall fun x => ?_⟩
  · rw [hg, id_eq] at hz
    subst hz
    exact Eventually.of_forall fun y => by
      simp only [f, norm_zero]
      nlinarith [sq_nonneg ‖y‖]
  · rw [isBounded_iff_forall_norm_le]
    refine ⟨‖x‖, ?_⟩
    rintro _ ⟨k, rfl⟩
    show ‖(gradientStep f (1 / 2))^[k] x‖ ≤ ‖x‖
    rw [horb, norm_mul, norm_pow, Real.norm_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)]
    exact mul_le_of_le_one_left (norm_nonneg x) (pow_le_one₀ (by norm_num) (by norm_num))

end Transformer.SaddleAvoidance
