/-
# Saddle avoidance — analytic functions below a level

An analytic function on a finite-dimensional real inner product space is
`C²` (`AnalyticOnNhd.contDiff`) and satisfies the Łojasiewicz inequality at
every point (`exists_isLojasiewiczAt_of_analyticOnNhd`), so for it the
theorems of `LevelLandscape` and `FlowLevelLandscape` need no hypothesis
`hloj`.  If every critical point below a level `c` is a global minimizer or
a strict saddle, gradient descent
(`ae_exists_tendsto_isMinOn_of_lt_of_analyticOnNhd`) and the gradient flow
(`ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_analyticOnNhd`) almost
surely converge to a global minimizer from the starts below `c` that have a
limit point up to translations leaving `f` invariant; when the translations
span `E`, from all starts below `c` (the forms with `_of_span_eq_top`).

Source: item M6.3 of `landscape_plan.md`.
-/

import Transformer.Landscape.SaddleAvoidance.FlowLevelLandscape
import Transformer.Landscape.SaddleAvoidance.LocalInequality

open Filter Set MeasureTheory
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L : ℝ≥0} {η M : ℝ}

/-- **`ae_exists_tendsto_isMinOn_of_lt` for analytic `f`**, without `hloj`:
below a level, gradient descent almost surely converges to a global
minimizer. -/
theorem ae_exists_tendsto_isMinOn_of_lt_of_analyticOnNhd (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hcl : ∀ᵐ x ∂ν, f x < c → ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ,
      ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) :=
  ae_exists_tendsto_isMinOn_of_lt μ hν hf.contDiff hL hη hηL hΛ hcrit
    (fun z _ _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z) hcl

/-- **`ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top` for analytic `f`**,
without `hloj`: below a level, gradient descent on a periodic analytic
function almost surely converges to a global minimizer. -/
theorem ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top_of_analyticOnNhd (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E}
    (hspan : Submodule.span ℝ Λ = ⊤) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) :=
  ae_exists_tendsto_isMinOn_of_lt_of_span_eq_top μ hν hf.contDiff hL hη hηL hspan hΛ hcrit
    fun z _ _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z

/-- **`ae_exists_tendsto_gradientFlow_isMinOn_of_lt` for analytic `f`**,
without `hloj`: below a level, a gradient flow almost surely converges to a
global minimizer. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_analyticOnNhd (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : AnalyticOnNhd ℝ f univ)
    (hL : LipschitzWith L (gradient f)) (hM : ∀ x, ‖gradient f x‖ ≤ M) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hcl : ∀ᵐ x ∂ν, f x < c → ∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ,
      ‖gradientFlow f t x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) :=
  ae_exists_tendsto_gradientFlow_isMinOn_of_lt μ hν hf.contDiff hL hM hΛ hcrit
    (fun z _ _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z) hcl

/-- **`ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_span_eq_top` for
analytic `f`**, without `hloj`: below a level, the gradient flow of a
periodic analytic function almost surely converges to a global minimizer. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_span_eq_top_of_analyticOnNhd
    (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ)
    (hf : AnalyticOnNhd ℝ f univ) {Λ : Set E} (hspan : Submodule.span ℝ Λ = ⊤)
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z) :
    ∀ᵐ x ∂ν, f x < c → ∃ w, IsMinOn f univ w ∧
      Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) :=
  ae_exists_tendsto_gradientFlow_isMinOn_of_lt_of_span_eq_top μ hν hf.contDiff hspan hΛ hcrit
    fun z _ _ => exists_isLojasiewiczAt_of_analyticOnNhd hf z

/-- The hypotheses of the theorems above hold for the analytic function
`f(x) = 2 - cos x - cos² x` of the example in `LevelLandscape`, on the line
with Lebesgue measure, `L = 3`, `η = 1/4`, `M = 2`, the period `2π` or its
integer multiples, and `c = 2`.  Below `2` the critical points are the
global minimizers, where `cos x = 1`; the local minimizer `π` at the level
`2` is not a global one, so the hypothesis `hcrit` of the global theorems
fails, as that example shows. -/
example : let f := fun x : ℝ => 2 - Real.cos x - Real.cos x ^ 2
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧
      AnalyticOnNhd ℝ f univ ∧ LipschitzWith 3 (gradient f) ∧ (0 : ℝ) < 1 / 4 ∧
      (1 / 4 : ℝ) * ((3 : ℝ≥0) : ℝ) < 1 ∧ (∀ x, ‖gradient f x‖ ≤ 2) ∧
      Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → f z < 2 → IsMinOn f univ z ∨ IsStrictSaddle f z) ∧
      (∀ᵐ x ∂(volume : Measure ℝ), f x < 2 → ∃ z, ∀ ε > 0, ∃ᶠ k in atTop,
        ∃ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ),
          ‖(gradientStep f (1 / 4))^[k] x + v - z‖ < ε) ∧
      ∀ᵐ x ∂(volume : Measure ℝ), f x < 2 → ∃ z, ∀ ε > 0, ∃ᶠ t in atTop,
        ∃ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ‖gradientFlow f t x + v - z‖ < ε := by
  intro f
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
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl,
    (analyticOnNhd_const.fun_sub Real.analyticOnNhd_cos).fun_sub
      (Real.analyticOnNhd_cos.fun_pow 2), ?_, by norm_num, by norm_num, fun x => ?_, hspan, hΛ,
    fun v hv => periodic_of_mem_span hΛ hv, fun z hz hc => Or.inl ?_,
    Eventually.of_forall fun x _ => ?_,
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
  · obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
    obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : ℝ) R)
      (Λ := (Submodule.span ℤ {2 * Real.pi} : Set ℝ))
      (u := fun k => (gradientStep f (1 / 4))^[k] x) fun k => hR _
    exact ⟨z, hz⟩

end Transformer.Landscape.SaddleAvoidance
