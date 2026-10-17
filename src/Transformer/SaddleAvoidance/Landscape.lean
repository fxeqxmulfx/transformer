/-
# Saddle avoidance — gradient descent finds a global minimizer

Let `f` be `C²` on a finite-dimensional real inner product space with an
`L`-Lipschitz gradient, and `0 < η`, `η L < 1`.  Suppose every critical
point of `f` is a global minimizer or a strict saddle, and `f` satisfies the
Łojasiewicz inequality with an exponent `0 ≤ a < 1` at each critical point.
Suppose also that almost every orbit of gradient descent has a limit point
up to translations leaving `f` invariant.  Then almost every orbit converges
to a global minimizer (`ae_exists_tendsto_isMinOn`).  The orbit converges to
a critical point that is almost surely not a strict saddle
(`ae_exists_tendsto_not_isStrictSaddle`), and the landscape hypothesis
leaves only the global minimizers.

When the translations span `E`, as the periods `2πℤⁿ` of a function of `n`
angles do, every point has a translate in one closed ball by an integer
combination of them (`exists_add_mem_closedBall_of_span_eq_top`).  These
combinations leave `f` invariant too (`periodic_of_mem_span`), so every orbit
has a limit point up to translations (`exists_frequently_norm_add_sub_lt`),
and that hypothesis can be dropped
(`ae_exists_tendsto_isMinOn_of_span_eq_top`).

The sources prove convergence to a local minimizer: `thm:local-min` of
arXiv:1602.04915, formalized as `ae_exists_tendsto_isLocalMin`.  The global
statement is the scheme that combines a landscape theorem, which gives
`hcrit`, with the Łojasiewicz inequality, which gives `hloj`.
-/

import Transformer.SaddleAvoidance.Minimizers
import Transformer.SaddleAvoidance.Periodic
import Transformer.SaddleAvoidance.CriticalPoints
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

open Filter Metric MeasureTheory
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L : ℝ≥0} {η : ℝ}

/-- **Gradient descent almost surely converges to a global minimizer** if
every critical point is a global minimizer or a strict saddle, `f`
satisfies the Łojasiewicz inequality with an exponent `0 ≤ a < 1` at each
critical point, and almost every orbit has a limit point up to translations
in a set `Λ` leaving `f` invariant.  `thm:local-min` of arXiv:1602.04915
reads: "Assume the same conditions as Theorem `thm:no-saddles` and
`lim_k x_k` exists, then `Pr(lim_k x_k = x*) = 1`, where `x*` is a local
minimizer."  As in `ae_exists_tendsto_not_isStrictSaddle`, `hloj` and `hcl`
replace the existence of the limit; the landscape hypothesis `hcrit`, with
global minimizers in place of local ones, makes the limit a global
minimizer. -/
theorem ae_exists_tendsto_isMinOn (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E}
    (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f)) (hη : 0 < η)
    (hηL : η * L < 1) {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f Set.univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ,
      ‖(gradientStep f η)^[k] x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f Set.univ w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  filter_upwards [ae_exists_tendsto_not_isStrictSaddle μ hν hf hL hη hηL hΛ hloj hcl]
    with x ⟨w, hw, hw0, hws⟩
  exact ⟨w, (hcrit w hw0).resolve_right hws, hw⟩

/-- **Gradient descent on a periodic function almost surely converges to a
global minimizer** if every critical point is a global minimizer or a strict
saddle and the Łojasiewicz inequality holds at each.  The translations `Λ`
leaving `f` invariant span `E`, so every orbit has a limit point up to their
integer combinations. -/
theorem ae_exists_tendsto_isMinOn_of_span_eq_top (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E} (hspan : Submodule.span ℝ Λ = ⊤)
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f Set.univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f Set.univ w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
  refine ae_exists_tendsto_isMinOn μ hν hf hL hη hηL (Λ := (Submodule.span ℤ Λ : Set E))
    (fun v hv => periodic_of_mem_span hΛ hv) hcrit hloj (Eventually.of_forall fun x => ?_)
  obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : E) R)
    (Λ := (Submodule.span ℤ Λ : Set E)) (u := fun k => (gradientStep f η)^[k] x)
    fun k => hR _
  exact ⟨z, hz⟩

/-- The hypotheses of `ae_exists_tendsto_isMinOn` hold for `f(x) = ‖x‖² / 2`
on the line with Lebesgue measure, `L = 1`, `η = 1/2` and `Λ = {0}`: the
only critical point `0` is the global minimizer, where the Łojasiewicz
inequality holds with `a = 1/2`, and gradient descent halves every start. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧ ContDiff ℝ 2 f ∧
      LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧
      (∀ v ∈ ({0} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → IsMinOn f Set.univ z ∨ IsStrictSaddle f z) ∧
      (∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) ∧
      ∀ᵐ x ∂(volume : Measure ℝ), ∃ z, ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ ({0} : Set ℝ),
        ‖(gradientStep f (1 / 2))^[k] x + v - z‖ < ε := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have horb : ∀ k x, (gradientStep f (1 / 2))^[k] x = (1 / 2 : ℝ) ^ k * x := fun k x => by
    simp only [f, iterate_gradientStep_half_mul_norm_sq, smul_eq_mul]
    norm_num
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl,
    contDiff_const.mul (contDiff_norm_sq ℝ), ?_, by norm_num, by norm_num, by simp,
    fun z hz => Or.inl ?_, fun z hz => ⟨1 / 2, by norm_num, by norm_num, ?_⟩,
    Eventually.of_forall fun x => ⟨0, fun ε hε => ?_⟩⟩
  · rw [hg]
    exact LipschitzWith.id
  · rw [hg, id_eq] at hz
    subst hz
    refine isMinOn_univ_iff.2 fun y => ?_
    simp only [f, norm_zero]
    nlinarith [sq_nonneg ‖y‖]
  · rw [hg, id_eq] at hz
    subst hz
    exact isLojasiewiczAt_half_mul_norm_sq
  · have hlim : Tendsto (fun k => (1 / 2 : ℝ) ^ k * x) atTop (𝓝 0) := by
      simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num : (0 : ℝ) ≤ 1 / 2)
        (by norm_num)).mul_const x
    exact (Metric.tendsto_nhds.1 hlim ε hε).frequently.mono fun k hk =>
      ⟨0, Set.mem_singleton 0, by rw [horb, add_zero, ← dist_eq_norm]; exact hk⟩

/-- The hypotheses of `ae_exists_tendsto_isMinOn_of_span_eq_top` hold for
`f(x) = 1 - cos x` on the line with Lebesgue measure, `L = 1`, `η = 1/2` and
the period `2π`, which spans `ℝ`.  The critical points are the multiples of
`π`: the even ones are global minimizers, the odd ones strict saddles, where
the Hessian `cos x = -1` is negative, and the Hessian `±1` is injective at
both, so the Łojasiewicz inequality holds with `a = 1/2`. -/
example : let f := fun x : ℝ => 1 - Real.cos x
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧ ContDiff ℝ 2 f ∧
      LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧
      Submodule.span ℝ {2 * Real.pi} = ⊤ ∧
      (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → IsMinOn f Set.univ z ∨ IsStrictSaddle f z) ∧
      ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
  intro f
  have hg : gradient f = Real.sin := funext fun x => by
    rw [gradient_eq_deriv', ((Real.hasDerivAt_cos x).const_sub 1).deriv, neg_neg]
  have hf : ContDiff ℝ 2 f := contDiff_const.sub Real.contDiff_cos
  have hH : ∀ z, fderiv ℝ (gradient f) z = ContinuousLinearMap.toSpanSingleton ℝ (Real.cos z) :=
    fun z => by rw [hg, (Real.hasDerivAt_sin z).hasFDerivAt.fderiv]
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl, hf,
    by rw [hg]; exact Real.lipschitzWith_sin, by norm_num, by norm_num,
    Ideal.span_singleton_eq_top.2 (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne'),
    fun v hv y => ?_, fun z hz => ?_, fun z hz => ⟨1 / 2, by norm_num, by norm_num, ?_⟩⟩
  · rw [Set.mem_singleton_iff.1 hv]
    simp [f]
  · rw [hg] at hz
    rcases Real.sin_eq_zero_iff_cos_eq.1 hz with h | h
    · refine Or.inl (isMinOn_univ_iff.2 fun y => ?_)
      simp only [f, h, sub_self, sub_nonneg]
      exact Real.cos_le_one y
    · refine Or.inr ⟨by rw [hg]; exact hz, -1, by norm_num,
        Module.End.hasEigenvalue_of_hasEigenvector ⟨Module.End.mem_eigenspace_iff.2 ?_,
          one_ne_zero⟩⟩
      simp [hH, h]
  · have hc : Real.cos z ≠ 0 := by
      rw [hg] at hz
      rcases Real.sin_eq_zero_iff_cos_eq.1 hz with h | h <;> rw [h] <;> norm_num
    refine isLojasiewiczAt_of_injective (hf.differentiable (by norm_num))
      (A := ContinuousLinearMap.toSpanSingleton ℝ (Real.cos z)) ?_ hz fun a b hab => ?_
    · rw [hg]
      exact (Real.hasDerivAt_sin z).hasFDerivAt
    · simp only [ContinuousLinearMap.toSpanSingleton_apply, smul_eq_mul] at hab
      exact mul_right_cancel₀ hc hab

end Transformer.SaddleAvoidance
