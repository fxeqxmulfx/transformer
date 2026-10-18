/-
# Saddle avoidance — gradient flows find a global minimizer

Let `f` be `C²` on a finite-dimensional real inner product space with a
bounded `L`-Lipschitz gradient.  If every critical point of `f` is a global
minimizer or a strict saddle, `f` satisfies the Łojasiewicz inequality with
an exponent `0 ≤ a < 1` at each, and almost every trajectory of the gradient
flow has a limit point up to translations leaving `f` invariant, then almost
every trajectory converges to a global minimizer
(`ae_exists_tendsto_gradientFlow_isMinOn`), as for gradient descent
(`ae_exists_tendsto_isMinOn`).  The trajectory converges
(`exists_tendsto_gradientFlow_of_isLojasiewiczAt`) to a critical point
(`gradient_eq_zero_of_tendsto_gradientFlow`), almost surely not a strict
saddle (`ae_not_tendsto_gradientFlow_of_isStrictSaddle`).  When the
translations span `E`, every curve has a limit point up to their integer
combinations (`exists_frequently_norm_add_sub_lt_of_span_eq_top`) and the
gradient is bounded and Lipschitz, so these hypotheses can be dropped
(`ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top`).

This is the scheme of the proof of Theorem `p:beta0` in Appendix A of
arXiv:2312.10794v5, for descent in a Euclidean space: Łojasiewicz's theorem
makes every trajectory converge, `l:nosaddleconv` excludes the strict
saddles, and a landscape lemma, there `lem: yury.lemma`, leaves the optima.
-/

import Transformer.SaddleAvoidance.FlowAvoidance
import Transformer.SaddleAvoidance.FlowConvergence
import Transformer.SaddleAvoidance.Landscape

open Filter Set MeasureTheory
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {M : ℝ}

/-- **The limit of a gradient flow is a critical point**: if `x` solves
`ẋ = -∇f(x)` from time `0` on and converges to `w`, where `∇f` is continuous,
then `∇f(w) = 0`.  Over `[t, t + 1]` the trajectory moves by nearly `-∇f(w)`
for large `t`, and by nearly `0`. -/
theorem gradient_eq_zero_of_tendsto_gradientFlow {x : ℝ → E}
    (hx : ∀ t, 0 ≤ t → HasDerivAt x (-gradient f (x t)) t) {w : E}
    (hc : ContinuousAt (gradient f) w) (hw : Tendsto x atTop (𝓝 w)) : gradient f w = 0 := by
  refine norm_le_zero_iff.1 (le_of_forall_pos_le_add fun ε hε => ?_)
  rw [zero_add]
  obtain ⟨T, hT⟩ := eventually_atTop.1 ((Metric.tendsto_nhds.1 (hc.tendsto.comp hw) ε hε).and
    (eventually_ge_atTop 0))
  have hstep : ∀ t, T ≤ t → ‖x (t + 1) - x t + gradient f w‖ ≤ ε := fun t ht => by
    have hd : ∀ s ∈ Icc t (t + 1), HasDerivWithinAt (fun s => x s + s • gradient f w)
        (-gradient f (x s) + (1 : ℝ) • gradient f w) (Icc t (t + 1)) s := fun s hs =>
      ((hx s (hT s (ht.trans hs.1)).2).add ((hasDerivAt_id' s).smul_const _)).hasDerivWithinAt
    have hb : ∀ s ∈ Icc t (t + 1), ‖-gradient f (x s) + (1 : ℝ) • gradient f w‖ ≤ ε :=
      fun s hs => by
        rw [one_smul, neg_add_eq_sub, ← dist_eq_norm, dist_comm]
        exact (hT s (ht.trans hs.1)).1.le
    have h := (convex_Icc t (t + 1)).norm_image_sub_le_of_norm_hasDerivWithin_le hd hb
      (left_mem_Icc.2 (by linarith)) (right_mem_Icc.2 (by linarith))
    rwa [add_smul, one_smul, add_sub_cancel_left, Real.norm_eq_abs, abs_one, mul_one,
      show x (t + 1) + (t • gradient f w + gradient f w) - (x t + t • gradient f w) =
        x (t + 1) - x t + gradient f w by abel] at h
  refine le_of_tendsto ?_ (eventually_atTop.2 ⟨T, hstep⟩)
  simpa using (((hw.comp (tendsto_atTop_add_const_right _ 1 tendsto_id)).sub hw).add_const
    (gradient f w)).norm

variable [FiniteDimensional ℝ E]

omit [CompleteSpace E] in
/-- **Up to translations spanning `E`, every curve has a limit point**: its
values at integer times have one in a closed ball of representatives. -/
theorem exists_frequently_norm_add_sub_lt_of_span_eq_top {Λ : Set E}
    (hspan : Submodule.span ℝ Λ = ⊤) (u : ℝ → E) :
    ∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ (Submodule.span ℤ Λ : Set E), ‖u t + v - z‖ < ε := by
  obtain ⟨R, hR⟩ := exists_add_mem_closedBall_of_span_eq_top hspan
  obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt (isCompact_closedBall (0 : E) R)
    (Λ := (Submodule.span ℤ Λ : Set E)) (u := fun k : ℕ => u k) fun k => hR _
  exact ⟨z, fun ε hε => tendsto_natCast_atTop_atTop.frequently (hz ε hε)⟩

variable [MeasurableSpace E] [BorelSpace E]

/-- **A gradient flow almost surely converges to a critical point that is not
a strict saddle**, if `f` satisfies the Łojasiewicz inequality with an
exponent `0 ≤ a < 1` at each critical point and almost every trajectory has a
limit point up to translations in a set `Λ` leaving `f` invariant. -/
theorem ae_exists_tendsto_gradientFlow_not_isStrictSaddle (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f)
    (hL : LipschitzWith L (gradient f)) (hM : ∀ x, ‖gradient f x‖ ≤ M) {Λ : Set E}
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hloj : ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, ∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ, ‖gradientFlow f t x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, ∃ w, Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) ∧
      gradient f w = 0 ∧ ¬IsStrictSaddle f w := by
  filter_upwards [hcl, ae_not_tendsto_gradientFlow_of_isStrictSaddle μ hν hf hL hM]
    with x ⟨z, hz⟩ hx
  have hd : ∀ t, 0 ≤ t → HasDerivAt (fun t => gradientFlow f t x)
      (-gradient f (gradientFlow f t x)) t := fun t _ => hasDerivAt_gradientFlow hL hM t x
  obtain ⟨a, -, ha1, hza⟩ : ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
    by_cases h0 : gradient f z = 0
    · exact hloj z h0
    · exact ⟨0, le_rfl, one_pos,
        isLojasiewiczAt_of_gradient_ne_zero hL.continuous.continuousAt h0 le_rfl⟩
  obtain ⟨w, hw⟩ := exists_tendsto_gradientFlow_of_isLojasiewiczAt
    (hf.differentiable (by norm_num)) hd hΛ ha1 hza hz
  exact ⟨w, hw, gradient_eq_zero_of_tendsto_gradientFlow hd hL.continuous.continuousAt hw,
    fun hs => hx w hs hw⟩

/-- **A gradient flow almost surely converges to a global minimizer** if
every critical point is a global minimizer or a strict saddle, `f` satisfies
the Łojasiewicz inequality with an exponent `0 ≤ a < 1` at each critical
point, and almost every trajectory has a limit point up to translations in a
set `Λ` leaving `f` invariant. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) {Λ : Set E} (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a)
    (hcl : ∀ᵐ x ∂ν, ∃ z, ∀ ε > 0, ∃ᶠ t in atTop, ∃ v ∈ Λ, ‖gradientFlow f t x + v - z‖ < ε) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f univ w ∧ Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) := by
  filter_upwards [ae_exists_tendsto_gradientFlow_not_isStrictSaddle μ hν hf hL hM hΛ hloj hcl]
    with x ⟨w, hw, hw0, hws⟩
  exact ⟨w, (hcrit w hw0).resolve_right hws, hw⟩

/-- **The gradient flow of a periodic function almost surely converges to a
global minimizer** if every critical point is a global minimizer or a strict
saddle and the Łojasiewicz inequality holds at each.  The translations `Λ`
leaving `f` invariant span `E`, so the gradient is bounded and Lipschitz, and
every trajectory has a limit point up to their integer combinations. -/
theorem ae_exists_tendsto_gradientFlow_isMinOn_of_span_eq_top (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) {Λ : Set E}
    (hspan : Submodule.span ℝ Λ = ⊤) (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y)
    (hcrit : ∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∀ᵐ x ∂ν, ∃ w, IsMinOn f univ w ∧ Tendsto (fun t => gradientFlow f t x) atTop (𝓝 w) := by
  obtain ⟨L, hM, -, hL⟩ := exists_lipschitzWith_gradient_of_periodic hf hspan hΛ
  exact ae_exists_tendsto_gradientFlow_isMinOn μ hν hf hL hM
    (fun v hv => periodic_of_mem_span hΛ hv) hcrit hloj (Eventually.of_forall fun x =>
      exists_frequently_norm_add_sub_lt_of_span_eq_top hspan _)

/-- The hypotheses of `gradient_eq_zero_of_tendsto_gradientFlow` hold for
`f(x) = ‖x‖² / 2` on the line and the trajectory `x(t) = e^{-t}`, which
converges to the critical point `0`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    (∀ t, 0 ≤ t → HasDerivAt (fun t => Real.exp (-t)) (-gradient f (Real.exp (-t))) t) ∧
      ContinuousAt (gradient f) 0 ∧ Tendsto (fun t => Real.exp (-t)) atTop (𝓝 0) := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  refine ⟨fun t _ => ?_, by rw [hg]; exact continuousAt_id, Real.tendsto_exp_neg_atTop_nhds_zero⟩
  rw [hg, id_eq]
  simpa using (hasDerivAt_neg t).exp

/-- The hypotheses of the other theorems above hold for `f(x) = 1 - cos x` on
the line with Lebesgue measure, `L = M = 1`, and the period `2π` or its integer
multiples.  The even multiples of `π` are global minimizers, the odd ones strict
saddles, and the Hessian `±1` is injective at both, so the Łojasiewicz
inequality holds with `a = 1/2`. -/
example : let f := fun x : ℝ => 1 - Real.cos x
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧ ContDiff ℝ 2 f ∧
      LipschitzWith 1 (gradient f) ∧ (∀ x, ‖gradient f x‖ ≤ 1) ∧
      Submodule.span ℝ {2 * Real.pi} = ⊤ ∧ (∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y) ∧
      (∀ z, gradient f z = 0 → IsMinOn f univ z ∨ IsStrictSaddle f z) ∧
      (∀ z, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) ∧
      ∀ᵐ x ∂(volume : Measure ℝ), ∃ z, ∀ ε > 0, ∃ᶠ t in atTop,
        ∃ v ∈ (Submodule.span ℤ {2 * Real.pi} : Set ℝ), ‖gradientFlow f t x + v - z‖ < ε := by
  intro f
  have hg : gradient f = Real.sin := funext fun x => by
    rw [gradient_eq_deriv', ((Real.hasDerivAt_cos x).const_sub 1).deriv, neg_neg]
  have hf : ContDiff ℝ 2 f := contDiff_const.sub Real.contDiff_cos
  have hH : ∀ z, fderiv ℝ (gradient f) z = ContinuousLinearMap.toSpanSingleton ℝ (Real.cos z) :=
    fun z => by rw [hg, (Real.hasDerivAt_sin z).hasFDerivAt.fderiv]
  have hspan : Submodule.span ℝ {2 * Real.pi} = ⊤ :=
    Ideal.span_singleton_eq_top.2 (isUnit_iff_ne_zero.2 Real.two_pi_pos.ne')
  have hΛ : ∀ v ∈ ({2 * Real.pi} : Set ℝ), ∀ y, f (y + v) = f y := fun v hv y => by
    simp [Set.mem_singleton_iff.1 hv, f]
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl, hf,
    by rw [hg]; exact Real.lipschitzWith_sin,
    fun x => by rw [hg, Real.norm_eq_abs]; exact Real.abs_sin_le_one x, hspan, hΛ,
    fun v hv => periodic_of_mem_span hΛ hv, fun z hz => ?_,
    fun z hz => ⟨1 / 2, by norm_num, by norm_num, ?_⟩,
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
