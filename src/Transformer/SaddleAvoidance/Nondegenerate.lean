/-
# Saddle avoidance — gradient descent near nondegenerate critical points

For almost every start, an orbit of gradient descent with a limit point `z`
up to translations where the Hessian is injective converges to a local
minimizer that differs from `z` by a translation
(`ae_exists_tendsto_isLocalMin_of_frequently`): `z` is a critical point
(`gradient_eq_zero_of_frequently`) where the Łojasiewicz inequality holds
(`isLojasiewiczAt_of_injective`), so the orbit converges
(`exists_tendsto_of_isLojasiewiczAt`); the limit has an injective Hessian
and is almost surely not a strict saddle (`ae_not_tendsto_of_isStrictSaddle`),
hence a local minimizer (`exists_pos_mul_norm_sq_le_inner`,
`isLocalMin_of_mul_norm_sq_le_inner`).  When every point has a translate in
one compact set and the Hessian is injective at every critical point, as for
a Morse function on a torus, almost every orbit converges to a local
minimizer (`ae_exists_tendsto_isLocalMin_of_injective`).
-/

import Transformer.SaddleAvoidance.CriticalPoints
import Transformer.SaddleAvoidance.SecondOrder
import Transformer.SaddleAvoidance.Minimizers
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

open Filter Metric MeasureTheory
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E] {f : E → ℝ} {L : ℝ≥0} {η : ℝ}

/-- **Gradient descent converges to a local minimizer from a nondegenerate
limit point**, for almost every start.  Let `f` be `C²` with an `L`-Lipschitz
gradient, `0 < η`, `η L < 1`, and let the closed set `Λ` leave `f` invariant.
For a start drawn from a law `ν` absolutely continuous with respect to a Haar
measure, almost surely: if `z` is a limit point of the orbit up to
translations in `Λ` and the Hessian at `z` is injective, then the orbit
converges to a local minimizer `w` with `z - w ∈ Λ`. -/
theorem ae_exists_tendsto_isLocalMin_of_frequently (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E} (hΛc : IsClosed Λ)
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) :
    ∀ᵐ x ∂ν, ∀ z, Function.Injective (fderiv ℝ (gradient f) z) →
      (∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f η)^[k] x + v - z‖ < ε) →
      ∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) ∧ IsLocalMin f w ∧
        z - w ∈ Λ := by
  filter_upwards [ae_not_tendsto_of_isStrictSaddle μ hν hf hL hη hηL] with x hx z hinj hz
  have hfd := hf.differentiable (by norm_num)
  have hdg := (contDiff_gradient hf).differentiable one_ne_zero
  obtain ⟨w, hw⟩ := exists_tendsto_of_isLojasiewiczAt hfd hL hη hηL hΛ (by norm_num)
    (by norm_num) (isLojasiewiczAt_of_injective hfd (hdg z).hasFDerivAt
      (gradient_eq_zero_of_frequently hfd hL hη hηL hΛc hΛ hz) hinj) hz
  have hzw := sub_mem_of_tendsto_of_frequently hΛc hw hz
  have hw0 := gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw
  have hfw : fderiv ℝ (gradient f) w = fderiv ℝ (gradient f) z := by
    have h := fderiv_comp_add_right (𝕜 := ℝ) (f := gradient f) (x := w) (z - w)
    rwa [funext (gradient_add_eq (hΛ _ hzw)), add_sub_cancel] at h
  obtain ⟨c, hc, hpos⟩ := exists_pos_mul_norm_sq_le_inner hf hw0 (fun hs => hx w hs hw)
    (by rw [hfw]; exact hinj)
  exact ⟨w, hw, isLocalMin_of_mul_norm_sq_le_inner hfd (hdg w).hasFDerivAt hw0 hc hpos, hzw⟩

/-- **Gradient descent converges to a local minimizer from almost every
start when every critical point is nondegenerate** and every point has a
translate in one compact set `K`, as for a Morse function on a torus.  Let
`f` be `C²` with an `L`-Lipschitz gradient, `0 < η`, `η L < 1`, let the
closed set `Λ` leave `f` invariant, and let the Hessian be injective at
every critical point.  For a start drawn from a law `ν` absolutely continuous
with respect to a Haar measure, the orbit almost surely converges to a local
minimizer. -/
theorem ae_exists_tendsto_isLocalMin_of_injective (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hη : 0 < η) (hηL : η * L < 1) {Λ : Set E} (hΛc : IsClosed Λ)
    (hΛ : ∀ v ∈ Λ, ∀ y, f (y + v) = f y) {K : Set E} (hK : IsCompact K)
    (hKΛ : ∀ y, ∃ v ∈ Λ, y + v ∈ K)
    (hcrit : ∀ z, gradient f z = 0 → Function.Injective (fderiv ℝ (gradient f) z)) :
    ∀ᵐ x ∂ν, ∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) ∧
      IsLocalMin f w := by
  filter_upwards [ae_exists_tendsto_isLocalMin_of_frequently μ hν hf hL hη hηL hΛc hΛ]
    with x hx
  obtain ⟨z, -, hz⟩ := exists_frequently_norm_add_sub_lt hK
    (u := fun k => (gradientStep f η)^[k] x) fun k => hKΛ _
  obtain ⟨w, hw, hmin, -⟩ := hx z (hcrit z (gradient_eq_zero_of_frequently
    (hf.differentiable (by norm_num)) hL hη hηL hΛc hΛ hz)) hz
  exact ⟨w, hw, hmin⟩

/-- The hypotheses of both theorems hold for `f(x) = 1 - cos x` on the line
with Lebesgue measure, `L = 1`, `η = 1/2`, the closed set of `v` with
`cos v = 1`, which leave `f` invariant, and the compact interval `[0, 2π]`.
The critical points are the multiples of `π`, where the Hessian `cos x` is
`±1`; the orbit of `0` stays at `0`. -/
example : let f := fun x : ℝ => 1 - Real.cos x
    let Λ := {v : ℝ | Real.cos v = 1}
    (volume : Measure ℝ).IsAddHaarMeasure ∧ (volume : Measure ℝ) ≪ volume ∧ ContDiff ℝ 2 f ∧
      LipschitzWith 1 (gradient f) ∧ (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * ((1 : ℝ≥0) : ℝ) < 1 ∧
      IsClosed Λ ∧ (∀ v ∈ Λ, ∀ y, f (y + v) = f y) ∧
      IsCompact (Set.Icc 0 (2 * Real.pi)) ∧ (∀ y, ∃ v ∈ Λ, y + v ∈ Set.Icc 0 (2 * Real.pi)) ∧
      (∀ z, gradient f z = 0 → Function.Injective (fderiv ℝ (gradient f) z)) ∧
      ∀ ε > 0, ∃ᶠ k in atTop, ∃ v ∈ Λ, ‖(gradientStep f (1 / 2))^[k] 0 + v - 0‖ < ε := by
  intro f Λ
  have hg : gradient f = Real.sin := funext fun x => by
    rw [gradient_eq_deriv', ((Real.hasDerivAt_cos x).const_sub 1).deriv, neg_neg]
  refine ⟨inferInstance, Measure.AbsolutelyContinuous.rfl, contDiff_const.sub Real.contDiff_cos,
    by rw [hg]; exact Real.lipschitzWith_sin, by norm_num, by norm_num,
    isClosed_eq Real.continuous_cos continuous_const, fun v hv y => ?_, isCompact_Icc,
    fun y => ?_, fun z hz a b hab => ?_, fun ε hε => ?_⟩
  · have hv' : Real.cos v = 1 := hv
    have hs : Real.sin v = 0 := by
      have h := Real.sin_sq_add_cos_sq v
      rw [hv'] at h
      exact pow_eq_zero_iff two_ne_zero |>.1 (by linarith)
    simp only [f, Real.cos_add, hv', hs, mul_one, mul_zero, sub_zero]
  · refine ⟨-(toIcoDiv Real.two_pi_pos 0 y • (2 * Real.pi)), ?_, ?_⟩
    · show Real.cos _ = 1
      rw [Real.cos_neg, zsmul_eq_mul, Real.cos_int_mul_two_pi]
    · rw [← sub_eq_add_neg, self_sub_toIcoDiv_zsmul]
      have h := toIcoMod_mem_Ico Real.two_pi_pos 0 y
      rw [zero_add] at h
      exact Set.Ico_subset_Icc_self h
  · rw [hg] at hz hab
    rw [(Real.hasDerivAt_sin z).hasFDerivAt.fderiv] at hab
    simp only [ContinuousLinearMap.toSpanSingleton_apply, smul_eq_mul] at hab
    have hc : Real.cos z ≠ 0 := by
      rcases Real.sin_eq_zero_iff_cos_eq.1 hz with h | h <;> rw [h] <;> norm_num
    exact mul_right_cancel₀ hc hab
  · have h0 : gradientStep f (1 / 2) 0 = 0 := by
      simp [gradientStep, hg]
    refine Frequently.of_forall fun k => ⟨0, show Real.cos 0 = 1 from Real.cos_zero, ?_⟩
    rw [Function.iterate_fixed h0, add_zero, sub_zero, norm_zero]
    exact hε

end Transformer.SaddleAvoidance
