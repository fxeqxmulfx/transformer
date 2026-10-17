/-
# Saddle avoidance — gradient descent from a compact sublevel set

`measure_tendsto_strictSaddle_eq_zero` asks for a bound `‖∇²f‖ ≤ L` on all of
`E`, which a function growing faster than quadratically, such as the square
loss `‖W₂W₁ - Y‖²` of a two-layer linear network, does not have.  A
forward-invariant set replaces it.

* If gradient descent maps a set `K` into itself and `‖∇²f‖ ≤ L` on `K`, with
  `0 < η` and `η L < 1`, the starts in `K` from which gradient descent
  converges to a strict saddle form a null set
  (`measure_inter_tendsto_strictSaddle_eq_zero`): the step has an invertible
  derivative on `K` (`det_fderiv_gradientStep_ne_zero_of_norm_le`), which
  suffices on a forward-invariant set
  (`measure_inter_tendsto_symmetricUnstable_eq_zero`).
* A sublevel set `{f ≤ c}` is forward-invariant when `‖∇f‖ ≤ M` on it,
  `‖∇²f‖ ≤ L` on its closed `δ`-thickening, `η M ≤ δ` and `η L ≤ 1`
  (`mapsTo_gradientStep_sublevel`): a step moves a point by at most `δ`, and
  the descent lemma on the ball of radius `δ` about it
  (`le_add_inner_add_mul_norm_sq_of_convex`) shows that the step does not raise
  `f`.  Hence the avoidance theorem on `{f ≤ c}`
  (`measure_sublevel_tendsto_strictSaddle_eq_zero`).
* When `{f ≤ c}` is compact these bounds exist, so for every step size below
  some `η₀ > 0` gradient descent maps `{f ≤ c}` into itself and the starts in
  `{f ≤ c}` converging to a strict saddle form a null set
  (`exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`).

The gradient-descent corollary in §5.1 of arXiv:1710.07406 assumes a globally
`L`-Lipschitz gradient (`assumption:lipschitz-gradient-descent`) and a step
size `α < 1/L`; here the bounds are needed only near a sublevel set.
-/

import Transformer.SaddleAvoidance.Descent
import Transformer.SaddleAvoidance.GradientDescent
import Mathlib.Topology.MetricSpace.Thickening

open Set Metric Filter MeasureTheory InnerProductSpace
open scoped Topology

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

local notation "⟪" x ", " y "⟫" => inner ℝ x y

/-- **Descent lemma on a convex set.**  If `‖∇²f‖ ≤ L` on a convex set `s`
containing `x` and `y`, then `f y ≤ f x + ⟪∇f(x), y - x⟫ + L ‖y - x‖²`: by
the mean value inequality, `∇f` is `L`-Lipschitz on `s`. -/
theorem le_add_inner_add_mul_norm_sq_of_convex {f : E → ℝ} (hf : ContDiff ℝ 2 f) {s : Set E}
    (hs : Convex ℝ s) {L : ℝ} (hL : ∀ w ∈ s, ‖fderiv ℝ (gradient f) w‖ ≤ L) {x y : E}
    (hx : x ∈ s) (hy : y ∈ s) :
    f y ≤ f x + ⟪gradient f x, y - x⟫ + L * ‖y - x‖ ^ 2 := by
  have hg := contDiff_gradient hf
  refine le_add_inner_add_mul_norm_sq_of_segment (hf.differentiable two_ne_zero)
    ((norm_nonneg _).trans (hL x hx)) fun w hw => ?_
  exact hs.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun v _ => (hg.differentiable one_ne_zero v).hasFDerivAt.hasFDerivWithinAt) hL hx
    (hs.segment_subset hx hy hw)

/-- **A sublevel set is forward-invariant under small steps.**  If `‖∇f‖ ≤ M`
on `{f ≤ c}`, `‖∇²f‖ ≤ L` on its closed `δ`-thickening, `0 ≤ η`, `η M ≤ δ`
and `η L ≤ 1`, gradient descent maps `{f ≤ c}` into itself: a step moves `x`
by `η ‖∇f(x)‖ ≤ δ`, and by the descent lemma on `B̄(x, δ)` it lowers `f` by at
least `η (1 - η L) ‖∇f(x)‖² ≥ 0`. -/
theorem mapsTo_gradientStep_sublevel {f : E → ℝ} (hf : ContDiff ℝ 2 f) {c δ M L η : ℝ}
    (hM : ∀ x, f x ≤ c → ‖gradient f x‖ ≤ M)
    (hL : ∀ y ∈ cthickening δ {x | f x ≤ c}, ‖fderiv ℝ (gradient f) y‖ ≤ L) (hη : 0 ≤ η)
    (hηM : η * M ≤ δ) (hηL : η * L ≤ 1) :
    MapsTo (gradientStep f η) {x | f x ≤ c} {x | f x ≤ c} := by
  intro x hx
  have hx' : f x ≤ c := hx
  have hstep : gradientStep f η x - x = -(η • gradient f x) := by simp [gradientStep]
  have hdist : ‖gradientStep f η x - x‖ ≤ δ := by
    rw [hstep, norm_neg, norm_smul, Real.norm_of_nonneg hη]
    exact (mul_le_mul_of_nonneg_left (hM x hx') hη).trans hηM
  have h := le_add_inner_add_mul_norm_sq_of_convex hf (convex_closedBall x δ)
    (fun w hw => hL w (mem_cthickening_of_dist_le w x δ _ hx (mem_closedBall.1 hw)))
    (mem_closedBall_self ((norm_nonneg _).trans hdist)) (mem_closedBall_iff_norm.2 hdist)
  rw [hstep, inner_neg_right, real_inner_smul_right, real_inner_self_eq_norm_sq, norm_neg,
    norm_smul, Real.norm_eq_abs, mul_pow, sq_abs] at h
  have hnn := mul_nonneg (mul_nonneg hη (sub_nonneg.2 hηL)) (sq_nonneg ‖gradient f x‖)
  show f (gradientStep f η x) ≤ c
  linarith

variable [MeasurableSpace E] [BorelSpace E]

/-- **Gradient descent almost never converges to a strict saddle from a
forward-invariant set.**  If `f` is `C²`, gradient descent maps `K` into
itself, `‖∇²f‖ ≤ L` on `K`, `0 < η` and `η L < 1`, the starts in `K` from
which gradient descent converges to a strict saddle form a null set.  No bound
on `∇²f` off `K` is needed. -/
theorem measure_inter_tendsto_strictSaddle_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure]
    {f : E → ℝ} (hf : ContDiff ℝ 2 f) {K : Set E} {L η : ℝ}
    (hK : MapsTo (gradientStep f η) K K) (hL : ∀ x ∈ K, ‖fderiv ℝ (gradient f) x‖ ≤ L)
    (hη : 0 < η) (hηL : η * L < 1) :
    μ (K ∩ {x | ∃ z, IsStrictSaddle f z ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z)}) = 0 := by
  refine measure_mono_null ?_ (measure_inter_tendsto_symmetricUnstable_eq_zero μ
    (contDiff_gradientStep hf η) hK fun x hx =>
      det_fderiv_gradientStep_ne_zero_of_norm_le hf (hL x hx) hη.le hηL)
  rintro x ⟨hxK, z, hz, hlim⟩
  exact ⟨hxK, z, hz.isSymmetricUnstable hf hη, hlim⟩

/-- **Gradient descent from a sublevel set almost never converges to a strict
saddle**, with bounds near the sublevel set only.  If `‖∇f‖ ≤ M` on
`{f ≤ c}`, `‖∇²f‖ ≤ L` on its closed `δ`-thickening, `0 < η`, `η M ≤ δ` and
`η L < 1`, the starts in `{f ≤ c}` from which gradient descent converges to a
strict saddle form a null set. -/
theorem measure_sublevel_tendsto_strictSaddle_eq_zero (μ : Measure E) [μ.IsAddHaarMeasure]
    {f : E → ℝ} (hf : ContDiff ℝ 2 f) {c δ M L η : ℝ}
    (hM : ∀ x, f x ≤ c → ‖gradient f x‖ ≤ M)
    (hL : ∀ y ∈ cthickening δ {x | f x ≤ c}, ‖fderiv ℝ (gradient f) y‖ ≤ L) (hη : 0 < η)
    (hηM : η * M ≤ δ) (hηL : η * L < 1) :
    μ ({x | f x ≤ c} ∩ {x | ∃ z, IsStrictSaddle f z ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z)}) = 0 :=
  measure_inter_tendsto_strictSaddle_eq_zero μ hf
    (mapsTo_gradientStep_sublevel hf hM hL hη.le hηM hηL.le)
    (fun x hx => hL x (self_subset_cthickening _ hx)) hη hηL

/-- **A compact sublevel set replaces the global Lipschitz constant.**  If `f`
is `C²` and `{f ≤ c}` is compact, there is `η₀ > 0` such that for every step
size `0 < η < η₀` gradient descent maps `{f ≤ c}` into itself and the starts
in `{f ≤ c}` from which it converges to a strict saddle form a null set.  A
bound `M` on `‖∇f‖` over `{f ≤ c}` and a bound `L` on `‖∇²f‖` over its closed
`1`-thickening, compact as well, give `η₀ = 1/(|M| + |L| + 1)`. -/
theorem exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero (μ : Measure E)
    [μ.IsAddHaarMeasure] {f : E → ℝ} (hf : ContDiff ℝ 2 f) {c : ℝ}
    (hK : IsCompact {x | f x ≤ c}) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ →
      MapsTo (gradientStep f η) {x | f x ≤ c} {x | f x ≤ c} ∧
      μ ({x | f x ≤ c} ∩ {x | ∃ z, IsStrictSaddle f z ∧
        Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 z)}) = 0 := by
  have hg := contDiff_gradient hf
  obtain ⟨M, hM⟩ := hK.exists_bound_of_continuousOn hg.continuous.continuousOn
  obtain ⟨L, hL⟩ := (hK.cthickening (r := 1)).exists_bound_of_continuousOn
    (hg.continuous_fderiv one_ne_zero).continuousOn
  refine ⟨1 / (|M| + |L| + 1), by positivity, fun η hη hη₀ => ?_⟩
  rw [lt_div_iff₀ (by positivity)] at hη₀
  have hηM : η * M ≤ 1 := by nlinarith [le_abs_self M, abs_nonneg L]
  have hηL : η * L < 1 := by nlinarith [le_abs_self L, abs_nonneg M]
  exact ⟨mapsTo_gradientStep_sublevel hf (fun x hx => hM x hx) hL hη.le hηM hηL.le,
    measure_sublevel_tendsto_strictSaddle_eq_zero μ hf (fun x hx => hM x hx) hL hη hηM hηL⟩

/-- The hypotheses of the first four theorems hold for `f(x) = ‖x‖² / 2` on
the line, with `s = ℝ`, `K = B̄(0, 1)`, `c = 1/2`, `δ = M = L = 1` and
`η = 1/2`: the gradient of `f` is `x`, of norm at most `1` on
`{f ≤ 1/2} = B̄(0, 1)`, its Hessian is `1`, and a step halves `x`. -/
example : let f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2
    ContDiff ℝ 2 f ∧ Convex ℝ (univ : Set ℝ) ∧
      (∀ w ∈ (univ : Set ℝ), ‖fderiv ℝ (gradient f) w‖ ≤ 1) ∧
      MapsTo (gradientStep f (1 / 2)) (closedBall 0 1) (closedBall 0 1) ∧
      (∀ x ∈ closedBall (0 : ℝ) 1, ‖fderiv ℝ (gradient f) x‖ ≤ 1) ∧
      (∀ x, f x ≤ 1 / 2 → ‖gradient f x‖ ≤ 1) ∧
      (∀ y ∈ cthickening 1 {x | f x ≤ 1 / 2}, ‖fderiv ℝ (gradient f) y‖ ≤ 1) ∧
      (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) * 1 ≤ 1 ∧ (1 / 2 : ℝ) * 1 < 1 := by
  intro f
  have hg : gradient f = id := gradient_half_mul_norm_sq
  have hs : gradientStep f (1 / 2) = fun x => (1 - 1 / 2 : ℝ) • x :=
    gradientStep_half_mul_norm_sq _
  have hH : ∀ y, ‖fderiv ℝ (gradient f) y‖ ≤ 1 := fun y => by
    rw [hg, fderiv_id]
    exact ContinuousLinearMap.norm_id_le
  refine ⟨contDiff_const.mul (contDiff_norm_sq ℝ), convex_univ, fun w _ => hH w, fun x hx => ?_,
    fun x _ => hH x, fun x hx => ?_, fun y _ => hH y, by norm_num, by norm_num, by norm_num⟩
  · norm_num [hs, norm_smul] at hx ⊢
    linarith
  · rw [hg, id]
    nlinarith [norm_nonneg x, show (1 / 2 : ℝ) * ‖x‖ ^ 2 ≤ 1 / 2 from hx]

/-- The hypotheses of `exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`
hold for the double well `f(x) = x⁴/4 - x²/2` on the line and `c = 0`:
`{f ≤ 0} = [-√2, √2]` is compact.  The Hessian `3x² - 1` of `f` is unbounded,
so `measure_tendsto_strictSaddle_eq_zero` does not apply, and `0` is a strict
saddle of `f`, with Hessian `-1`. -/
example : let f := fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2
    ContDiff ℝ 2 f ∧ IsCompact {x | f x ≤ 0} ∧ IsStrictSaddle f 0 ∧
      ∀ L, ∃ x, L < ‖fderiv ℝ (gradient f) x‖ := by
  intro f
  have hd : ∀ x, HasDerivAt f (x ^ 3 - x) x := fun x => by
    convert ((hasDerivAt_pow 4 x).div_const 4).sub ((hasDerivAt_pow 2 x).div_const 2) using 1
    push_cast
    ring
  have hg : gradient f = fun x => x ^ 3 - x :=
    funext fun x => by simp [gradient_eq_deriv, (hd x).deriv]
  have hH : ∀ x, fderiv ℝ (gradient f) x =
      ContinuousLinearMap.toSpanSingleton ℝ (3 * x ^ 2 - 1) := fun x => by
    have h3 : HasDerivAt (fun x : ℝ => x ^ 3 - x) (3 * x ^ 2 - 1) x := by
      convert (hasDerivAt_pow 3 x).fun_sub (hasDerivAt_id' x) using 1
    rw [hg, h3.hasFDerivAt.fderiv]
  refine ⟨by fun_prop, (isCompact_Icc (a := -2) (b := 2)).of_isClosed_subset
    (isClosed_le (by fun_prop) continuous_const) fun x hx => ?_, ⟨by simp [hg], -1, by norm_num,
      Module.End.hasEigenvalue_of_hasEigenvector (x := 1) ⟨?_, one_ne_zero⟩⟩, fun L => ?_⟩
  · have hx' : x ^ 4 / 4 - x ^ 2 / 2 ≤ 0 := hx
    constructor <;> nlinarith [sq_nonneg x, sq_nonneg (x ^ 2 - 2)]
  · simp [hH]
  · refine ⟨|L| + 1, ?_⟩
    rw [hH, ContinuousLinearMap.norm_toSpanSingleton, Real.norm_eq_abs]
    nlinarith [le_abs_self L, abs_nonneg L, le_abs_self (3 * (|L| + 1) ^ 2 - 1)]

end Transformer.SaddleAvoidance
