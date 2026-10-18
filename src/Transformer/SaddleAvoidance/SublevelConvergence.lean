/-
# Saddle avoidance — the scheme on a compact sublevel set

The convergence theorems ask for a gradient that is Lipschitz on the whole space.  A
polynomial of degree above two or a network has none, while a regularized loss has
compact sublevel sets.  A cutoff replaces the global constant.  For `f` of class `C²`
and a compact `K`, the function `χ f`, with `χ` smooth with compact support and equal to
`1` near `K`, has a Lipschitz gradient and agrees with `f` near `K`
(`exists_lipschitzWith_gradient_eqOn`, in `SaddleAvoidance.Cutoff`).  When gradient
descent maps `K` into itself, its
orbits from `K` are those of `χ f` (`iterate_gradientStep_eq_of_eqOn`).  So for small
steps they converge to a critical point in `K` if `f` satisfies the Łojasiewicz
inequality at the critical points in `K` (`exists_pos_tendsto_of_mapsTo`).

With the avoidance theorem on a compact sublevel set
(`exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`): if every critical point in
`{f ≤ c}` is a local minimizer or a strict saddle and satisfies the Łojasiewicz
inequality, then for every small step almost every start in `{f ≤ c}` converges to a
local minimizer (`exists_pos_ae_sublevel_exists_tendsto_isLocalMin`).  For analytic `f`
the inequality holds everywhere
(`exists_pos_ae_sublevel_exists_tendsto_isLocalMin_of_analyticOnNhd`).

This is item M6.6 of `landscape_plan.md`: the first part of `prop:rates` of
arXiv:1602.04915, as in `ae_exists_tendsto_isLocalMin`, with a compact sublevel set in
place of the global Lipschitz constant and of the bounded orbits.
-/

import Transformer.SaddleAvoidance.Cutoff
import Transformer.SaddleAvoidance.Minimizers
import Transformer.SaddleAvoidance.Sublevel
import Transformer.SaddleAvoidance.LocalInequality

open Filter Metric MeasureTheory Set
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  {f : E → ℝ} {K : Set E}

/-- **Orbits from a forward-invariant set depend only on the function near it.**  If
`F = f` on an open `V ⊇ K` and gradient descent on `f` maps `K` into itself, the orbits of
gradient descent on `F` and on `f` from a point of `K` coincide. -/
theorem iterate_gradientStep_eq_of_eqOn {F : E → ℝ} {V : Set E} (hV : IsOpen V)
    (hKV : K ⊆ V) (hFf : EqOn F f V) {η : ℝ} (hK : MapsTo (gradientStep f η) K K) {x : E}
    (hx : x ∈ K) (k : ℕ) : (gradientStep F η)^[k] x = (gradientStep f η)^[k] x := by
  induction k with
  | zero => rfl
  | succ k ih =>
    have hg : gradient F ((gradientStep f η)^[k] x) = gradient f ((gradientStep f η)^[k] x) :=
      (eventuallyEq_of_mem (hV.mem_nhds (hKV (hK.iterate k hx))) hFf).gradient_eq
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply', ih, gradientStep,
      gradientStep, hg]

/-- **Gradient descent converges from a compact set it maps into itself**, with no global
Lipschitz constant.  Let `f` be `C²`, `K` compact, and let `f` satisfy the Łojasiewicz
inequality with an exponent `0 ≤ a < 1` at its critical points in `K`.  For some `η₁ > 0`
and every `0 < η < η₁` for which gradient descent maps `K` into itself, every orbit from
`K` converges to a critical point in `K`.  The orbits from `K` are those of the cutoff of
`exists_lipschitzWith_gradient_eqOn`, to which `tendsto_of_isLojasiewiczAt` applies at a
limit point in `K`. -/
theorem exists_pos_tendsto_of_mapsTo (hf : ContDiff ℝ 2 f) (hK : IsCompact K)
    (hloj : ∀ z ∈ K, gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∃ η₁ > 0, ∀ η, 0 < η → η < η₁ → MapsTo (gradientStep f η) K K → ∀ x ∈ K,
      ∃ w ∈ K, gradient f w = 0 ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨F, hF, ⟨L, hL⟩, V, hV, hKV, hFf⟩ := exists_lipschitzWith_gradient_eqOn hf hK
  refine ⟨1 / ((L : ℝ) + 1), by positivity, fun η hη hη₁ hmaps x hx => ?_⟩
  rw [lt_div_iff₀ (by positivity)] at hη₁
  have hηL : η * L < 1 := by nlinarith
  have hgerm : ∀ z ∈ K, F =ᶠ[𝓝 z] f := fun z hz =>
    eventuallyEq_of_mem (hV.mem_nhds (hKV hz)) hFf
  have horb : (fun k => (gradientStep F η)^[k] x) = fun k => (gradientStep f η)^[k] x :=
    funext (iterate_gradientStep_eq_of_eqOn hV hKV hFf hmaps hx)
  obtain ⟨z, hzK, hcl⟩ := hK.exists_mapClusterPt (f := atTop)
    (u := fun k => (gradientStep f η)^[k] x)
    (tendsto_principal.2 (Eventually.of_forall fun k => hmaps.iterate k hx))
  obtain ⟨a, ha0, ha1, ha⟩ : ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt F z a := by
    by_cases h0 : gradient F z = 0
    · obtain ⟨a, ha0, ha1, ha⟩ := hloj z hzK (by rwa [← (hgerm z hzK).gradient_eq])
      exact ⟨a, ha0, ha1, ha.congr_of_eventuallyEq (hgerm z hzK).symm⟩
    · exact ⟨0, le_rfl, one_pos,
        isLojasiewiczAt_of_gradient_ne_zero hL.continuous.continuousAt h0 le_rfl⟩
  have hlim := tendsto_of_isLojasiewiczAt (hF.differentiable (by norm_num)) hL hη hηL ha0 ha1
    ha (by rwa [horb])
  refine ⟨z, hzK, ?_, by rwa [← horb]⟩
  rw [← (hgerm z hzK).gradient_eq]
  exact gradient_eq_zero_of_tendsto hL.continuous hη.ne' hlim

variable [MeasurableSpace E] [BorelSpace E]

/-- **The scheme on a compact sublevel set.**  Let `f` be `C²` with `{f ≤ c}` compact, and
let every critical point in `{f ≤ c}` be a local minimizer or a strict saddle and satisfy
the Łojasiewicz inequality with an exponent `0 ≤ a < 1`.  For some `η₀ > 0` and every
`0 < η < η₀`, the orbit of gradient descent from almost every start in `{f ≤ c}`
converges to a local minimizer in `{f ≤ c}`.  No global Lipschitz constant is needed:
small steps keep the orbits in `{f ≤ c}`, where almost none converges to a strict saddle
(`exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero`) and all converge
(`exists_pos_tendsto_of_mapsTo`). -/
theorem exists_pos_ae_sublevel_exists_tendsto_isLocalMin (μ : Measure E) [μ.IsAddHaarMeasure]
    (hf : ContDiff ℝ 2 f) {c : ℝ} (hK : IsCompact {x | f x ≤ c})
    (hcrit : ∀ z, f z ≤ c → gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, f z ≤ c → gradient f z = 0 → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ᵐ x ∂μ, f x ≤ c → ∃ w, f w ≤ c ∧ IsLocalMin f w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) := by
  obtain ⟨η₁, hη₁, havoid⟩ := exists_pos_measure_sublevel_tendsto_strictSaddle_eq_zero μ hf hK
  obtain ⟨η₂, hη₂, hconv⟩ := exists_pos_tendsto_of_mapsTo hf hK hloj
  refine ⟨min η₁ η₂, lt_min hη₁ hη₂, fun η hη hη₀ => ?_⟩
  obtain ⟨hmaps, hnull⟩ := havoid η hη (hη₀.trans_le (min_le_left _ _))
  filter_upwards [measure_eq_zero_iff_ae_notMem.1 hnull] with x hx hxc
  obtain ⟨w, hwK, hw0, hw⟩ := hconv η hη (hη₀.trans_le (min_le_right _ _)) hmaps x hxc
  exact ⟨w, hwK, (hcrit w hwK hw0).resolve_right fun hs => hx ⟨hxc, w, hs, hw⟩, hw⟩

/-- **The scheme on a compact sublevel set for an analytic function**, without the
Łojasiewicz hypothesis, which analyticity gives at every point
(`exists_isLojasiewiczAt_of_analyticOnNhd`). -/
theorem exists_pos_ae_sublevel_exists_tendsto_isLocalMin_of_analyticOnNhd (μ : Measure E)
    [μ.IsAddHaarMeasure] (hf : AnalyticOnNhd ℝ f univ) {c : ℝ} (hK : IsCompact {x | f x ≤ c})
    (hcrit : ∀ z, f z ≤ c → gradient f z = 0 → IsLocalMin f z ∨ IsStrictSaddle f z) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ᵐ x ∂μ, f x ≤ c → ∃ w, f w ≤ c ∧ IsLocalMin f w ∧
      Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) :=
  exists_pos_ae_sublevel_exists_tendsto_isLocalMin μ hf.contDiff hK hcrit fun z _ _ =>
    exists_isLojasiewiczAt_of_analyticOnNhd hf z

/-- The hypotheses of `iterate_gradientStep_eq_of_eqOn` hold for `F = f = x²` and
`V = K = ℝ`. -/
example := iterate_gradientStep_eq_of_eqOn (f := fun x : ℝ => x ^ 2) isOpen_univ
  (subset_univ univ) (eqOn_refl _ univ) (η := 1) (mapsTo_univ _ _) (mem_univ 0) 3

/-- The hypotheses of `exists_pos_tendsto_of_mapsTo` hold for `f(x) = ‖x‖² / 2` on the
line and `K = B̄(0, 1)`: the only critical point `0` is a minimizer, where the Łojasiewicz
inequality holds with `a = 1/2`. -/
example := exists_pos_tendsto_of_mapsTo (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2)
  (contDiff_const.mul (contDiff_norm_sq ℝ)) (isCompact_closedBall (0 : ℝ) 1) fun z _ hz =>
    ⟨1 / 2, by norm_num, by norm_num, by
      rw [gradient_half_mul_norm_sq, id_eq] at hz
      subst hz
      exact isLojasiewiczAt_half_mul_norm_sq⟩

/-- The hypotheses of `exists_pos_ae_sublevel_exists_tendsto_isLocalMin` hold for
`f(x) = ‖x‖² / 2` on the line, Lebesgue measure and `c = 1/2`: `{f ≤ 1/2} = B̄(0, 1)`, and
the only critical point `0` is a minimizer, where the Łojasiewicz inequality holds with
`a = 1/2`. -/
example := exists_pos_ae_sublevel_exists_tendsto_isLocalMin (volume : Measure ℝ)
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (c := 1 / 2) (contDiff_const.mul (contDiff_norm_sq ℝ))
  ((isCompact_closedBall (0 : ℝ) 1).of_isClosed_subset
    (isClosed_le (by fun_prop) continuous_const) fun x hx => by
      have hx' : (1 / 2 : ℝ) * ‖x‖ ^ 2 ≤ 1 / 2 := hx
      rw [mem_closedBall_zero_iff]
      nlinarith [norm_nonneg x])
  (fun z _ hz => Or.inl (by
    rw [gradient_half_mul_norm_sq, id_eq] at hz
    subst hz
    exact Eventually.of_forall fun y => by
      simp only [norm_zero]
      nlinarith [sq_nonneg ‖y‖]))
  fun z _ hz => ⟨1 / 2, by norm_num, by norm_num, by
    rw [gradient_half_mul_norm_sq, id_eq] at hz
    subst hz
    exact isLojasiewiczAt_half_mul_norm_sq⟩

/-- The hypotheses of `exists_pos_ae_sublevel_exists_tendsto_isLocalMin_of_analyticOnNhd`
hold for the double well `f(x) = x⁴/4 - x²/2` on the line, Lebesgue measure and `c = 0`.
The Hessian `3x² - 1` of `f` is unbounded, and `{f ≤ 0} = [-√2, √2]` is compact.  The
critical points `±1` are minimizers, as `f + 1/4 = (x² - 1)²/4`, and `0` is a strict
saddle, with Hessian `-1`. -/
example := exists_pos_ae_sublevel_exists_tendsto_isLocalMin_of_analyticOnNhd
  (volume : Measure ℝ) (f := fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2) (c := 0)
  (fun x _ => by fun_prop)
  ((isCompact_Icc (a := -2) (b := 2)).of_isClosed_subset
    (isClosed_le (by fun_prop) continuous_const) fun x hx => by
      have hx' : x ^ 4 / 4 - x ^ 2 / 2 ≤ 0 := hx
      constructor <;> nlinarith [sq_nonneg x, sq_nonneg (x ^ 2 - 2)])
  fun z _ hz => by
    have hd : ∀ x : ℝ, HasDerivAt (fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2) (x ^ 3 - x) x :=
      fun x => by
        convert ((hasDerivAt_pow 4 x).div_const 4).sub ((hasDerivAt_pow 2 x).div_const 2)
          using 1
        push_cast
        ring
    have hg : gradient (fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2) = fun x => x ^ 3 - x :=
      funext fun x => by simp [gradient_eq_deriv, (hd x).deriv]
    have hH : ∀ x, fderiv ℝ (gradient fun x : ℝ => x ^ 4 / 4 - x ^ 2 / 2) x =
        ContinuousLinearMap.toSpanSingleton ℝ (3 * x ^ 2 - 1) := fun x => by
      have h3 : HasDerivAt (fun x : ℝ => x ^ 3 - x) (3 * x ^ 2 - 1) x := by
        convert (hasDerivAt_pow 3 x).fun_sub (hasDerivAt_id' x) using 1
      rw [hg, h3.hasFDerivAt.fderiv]
    have hz' : z * (z ^ 2 - 1) = 0 := by
      rw [hg] at hz
      linear_combination hz
    rcases mul_eq_zero.1 hz' with h0 | h1
    · subst h0
      exact Or.inr ⟨by simp [hg], -1, by norm_num,
        Module.End.hasEigenvalue_of_hasEigenvector (x := 1) ⟨by simp [hH], one_ne_zero⟩⟩
    · refine Or.inl (Eventually.of_forall fun y => ?_)
      change z ^ 4 / 4 - z ^ 2 / 2 ≤ y ^ 4 / 4 - y ^ 2 / 2
      nlinarith [sq_nonneg (y ^ 2 - 1)]

end Transformer.SaddleAvoidance
