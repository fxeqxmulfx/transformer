/-
# Saddle avoidance — the scheme along the orbit

A run of gradient descent can check a bound `‖∇²f‖ ≤ L` with `η L < 1` only along
its own orbit: on the segment `[x_k, x_{k+1}]` of every step (S).  That is enough for
the scheme.  Under (S) every step lowers `f` by at least `η (1 - η L) ‖∇f(x_k)‖²`
(`iterate_succ_le_sub_of_segment`).  An orbit that does not run off to infinity
comes back to some closed ball infinitely often, so it has a limit point `z` there
(`IsCompact.exists_mapClusterPt_of_frequently`), with `f z ≤ f x`.  The Łojasiewicz
inequality holds at `z`: by hypothesis if `z` is critical, with exponent `0`
otherwise (`isLojasiewiczAt_of_gradient_ne_zero`).  So the orbit converges to `z`
(`tendsto_of_isLojasiewiczAt_of_le_sub`), a critical point: an orbit satisfying (S)
converges to a critical point no higher than its start, or `‖x_k‖ → ∞`
(`exists_tendsto_or_tendsto_norm_of_segment`).  Almost no start whose orbit
satisfies (S) converges to a strict saddle (`ae_not_tendsto_strictSaddle_of_segment`).
So almost every start below a level `c` whose orbit satisfies (S) converges to a
critical point that is not a strict saddle, or runs off to infinity
(`ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_lt_of_segment`); to a
global minimizer if every critical point below `c` is one or a strict saddle
(`ae_exists_tendsto_isMinOn_or_tendsto_norm_of_lt_of_segment`).  The Łojasiewicz
inequality is asked only below `c`; there is no global Lipschitz constant of `∇f`,
no hypothesis of a limit point and no compact sublevel set.  `AnalyticOrbitLandscape`
has the forms for analytic `f`, with neither the Łojasiewicz hypothesis nor a level.

Source: item M8.3 of `landscape_plan.md`.
-/

import Transformer.Landscape.SaddleAvoidance.Orbit
import Transformer.Landscape.SaddleAvoidance.Convergence

open Set Filter MeasureTheory Metric
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  {f : E → ℝ} {L η : ℝ}

/-- **An orbit satisfying (S) converges to a critical point or runs off to
infinity.**  Let `f` be `C²`, `0 < η` and `η L < 1`, let the orbit of `x` have
`‖∇²f‖ ≤ L` on the segment of every step, and let `f` satisfy the Łojasiewicz
inequality, with an exponent `0 ≤ a < 1`, at every critical point `z` with
`f z ≤ f x`.  Then the orbit converges to a critical point `w` with `f w ≤ f x`, or
`‖x_k‖ → ∞`. -/
theorem exists_tendsto_or_tendsto_norm_of_segment (hf : ContDiff ℝ 2 f) (hη : 0 < η)
    (hηL : η * L < 1) {x : E}
    (hS : ∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
      ‖fderiv ℝ (gradient f) y‖ ≤ L)
    (hloj : ∀ z, gradient f z = 0 → f z ≤ f x → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    (∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) ∧ gradient f w = 0 ∧
        f w ≤ f x) ∨
      Tendsto (fun k => ‖(gradientStep f η)^[k] x‖) atTop atTop := by
  have hc : Continuous (gradient f) := (contDiff_gradient hf).continuous
  have hfc : Continuous f := hf.continuous
  have hκ : 0 < 1 - η * L := by linarith
  have hdec := iterate_succ_le_sub_of_segment hf hS
  refine or_iff_not_imp_right.2 fun hinf => ?_
  obtain ⟨R, hR⟩ : ∃ R, ∃ᶠ k in atTop, (gradientStep f η)^[k] x ∈ closedBall (0 : E) R := by
    simp only [Filter.tendsto_atTop, not_forall, not_eventually, not_le] at hinf
    obtain ⟨R, hR⟩ := hinf
    exact ⟨R, hR.mono fun k hk => by rw [mem_closedBall, dist_zero_right]; exact hk.le⟩
  obtain ⟨z, -, hz⟩ := (isCompact_closedBall (0 : E) R).exists_mapClusterPt_of_frequently hR
  have hb : ∀ k, f ((gradientStep f η)^[k] x) ≤ f x := fun k =>
    antitone_iterate_of_le_sub hη.le hκ.le hdec (Nat.zero_le k)
  have hfz : f z ≤ f x := by
    by_contra h
    push Not at h
    obtain ⟨k, hk⟩ := (mapClusterPt_iff_frequently.1 hz _
      ((hfc.tendsto z).eventually (lt_mem_nhds h))).exists
    exact (hb k).not_gt hk
  obtain ⟨a, ha0, ha1, hza⟩ : ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a := by
    by_cases h0 : gradient f z = 0
    · exact hloj z h0 hfz
    · exact ⟨0, le_rfl, one_pos, isLojasiewiczAt_of_gradient_ne_zero hc.continuousAt h0 le_rfl⟩
  have hw := tendsto_of_isLojasiewiczAt_of_le_sub hη hκ ha0 ha1 hza hfc.continuousAt hdec hz
  exact ⟨z, hw, gradient_eq_zero_of_tendsto hc hη.ne' hw, hfz⟩

variable [MeasurableSpace E] [BorelSpace E]

/-- **The scheme along the orbit.**  Let `f` be `C²`, `0 < η` and `η L < 1`, let `μ`
be a Haar measure and `ν ≪ μ`, and let `f` satisfy the Łojasiewicz inequality, with
an exponent `0 ≤ a < 1`, at every critical point below `c`.  For `ν`-almost every `x`
with `f x < c` whose orbit has `‖∇²f‖ ≤ L` on the segment of every step, either the
orbit converges to a critical point `w` with `f w ≤ f x` that is not a strict saddle,
or `‖x_k‖ → ∞`. -/
theorem ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_lt_of_segment
    (μ : Measure E) [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f)
    (hη : 0 < η) (hηL : η * L < 1) {c : ℝ}
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∀ᵐ x ∂ν, f x < c →
      (∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
        ‖fderiv ℝ (gradient f) y‖ ≤ L) →
      (∃ w, Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w) ∧ gradient f w = 0 ∧
          f w ≤ f x ∧ ¬IsStrictSaddle f w) ∨
        Tendsto (fun k => ‖(gradientStep f η)^[k] x‖) atTop atTop := by
  filter_upwards [ae_not_tendsto_strictSaddle_of_segment μ hν hf hη hηL] with x hx hxc hS
  exact (exists_tendsto_or_tendsto_norm_of_segment hf hη hηL hS
    fun z hz hzx => hloj z hz (hzx.trans_lt hxc)).imp_left fun ⟨w, hw, hw0, hwx⟩ =>
      ⟨w, hw, hw0, hwx, fun hs => hx hS w hs hw⟩

/-- **The scheme along the orbit, to a global minimizer.**  If moreover every
critical point below `c` is a global minimizer or a strict saddle, then for
`ν`-almost every `x` with `f x < c` whose orbit satisfies (S), the orbit converges to
a global minimizer, or `‖x_k‖ → ∞`. -/
theorem ae_exists_tendsto_isMinOn_or_tendsto_norm_of_lt_of_segment (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hη : 0 < η)
    (hηL : η * L < 1) {c : ℝ}
    (hcrit : ∀ z, gradient f z = 0 → f z < c → IsMinOn f univ z ∨ IsStrictSaddle f z)
    (hloj : ∀ z, gradient f z = 0 → f z < c → ∃ a, 0 ≤ a ∧ a < 1 ∧ IsLojasiewiczAt f z a) :
    ∀ᵐ x ∂ν, f x < c →
      (∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
        ‖fderiv ℝ (gradient f) y‖ ≤ L) →
      (∃ w, IsMinOn f univ w ∧ Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) ∨
        Tendsto (fun k => ‖(gradientStep f η)^[k] x‖) atTop atTop := by
  filter_upwards [ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_lt_of_segment μ hν
    hf hη hηL hloj] with x hx hxc hS
  exact (hx hxc hS).imp_left fun ⟨w, hw, hw0, hwx, hws⟩ =>
    ⟨w, (hcrit w hw0 (hwx.trans_lt hxc)).resolve_right hws, hw⟩

/-- The gradient of `-x` on the line is `-1`. -/
theorem gradient_neg_real : gradient (fun x : ℝ => -x) = fun _ => -1 := funext fun x => by
  rw [gradient_eq_deriv']
  exact (hasDerivAt_neg x).deriv

/-- The orbit of `x` for `-x` on the line is `x + k η`. -/
theorem iterate_gradientStep_neg_real (η x : ℝ) (k : ℕ) :
    (gradientStep (fun x : ℝ => -x) η)^[k] x = x + k * η := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih]
    simp only [gradientStep, gradient_neg_real, smul_eq_mul]
    push_cast
    ring

/-- For `-x` on the line the Hessian is `0`, so every orbit satisfies (S) with
`L = 0`, here for `η = 1/10`; and every orbit `x + k η` runs off to infinity. -/
example : (∀ x : ℝ, ∀ k, ∀ y ∈ segment ℝ ((gradientStep (fun x : ℝ => -x) (1 / 10))^[k] x)
      ((gradientStep (fun x : ℝ => -x) (1 / 10))^[k + 1] x),
      ‖fderiv ℝ (gradient fun x : ℝ => -x) y‖ ≤ 0) ∧
    ∀ x : ℝ, Tendsto (fun k => ‖(gradientStep (fun x : ℝ => -x) (1 / 10))^[k] x‖) atTop
      atTop := by
  refine ⟨fun x k y _ => by rw [gradient_neg_real, fderiv_const_apply, norm_zero], fun x => ?_⟩
  simp only [iterate_gradientStep_neg_real, Real.norm_eq_abs]
  exact tendsto_abs_atTop_atTop.comp (tendsto_atTop_add_const_left _ x
    (tendsto_natCast_atTop_atTop.atTop_mul_const (by norm_num)))

/-- The dichotomy for `-x`, `L = 0` and `η = 1/10`: `f` has no critical point, so the
Łojasiewicz hypothesis is empty, and the orbit runs off to infinity. -/
example := exists_tendsto_or_tendsto_norm_of_segment (f := fun x : ℝ => -x) (L := 0)
  (η := 1 / 10) contDiff_neg (by norm_num) (by norm_num) (x := 0)
  (fun _ y _ => by rw [gradient_neg_real, fderiv_const_apply, norm_zero])
  fun z hz _ => by rw [gradient_neg_real] at hz; norm_num at hz

/-- `‖x‖² / 2` on the line with Lebesgue measure, `L = 1`, `η = 1/2` and `c = 1`: its
only critical point `0` satisfies the Łojasiewicz inequality with `a = 1/2`. -/
example := ae_exists_tendsto_not_isStrictSaddle_or_tendsto_norm_of_lt_of_segment
  (volume : Measure ℝ) Measure.AbsolutelyContinuous.rfl
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1) (η := 1 / 2)
  (contDiff_const.mul (contDiff_norm_sq ℝ)) (by norm_num) (by norm_num) (c := 1)
  fun z hz _ => ⟨1 / 2, by norm_num, by norm_num, by
    rw [gradient_half_mul_norm_sq] at hz
    rw [show z = 0 from hz]
    exact isLojasiewiczAt_half_mul_norm_sq⟩

/-- The same, where `0` is the global minimizer. -/
example := ae_exists_tendsto_isMinOn_or_tendsto_norm_of_lt_of_segment (volume : Measure ℝ)
  Measure.AbsolutelyContinuous.rfl (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1)
  (η := 1 / 2) (contDiff_const.mul (contDiff_norm_sq ℝ)) (by norm_num) (by norm_num) (c := 1)
  (fun z hz _ => Or.inl (by
    rw [gradient_half_mul_norm_sq] at hz
    rw [show z = 0 from hz]
    exact isMinOn_univ_iff.2 fun y => by simp only [norm_zero]; nlinarith [sq_nonneg ‖y‖]))
  fun z hz _ => ⟨1 / 2, by norm_num, by norm_num, by
    rw [gradient_half_mul_norm_sq] at hz
    rw [show z = 0 from hz]
    exact isLojasiewiczAt_half_mul_norm_sq⟩

end Transformer.Landscape.SaddleAvoidance
