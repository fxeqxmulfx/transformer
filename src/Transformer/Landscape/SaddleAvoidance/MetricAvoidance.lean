/-
# Saddle avoidance — flows `ẋ = -G(x) ∇f(x)`

Let `f` be a `C²` function and `G` a `C¹` field of linear maps on a
finite-dimensional real inner product space such that `G ∇f` is
`L`-Lipschitz and bounded by `M`, and `G` is symmetric positive definite at
the strict saddles of `f`.  The flow `ẋ = -G(x) ∇f(x)` converges to a strict
saddle of `f` only from a null set of starts
(`measure_tendsto_metricGradientFlow_strictSaddle_eq_zero`); started from a
law with a density, it almost surely converges to none
(`ae_not_tendsto_metricGradientFlow_of_isStrictSaddle`).

The field does not depend on time, so the flow is a one-parameter group
(`metricGradientFlow_metricGradientFlow`), and its value at the integer time
`k` is the `k`-th iterate of the time-one map
(`iterate_metricGradientFlow_one`).  The time-one map is `C¹` with an
invertible derivative, and every strict saddle is a symmetrizable unstable
fixed point of it (`Landscape.SaddleAvoidance.MetricFlow`), so
`measure_tendsto_symmetrizableUnstable_eq_zero` concludes, as
`measure_tendsto_symmetricUnstable_eq_zero` does for the gradient flow
(`Landscape.SaddleAvoidance.FlowAvoidance`).

This is the Euclidean analog of `l:nosaddleconv` in Appendix A of
arXiv:2312.10794v5 for gradient flows in a metric, with `thm:main-general`
of §4 of arXiv:1710.07406, applied to the time-one map, in place of the
center-stable manifold theorem for flows.
-/

import Transformer.Landscape.SaddleAvoidance.MetricFlow

open Filter MeasureTheory
open scoped Topology NNReal

namespace Transformer.Landscape.SaddleAvoidance

open Landscape.GlobalFlow

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {G : E → E →L[ℝ] E} {f : E → ℝ} {L : ℝ≥0} {M : ℝ}

/-- The trajectories of `metricGradientFlow G f` solve `ẋ = -G(x) ∇f(x)`. -/
theorem hasDerivAt_metricGradientFlow (hL : LipschitzWith L fun x => G x (gradient f x))
    (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M) (t : ℝ) (x : E) :
    HasDerivAt (fun t => metricGradientFlow G f t x)
      (-G (metricGradientFlow G f t x) (gradient f (metricGradientFlow G f t x))) t :=
  (isFlowField_neg_metricGradient hL hM).hasDerivAt_flow 0 t x

/-- **The flow is a one-parameter group**: flowing for time `t` and then for
time `s` is flowing for time `s + t`.  The field does not depend on time, so
`τ ↦ Φ^{τ + t}(x)` is the trajectory through `Φ^t(x)`. -/
theorem metricGradientFlow_metricGradientFlow
    (hL : LipschitzWith L fun x => G x (gradient f x)) (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M)
    (s t : ℝ) (x : E) :
    metricGradientFlow G f s (metricGradientFlow G f t x) = metricGradientFlow G f (s + t) x := by
  have hγ : ∀ τ, HasDerivAt (fun τ => metricGradientFlow G f (τ + t) x)
      (-G (metricGradientFlow G f (τ + t) x) (gradient f (metricGradientFlow G f (τ + t) x)))
      τ := fun τ =>
    HasDerivAt.comp_add_const τ t (hasDerivAt_metricGradientFlow hL hM (τ + t) x)
  simpa [metricGradientFlow] using (isFlowField_neg_metricGradient hL hM).flow_eq hγ 0 s

/-- The value of the flow at the integer time `k` is the `k`-th iterate of its
time-one map. -/
theorem iterate_metricGradientFlow_one (hL : LipschitzWith L fun x => G x (gradient f x))
    (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M) (k : ℕ) (x : E) :
    (metricGradientFlow G f 1)^[k] x = metricGradientFlow G f k x := by
  induction k with
  | zero =>
    simpa [metricGradientFlow] using ((isFlowField_neg_metricGradient hL hM).flow_self 0 x).symm
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih, metricGradientFlow_metricGradientFlow hL hM,
      Nat.cast_succ, add_comm]

variable [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]

/-- **Flows `ẋ = -G(x) ∇f(x)` almost never converge to a strict saddle.**
For `f` of class `C²` and `G` of class `C¹` with `G ∇f` `L`-Lipschitz and
bounded by `M`, and `G` symmetric positive definite at the strict saddles of
`f`, the starts from which the flow converges to a strict saddle as `t → ∞`
form a null set.  Along the integer times such a trajectory is an orbit of
the time-one map converging to a symmetrizable unstable fixed point of it. -/
theorem measure_tendsto_metricGradientFlow_strictSaddle_eq_zero (μ : Measure E)
    [μ.IsAddHaarMeasure] (hf : ContDiff ℝ 2 f) (hG : ContDiff ℝ 1 G)
    (hL : LipschitzWith L fun x => G x (gradient f x)) (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M)
    (hpos : ∀ z, IsStrictSaddle f z →
      (G z : E →ₗ[ℝ] E).IsSymmetric ∧ ∀ v, v ≠ 0 → 0 < ⟪G z v, v⟫) :
    μ {x | ∃ z, IsStrictSaddle f z ∧
      Tendsto (fun t => metricGradientFlow G f t x) atTop (𝓝 z)} = 0 := by
  refine measure_mono_null ?_ (measure_tendsto_symmetrizableUnstable_eq_zero μ
    (contDiff_metricGradientFlow hf hG hL hM zero_le_one)
    (det_fderiv_metricGradientFlow_ne_zero hf hG hL hM zero_le_one))
  rintro x ⟨z, hz, hlim⟩
  refine ⟨z, hz.isSymmetrizableUnstable_metricGradientFlow hf hG hL hM (hpos z hz).1
    (hpos z hz).2, ?_⟩
  simpa only [Function.comp_def, iterate_metricGradientFlow_one hL hM] using
    hlim.comp tendsto_natCast_atTop_atTop

/-- **The same for a start drawn from a law with a density**: for `ν`-almost
every start, `ν` absolutely continuous with respect to a Haar measure, the
flow `ẋ = -G(x) ∇f(x)` converges to no strict saddle. -/
theorem ae_not_tendsto_metricGradientFlow_of_isStrictSaddle (μ : Measure E)
    [μ.IsAddHaarMeasure] {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f)
    (hG : ContDiff ℝ 1 G) (hL : LipschitzWith L fun x => G x (gradient f x))
    (hM : ∀ x, ‖G x (gradient f x)‖ ≤ M)
    (hpos : ∀ z, IsStrictSaddle f z →
      (G z : E →ₗ[ℝ] E).IsSymmetric ∧ ∀ v, v ≠ 0 → 0 < ⟪G z v, v⟫) :
    ∀ᵐ x ∂ν, ∀ z, IsStrictSaddle f z →
      ¬Tendsto (fun t => metricGradientFlow G f t x) atTop (𝓝 z) := by
  rw [ae_iff]
  refine hν (measure_mono_null (fun x hx => ?_)
    (measure_tendsto_metricGradientFlow_strictSaddle_eq_zero μ hf hG hL hM hpos))
  rw [Set.mem_ofPred_eq] at hx
  push Not at hx
  exact hx

/-- The hypotheses of the theorems above hold for `f = cos` on the line in the
constant field `G = 2`, with `L = M = 2` and the restriction of Lebesgue
measure to the unit ball: `G ∇f = -2 sin` is `2`-Lipschitz and bounded by
`2`, and `G = 2` is symmetric positive definite everywhere. -/
example : let G : ℝ → ℝ →L[ℝ] ℝ := fun _ => (2 : ℝ) • ContinuousLinearMap.id ℝ ℝ
    ContDiff ℝ 2 Real.cos ∧ ContDiff ℝ 1 G ∧
      LipschitzWith 2 (fun x => G x (gradient Real.cos x)) ∧
      (∀ x, ‖G x (gradient Real.cos x)‖ ≤ 2) ∧
      (volume : Measure ℝ).restrict (Metric.ball 0 1) ≪ volume ∧
      ∀ z, IsStrictSaddle Real.cos z →
        (G z : ℝ →ₗ[ℝ] ℝ).IsSymmetric ∧ ∀ v, v ≠ 0 → 0 < ⟪G z v, v⟫ := by
  intro G
  have hgrad : gradient Real.cos = fun x => -Real.sin x :=
    funext fun x => (Real.hasDerivAt_cos x).hasGradientAt'.gradient
  have hGx : ∀ x, G x (gradient Real.cos x) = 2 * -Real.sin x := fun x => by simp [G, hgrad]
  refine ⟨Real.contDiff_cos, contDiff_const, ?_, fun x => ?_,
    Measure.absolutelyContinuous_of_le Measure.restrict_le_self,
    fun z _ => ⟨fun u v => ?_, fun v hv => ?_⟩⟩
  · have h := (lipschitzWith_smul (2 : ℝ)).comp Real.lipschitzWith_sin.neg
    convert h using 1
    · simp
    · funext x
      simp [G, hgrad]
  · rw [hGx, norm_mul, norm_neg, Real.norm_eq_abs, Real.norm_eq_abs, abs_two]
    linarith [Real.abs_sin_le_one x]
  · simp [G]
    ring
  · change 0 < ⟪(2 : ℝ) • v, v⟫
    rw [real_inner_smul_left]
    exact mul_pos two_pos (real_inner_self_pos.2 hv)

end Transformer.Landscape.SaddleAvoidance
