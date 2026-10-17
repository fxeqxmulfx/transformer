/-
# Saddle avoidance — gradient flows

Let `f` be a `C²` function on a finite-dimensional real inner product space
whose gradient is `L`-Lipschitz and bounded by `M`.  The gradient flow
`ẋ = -∇f(x)` converges to a strict saddle of `f` only from a null set of
starts (`measure_tendsto_gradientFlow_strictSaddle_eq_zero`); started from a
law with a density, it almost surely converges to none
(`ae_not_tendsto_gradientFlow_of_isStrictSaddle`).

The field `-∇f` does not depend on time, so the flow is a one-parameter group
(`gradientFlow_gradientFlow`), and its value at the integer time `k` is the
`k`-th iterate of the time-one map (`iterate_gradientFlow_one`).  A trajectory
converging to `z` converges along the integer times.  The time-one map is
`C¹` with an invertible derivative, and every strict saddle is a symmetric
unstable fixed point of it (`SaddleAvoidance.TimeOne`), so the avoidance
theorem for maps, `measure_tendsto_symmetricUnstable_eq_zero`, concludes.

Gradient ascent `ẋ = ∇f(x)` is the gradient flow of `-f`
(`hasDerivAt_gradientFlow_neg`).  The strict saddles of `-f` are the critical
points of `f` at which the Hessian of `f` has a positive eigenvalue
(`isStrictSaddle_neg`), so ascent converges to such a point only from a null
set of starts (`measure_tendsto_gradientAscent_strictSaddle_eq_zero`): the
Euclidean analog of `l:nosaddleconv` in Appendix A of arXiv:2312.10794v5, whose
proof it follows with `thm:main-general` of §4 of arXiv:1710.07406, applied to
the time-one map, in place of the center-stable manifold theorem for flows.
-/

import Transformer.SaddleAvoidance.TimeOne

open Filter MeasureTheory
open scoped Topology NNReal

namespace Transformer.SaddleAvoidance

open GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  {f : E → ℝ} {L : ℝ≥0} {M : ℝ}

/-- **The gradient flow is a one-parameter group**: flowing for time `t` and
then for time `s` is flowing for time `s + t`.  The field `-∇f` does not depend
on time, so `τ ↦ Φ^{τ + t}(x)` is the trajectory through `Φ^t(x)`. -/
theorem gradientFlow_gradientFlow (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) (s t : ℝ) (x : E) :
    gradientFlow f s (gradientFlow f t x) = gradientFlow f (s + t) x := by
  have hγ : ∀ τ, HasDerivAt (fun τ => gradientFlow f (τ + t) x)
      (-gradient f (gradientFlow f (τ + t) x)) τ := fun τ =>
    HasDerivAt.comp_add_const τ t (hasDerivAt_gradientFlow hL hM (τ + t) x)
  simpa [gradientFlow] using (isFlowField_neg_gradient hL hM).flow_eq hγ 0 s

/-- The value of the gradient flow at the integer time `k` is the `k`-th
iterate of its time-one map. -/
theorem iterate_gradientFlow_one (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) (k : ℕ) (x : E) :
    (gradientFlow f 1)^[k] x = gradientFlow f k x := by
  induction k with
  | zero => simpa [gradientFlow] using ((isFlowField_neg_gradient hL hM).flow_self 0 x).symm
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih, gradientFlow_gradientFlow hL hM, Nat.cast_succ,
      add_comm]

/-- The gradient of `-f` is `-∇f`. -/
theorem gradient_fun_neg (y : E) : gradient (fun y => -f y) y = -gradient f y := by
  simp only [gradient, fderiv_fun_neg, map_neg]

/-- `∇(-f)` is `L`-Lipschitz when `∇f` is. -/
theorem lipschitzWith_gradient_neg (hL : LipschitzWith L (gradient f)) :
    LipschitzWith L (gradient fun y => -f y) := by
  rw [show (gradient fun y => -f y) = fun y => -gradient f y from funext gradient_fun_neg]
  exact hL.neg

/-- **Gradient ascent is the gradient flow of `-f`**: its trajectories solve
`ẋ = ∇f(x)`. -/
theorem hasDerivAt_gradientFlow_neg (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) (t : ℝ) (x : E) :
    HasDerivAt (fun t => gradientFlow (fun y => -f y) t x)
      (gradient f (gradientFlow (fun y => -f y) t x)) t := by
  simpa [gradient_fun_neg] using hasDerivAt_gradientFlow (lipschitzWith_gradient_neg hL)
    (fun y => by rw [gradient_fun_neg, norm_neg]; exact hM y) t x

/-- **The strict saddles of gradient ascent are those of `-f`**: a critical
point of `f` at which the Hessian of `f` has a positive eigenvalue `λ`, a
strict saddle point in the sense of `l:nosaddleconv` of arXiv:2312.10794v5, is
a strict saddle of `-f`, whose Hessian has the eigenvalue `-λ`. -/
theorem isStrictSaddle_neg {z : E} (hz : gradient f z = 0) {lam : ℝ} (hlam : 0 < lam)
    (hev : Module.End.HasEigenvalue (fderiv ℝ (gradient f) z : E →ₗ[ℝ] E) lam) :
    IsStrictSaddle (fun y => -f y) z := by
  have hg : (gradient fun y => -f y) = fun y => -gradient f y := funext gradient_fun_neg
  refine ⟨by rw [gradient_fun_neg, hz, neg_zero], -lam, by linarith, ?_⟩
  rw [hg, fderiv_fun_neg]
  obtain ⟨w, hw⟩ := hev.exists_hasEigenvector
  refine Module.End.hasEigenvalue_of_hasEigenvector (x := w) ⟨?_, hw.2⟩
  have hHw : fderiv ℝ (gradient f) z w = lam • w := hw.apply_eq_smul
  simp [hHw]

variable [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E]

/-- **Gradient flows almost never converge to a strict saddle.**  For `f` of
class `C²` with an `L`-Lipschitz gradient bounded by `M`, the starts from
which the gradient flow `ẋ = -∇f(x)` converges to a strict saddle as `t → ∞`
form a null set.  Along the integer times such a trajectory is an orbit of the
time-one map converging to a symmetric unstable fixed point of it. -/
theorem measure_tendsto_gradientFlow_strictSaddle_eq_zero (μ : Measure E)
    [μ.IsAddHaarMeasure] (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) :
    μ {x | ∃ z, IsStrictSaddle f z ∧ Tendsto (fun t => gradientFlow f t x) atTop (𝓝 z)} = 0 := by
  refine measure_mono_null ?_ (measure_tendsto_symmetricUnstable_eq_zero μ
    (contDiff_gradientFlow hf hL hM zero_le_one)
    (det_fderiv_gradientFlow_ne_zero hf hL hM zero_le_one))
  rintro x ⟨z, hz, hlim⟩
  refine ⟨z, hz.isSymmetricUnstable_gradientFlow hf hL hM, ?_⟩
  simpa only [Function.comp_def, iterate_gradientFlow_one hL hM] using
    hlim.comp tendsto_natCast_atTop_atTop

/-- **The same for a start drawn from a law with a density**: for `ν`-almost
every start, `ν` absolutely continuous with respect to a Haar measure, the
gradient flow converges to no strict saddle. -/
theorem ae_not_tendsto_gradientFlow_of_isStrictSaddle (μ : Measure E) [μ.IsAddHaarMeasure]
    {ν : Measure E} (hν : ν ≪ μ) (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) :
    ∀ᵐ x ∂ν, ∀ z, IsStrictSaddle f z → ¬Tendsto (fun t => gradientFlow f t x) atTop (𝓝 z) := by
  rw [ae_iff]
  refine hν (measure_mono_null (fun x hx => ?_)
    (measure_tendsto_gradientFlow_strictSaddle_eq_zero μ hf hL hM))
  rw [Set.mem_ofPred_eq] at hx
  push Not at hx
  exact hx

/-- **Gradient ascent almost never converges to a strict saddle**: the
Euclidean analog of Lemma `l:nosaddleconv`.  For `f` of class `C²` with an
`L`-Lipschitz gradient bounded by `M`, the starts from which gradient ascent
`ẋ = ∇f(x)`, the gradient flow of `-f`, converges to a critical point of `f`
at which the Hessian has a positive eigenvalue form a null set.

The source: "Let `ℳ` be a compact Riemannian manifold and let `f : ℳ → ℝ` be a
smooth function.  The set of initial conditions `X₀ ∈ ℳ` for which the
gradient ascent `Ẋ(t) = ∇f(X(t))`, `X(0) = X₀` converges to a strict saddle
point of `f` is of volume zero", a strict saddle point being a critical point
at which the Hessian has at least one strictly positive eigenvalue.

**What the source says and what is changed here.**  `ℳ` is a
finite-dimensional real inner product space `E` instead of a compact
manifold, and volume is a Haar measure `μ`.  On the non-compact `E`, the
smooth `f` is replaced by an `f` of class `C²` whose gradient is bounded and
Lipschitz, as on a compact manifold, so that the flow exists for all times.
The proof replaces the center-stable manifold theorem for flows by
`thm:main-general` of §4 of arXiv:1710.07406 for the time-one map.  The
instance of the source on `(𝕊^{d-1})^n` is `Perspective.no_saddle_convergence`.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`. -/
theorem measure_tendsto_gradientAscent_strictSaddle_eq_zero (μ : Measure E)
    [μ.IsAddHaarMeasure] (hf : ContDiff ℝ 2 f) (hL : LipschitzWith L (gradient f))
    (hM : ∀ x, ‖gradient f x‖ ≤ M) :
    μ {x | ∃ z, gradient f z = 0 ∧
      (∃ lam > 0, Module.End.HasEigenvalue (fderiv ℝ (gradient f) z : E →ₗ[ℝ] E) lam) ∧
      Tendsto (fun t => gradientFlow (fun y => -f y) t x) atTop (𝓝 z)} = 0 := by
  refine measure_mono_null ?_ (measure_tendsto_gradientFlow_strictSaddle_eq_zero μ hf.neg
    (lipschitzWith_gradient_neg hL) fun y => by rw [gradient_fun_neg, norm_neg]; exact hM y)
  rintro x ⟨z, hz, ⟨lam, hlam, hev⟩, hlim⟩
  exact ⟨z, isStrictSaddle_neg hz hlam hev, hlim⟩

/-- The hypotheses of the theorems above hold for `f = cos` on the line, with
`L = M = 1`, the restriction of Lebesgue measure to the unit ball, the strict
saddle `0` of descent, a maximum, and the strict saddle `π` of ascent, a
minimum: `∇f = -sin` is `1`-Lipschitz and bounded by `1`, and the Hessian
`-cos` is `-1` at `0` and `1` at `π`. -/
example : ContDiff ℝ 2 Real.cos ∧ LipschitzWith 1 (gradient Real.cos) ∧
    (∀ x, ‖gradient Real.cos x‖ ≤ 1) ∧
    (volume : Measure ℝ).restrict (Metric.ball 0 1) ≪ volume ∧ IsStrictSaddle Real.cos 0 ∧
    gradient Real.cos Real.pi = 0 ∧ (0 : ℝ) < 1 ∧
    Module.End.HasEigenvalue (fderiv ℝ (gradient Real.cos) Real.pi : ℝ →ₗ[ℝ] ℝ) 1 := by
  have hgrad : gradient Real.cos = fun x => -Real.sin x :=
    funext fun x => (Real.hasDerivAt_cos x).hasGradientAt'.gradient
  have hH : ∀ x, fderiv ℝ (gradient Real.cos) x =
      ContinuousLinearMap.smulRight (1 : ℝ →L[ℝ] ℝ) (-Real.cos x) := fun x => by
    rw [hgrad]
    exact (Real.hasDerivAt_sin x).neg.hasFDerivAt.fderiv
  refine ⟨Real.contDiff_cos, ?_, fun x => ?_,
    Measure.absolutelyContinuous_of_le Measure.restrict_le_self,
    ⟨by simp [hgrad], -1, by norm_num, ?_⟩, by simp [hgrad], one_pos, ?_⟩
  · rw [hgrad]
    exact Real.lipschitzWith_sin.neg
  · rw [hgrad, norm_neg, Real.norm_eq_abs]
    exact Real.abs_sin_le_one x
  · refine Module.End.hasEigenvalue_of_hasEigenvector (x := 1) ⟨?_, one_ne_zero⟩
    simp [hH]
  · refine Module.End.hasEigenvalue_of_hasEigenvector (x := 1) ⟨?_, one_ne_zero⟩
    simp [hH]

end Transformer.SaddleAvoidance
