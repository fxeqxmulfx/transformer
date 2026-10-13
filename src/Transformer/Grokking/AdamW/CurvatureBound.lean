import Transformer.Grokking.AdamW.LossDescent
import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Finite descent requires control along the update interval

Source: Mathlib's real mean-value theorem, antitoneOn_of_deriv_nonpos;
comparison with the actual native AdamW loss curves measured at 444b4ad.
The parameter displacement is held fixed while varying its rate. The
current derivative and an upper derivative envelope are explicit inputs;
no transformer or autograd program is assumed to supply either exactly.

Subtracting the stated linear and quadratic terms gives an antitone
remainder. This proves the sharp one-half coefficient without requiring
twice continuous differentiability across every activation boundary.
The envelope covers the whole open update interval, not just its initial
Hessian. A negative starting slope alone does not certify a prescribed
rate. No optimizer, learning rate or original training run is modified.
-/

namespace Transformer.Grokking.AdamW

/-- Concrete quadratic curve used to show sharpness and instantiate all
hypotheses. Source: the Taylor quadratic, and the symmetric MSE curve
comparison in Grokking.Perceptron; no descent is inserted into its definition. -/
noncomputable def quadraticCurve (base slope curvature eta : ℝ) : ℝ :=
  base + slope * eta + curvature * eta ^ 2 / 2

/-- Differentiate the actual quadratic example. Source: real polynomial
calculus in the Taylor/mean-value comparison above; all coefficients remain
explicit and may have either sign. -/
theorem quadraticCurve_deriv (base slope curvature eta : ℝ) :
    HasDerivAt (quadraticCurve base slope curvature) (slope + curvature * eta) eta := by
  have h := (((hasDerivAt_id eta).const_mul slope).const_add base).add
    ((((hasDerivAt_id eta).pow 2).const_mul curvature).div_const 2)
  convert h using 1
  · funext t
    rfl
  · dsimp
    ring

/-- A verified interval envelope gives a finite quadratic upper bound.
Source: antitoneOn_of_deriv_nonpos, applied to the actual loss minus its
stated linear/quadratic terms. This hypothesis is stronger than a sampled
curvature or a Hessian computed only at the starting point. -/
theorem loss_bound_of_derivative_envelope (f d : ℝ → ℝ) (curvature eta : ℝ)
    (heta : 0 ≤ eta)
    (hd : ∀ t ∈ Set.Icc 0 eta, HasDerivAt f (d t) t)
    (hu : ∀ t ∈ Set.Ioo 0 eta, d t ≤ d 0 + curvature * t) :
    f eta ≤ f 0 + d 0 * eta + curvature * eta ^ 2 / 2 := by
  let q := fun t => f t - f 0 - d 0 * t - curvature * t ^ 2 / 2
  have hq (t : ℝ) (ht : t ∈ Set.Icc 0 eta) :
      HasDerivAt q (d t - d 0 - curvature * t) t := by
    have h := (((hd t ht).sub_const (f 0)).sub ((hasDerivAt_id t).const_mul (d 0))).sub
      ((((hasDerivAt_id t).pow 2).const_mul curvature).div_const 2)
    convert h using 1
    · rfl
    · dsimp
      ring
  have hc : ContinuousOn q (Set.Icc 0 eta) := fun t ht =>
    (hq t ht).continuousAt.continuousWithinAt
  have hf : DifferentiableOn ℝ q (interior (Set.Icc 0 eta)) := by
    intro t ht
    rw [interior_Icc] at ht
    exact (hq t ⟨ht.1.le, ht.2.le⟩).differentiableAt.differentiableWithinAt
  have hn : ∀ t ∈ interior (Set.Icc 0 eta), deriv q t ≤ 0 := by
    intro t ht
    rw [interior_Icc] at ht
    rw [(hq t ⟨ht.1.le, ht.2.le⟩).deriv]
    have hb := hu t ht
    linarith
  have ha := antitoneOn_of_deriv_nonpos (convex_Icc 0 eta) hc hf hn
  have h := ha (show (0 : ℝ) ∈ Set.Icc 0 eta from ⟨le_rfl, heta⟩)
    (show eta ∈ Set.Icc 0 eta from ⟨heta, le_rfl⟩) heta
  dsimp [q] at h
  nlinarith

example : 0 ≤ (1 / 4 : ℝ) ∧
    (∀ t ∈ Set.Icc 0 (1 / 4 : ℝ), HasDerivAt (quadraticCurve 1 (-2) 2) (-2 + 2 * t) t) ∧
    (∀ t ∈ Set.Ioo 0 (1 / 4 : ℝ), -2 + 2 * t ≤ (-2 + 2 * 0) + 2 * t) := by
  refine ⟨by norm_num, ?_, ?_⟩
  · intro t _
    exact quadraticCurve_deriv 1 (-2) 2 t
  · intro t _
    norm_num

/-- The interval bound supplies a strict finite decrease when its
quadratic term cannot cancel its linear gain. Source: the preceding
mean-value bound; no current-point curvature is substituted for hu. -/
theorem loss_decreases_of_derivative_envelope (f d : ℝ → ℝ) (curvature eta : ℝ)
    (heta : 0 < eta)
    (hd : ∀ t ∈ Set.Icc 0 eta, HasDerivAt f (d t) t)
    (hu : ∀ t ∈ Set.Ioo 0 eta, d t ≤ d 0 + curvature * t)
    (hr : curvature * eta < -2 * d 0) : f eta < f 0 := by
  have hb := loss_bound_of_derivative_envelope f d curvature eta heta.le hd hu
  have hp : 0 < eta * (-2 * d 0 - curvature * eta) := by positivity
  nlinarith

example : 0 < (1 / 4 : ℝ) ∧
    (∀ t ∈ Set.Icc 0 (1 / 4 : ℝ), HasDerivAt (quadraticCurve 1 (-2) 2) (-2 + 2 * t) t) ∧
    (∀ t ∈ Set.Ioo 0 (1 / 4 : ℝ), -2 + 2 * t ≤ (-2 + 2 * 0) + 2 * t) ∧
    2 * (1 / 4 : ℝ) < -2 * (-2 + 2 * 0) := by
  refine ⟨by norm_num, ?_, ?_, by norm_num⟩
  · intro t _
    exact quadraticCurve_deriv 1 (-2) 2 t
  · intro t _
    norm_num

/-- For a concrete quadratic, the same strict condition is also necessary.
Source: the Taylor quadratic comparison; equality is a loss tie, not a
strict decrease, showing that the bound's one-half coefficient is sharp. -/
theorem quadraticCurve_decreases_iff (base slope curvature eta : ℝ) (heta : 0 < eta) :
    quadraticCurve base slope curvature eta < quadraticCurve base slope curvature 0 ↔
      curvature * eta < -2 * slope := by
  unfold quadraticCurve
  constructor
  · intro h
    by_contra hn
    have hb : -2 * slope ≤ curvature * eta := by linarith
    have hp : 0 ≤ eta * (2 * slope + curvature * eta) :=
      mul_nonneg heta.le (by linarith)
    nlinarith
  · intro h
    have hp : 0 < eta * (-2 * slope - curvature * eta) := by positivity
    nlinarith

example : 0 < (1 / 4 : ℝ) := by norm_num

/-- An everywhere nonincreasing derivative on the update interval makes
every such positive step improve when the initial slope is negative.
Source: the zero-curvature instance of the mean-value bound; this is an
interval premise, not an inference from one negative measured slope. -/
theorem loss_decreases_with_nonincreasing_derivative (f d : ℝ → ℝ) (eta : ℝ)
    (heta : 0 < eta) (hs : d 0 < 0)
    (hd : ∀ t ∈ Set.Icc 0 eta, HasDerivAt f (d t) t)
    (hu : ∀ t ∈ Set.Ioo 0 eta, d t ≤ d 0) : f eta < f 0 := by
  apply loss_decreases_of_derivative_envelope f d 0 eta heta hd
  · intro t ht
    simpa using hu t ht
  · linarith

example : 0 < (1 : ℝ) ∧ (-1 : ℝ) < 0 ∧
    (∀ t ∈ Set.Icc 0 (1 : ℝ), HasDerivAt (quadraticCurve 1 (-1) 0) (-1) t) ∧
    (∀ t ∈ Set.Ioo 0 (1 : ℝ), (-1 : ℝ) ≤ -1) := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  · intro t _
    simpa using quadraticCurve_deriv 1 (-1) 0 t
  · intro t _
    norm_num

end Transformer.Grokking.AdamW
